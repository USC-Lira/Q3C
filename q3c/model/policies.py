from typing import Any, Dict, List, Optional, Type, Union
from gymnasium import spaces

import torch as th
import torch.nn as nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.common.torch_layers import (
   BaseFeaturesExtractor,
   CombinedExtractor,
   FlattenExtractor,
   NatureCNN,
   create_mlp,
)

EPS = 1e-6


class Q3CNetwork(BasePolicy):
    action_space: spaces.Box
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        num_control_points: int = 10,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.num_control_points = num_control_points
        self.action_size = action_space.shape[0]

        # k = total number of control points
        # Output layer represents k control points
        # Each point consists of a u for each dimension of the action space, and Q-value
        self.output_layer_size = self.num_control_points * self.action_size + self.num_control_points

        # The number of values that correspond to the action part of the control points
        self.num_u_values = self.action_size * self.num_control_points

        act_net = create_mlp(self.features_dim, self.num_u_values, self.net_arch, self.activation_fn, squash_output=True) # Tanh activation function on action layer
        q_net = create_mlp(self.features_dim + self.action_size, 1, self.net_arch, self.activation_fn, squash_output=False) # No activation function on output layer!

        self.q_net = th.nn.Sequential(*q_net)
        self.act_net = th.nn.Sequential(*act_net)


    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Calculate the control point values to match action space.
        :param obs: Observation
        :return: The adjusted control points for each observation that define the observation's critic function.
        """
        B = obs.shape[0]
        features = self.extract_features(obs, self.features_extractor)
        control_point_act = self.act_net(features)
        control_point_act_reshaped = control_point_act.view(B, self.num_control_points, self.action_size)
        x = th.cat((features.unsqueeze(1).expand(-1, self.num_control_points, -1), control_point_act_reshaped), dim=-1)
        x = self.q_net(x)                
        x = x.view(B, -1)
        x = th.cat((control_point_act, x), dim=-1)

        return x

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        pass

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                num_control_points=self.num_control_points,
            )
        )
        return data


class Q3CPolicy(BasePolicy):
    q_net_1: Q3CNetwork
    q_net_2: Q3CNetwork
    q_net_target_1: Q3CNetwork
    q_net_target_2: Q3CNetwork
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        num_control_points: int,
        k: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_learnable_smoothing: bool = False,
        smoothing_value: float = 0.1,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
            squash_output=True, # This is important for eval since we are using tanh to squash the actions part of the control values
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.num_control_points = num_control_points
        self.k = k

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "num_control_points": self.num_control_points,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        if use_learnable_smoothing:
            self.use_learnable_smoothing = True
            self.C1 = th.nn.Parameter(th.rand(1, device=self.device), requires_grad=True)
            self.C2 = th.nn.Parameter(th.rand(1, device=self.device), requires_grad=True)
        else:
            self.use_learnable_smoothing = False
            self.C = th.tensor([smoothing_value]).to(self.device)

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        Put the target network into evaluation mode.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.q_net_1 = self.make_q_net()
        self.q_net_2 = self.make_q_net()
        self.q_net_target_1 = self.make_q_net()
        self.q_net_target_2 = self.make_q_net()
        self.q_net_target_1.load_state_dict(self.q_net_1.state_dict())
        self.q_net_target_2.load_state_dict(self.q_net_2.state_dict())
        self.q_net_target_1.set_training_mode(False)
        self.q_net_target_2.set_training_mode(False)

        # Setup optimizer with initial learning rate
        params = list(self.q_net_1.parameters()) + list(self.q_net_2.parameters())
        if self.use_learnable_smoothing:
            params.append(self.C1)
            params.append(self.C2)
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            params,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> Q3CNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return Q3CNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Calculate the control point values to match action space.
        :param obs: Observation
        :return: The adjusted control points for each observation that define the observation's critic function.
        """
        return self.q_net_1(obs)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        Predict the best actions for the given state.
        :param obs: Observation
        :return: The best actions with maximal reward according to local policy.
        """
        network_outputs = self.q_net_1(obs)
        num_u_values = self.q_net_1.action_size * self.num_control_points
        actions = network_outputs[:, :num_u_values].view(
            obs.shape[0], self.num_control_points, self.q_net_1.action_size
        )
        q_values = network_outputs[:, num_u_values:num_u_values + self.num_control_points]
        max_indices = q_values.argmax(dim=1)
        best_actions = actions[th.arange(obs.shape[0]), max_indices]

        return best_actions

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                num_control_points=self.num_control_points,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                use_learnable_smoothing=self.use_learnable_smoothing,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.


        This affects certain modules, such as batch normalisation and dropout.


        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net_1.set_training_mode(mode)
        self.q_net_2.set_training_mode(mode)
        if self.use_learnable_smoothing:
            self.C1.requires_grad = mode
            self.C2.requires_grad = mode
        self.training = mode


MlpPolicy = Q3CPolicy


class CnnPolicy(Q3CPolicy):
    """
    Policy class for Q3C with CNN features extractor.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param use_learnable_smoothing: Whether to use learnable smoothing.
    :param smoothing_value: The initial value of the smoothing parameter.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        num_control_points: int,
        k: int,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        use_learnable_smoothing: bool = False,
        smoothing_value: float = 0.1,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            num_control_points,
            k,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            use_learnable_smoothing,
            smoothing_value,
        )


class MultiInputPolicy(Q3CPolicy):
    """
    Policy class for Q3C to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param use_learnable_smoothing: Whether to use learnable smoothing.
    :param smoothing_value: The initial value of the smoothing parameter.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        num_control_points: int,
        k: int,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        use_learnable_smoothing: bool = False,
        smoothing_value: float = 0.1,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            num_control_points,
            k,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            use_learnable_smoothing,
            smoothing_value,
        )
