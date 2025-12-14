import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
from gymnasium import spaces

import torch as th
import torch.nn as nn
from torch.nn import functional as F

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from q3c.utils.smoothing_decay_functions import ExponentialSmoothingDecay, LinearSmoothingDecay
from q3c.utils.separation_loss_functions import maximin_loss, min_n_pairwise_distances_loss, repulsion_loss
from q3c.model.policies import Q3CPolicy, MlpPolicy, CnnPolicy, MultiInputPolicy, Q3CNetwork


EPS = 1e-6


SelfQ3C = TypeVar("SelfQ3C", bound="Q3C")


class Q3C(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    q_net_1: Q3CNetwork
    q_net_2: Q3CNetwork
    q_net_target_1: Q3CNetwork
    q_net_target_2: Q3CNetwork
    policy: Q3CPolicy

    def __init__(
        self,
        policy: Union[str, Type[Q3CPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        target_update_interval: int = 4,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        use_knn: bool = False,
        use_separation_loss: bool = False,
        separation_loss_function: str = 'repulsion_loss',
        separation_loss_coefficient: float = 0.01,
        use_learnable_smoothing: bool = False,
        normalize_q_values: bool = False,
        smoothing_value: float = 0.1,
        learnable_smoothing_function: str = None,
        smoothing_decay_function: str = 'exponential',
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_update_interval = target_update_interval
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm

        self.use_knn = use_knn
        self.use_separation_loss = use_separation_loss
        self.separation_loss_function = separation_loss_function
        self.separation_loss_coefficient = separation_loss_coefficient
        self.use_learnable_smoothing = use_learnable_smoothing
        self.normalize_q_values = normalize_q_values
        self.learnable_smoothing_function = learnable_smoothing_function
        self.smoothing_decay_function = smoothing_decay_function
        self.smoothing_value = smoothing_value

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net_1, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target_1, ["running_"])
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )
        if self.use_learnable_smoothing:
            self.smoothing_decay = None
            if self.learnable_smoothing_function == "sigmoid":
                self.smoothing_function = F.sigmoid
            elif self.learnable_smoothing_function == "softplus":
                self.smoothing_function = F.softplus
            else:
                raise ValueError(f"Invalid smoothing function: {self.learnable_smoothing_function}")
        else:
            self.smoothing_function = nn.Identity()
            if self.smoothing_decay_function is None:
                self.smoothing_decay = None
            elif self.smoothing_decay_function == "linear":
                self.smoothing_decay = LinearSmoothingDecay(initial_value=self.smoothing_value, min_value=0, device=self.device)
            elif self.smoothing_decay_function == "exponential":
                self.smoothing_decay = ExponentialSmoothingDecay(initial_value=self.smoothing_value, min_value=0, device=self.device)
            else:
                raise ValueError(f"Invalid smoothing decay: {self.smoothing_decay_function}")

    def _create_aliases(self) -> None:
        self.q_net_1 = self.policy.q_net_1
        self.q_net_2 = self.policy.q_net_2
        self.q_net_target_1 = self.policy.q_net_target_1
        self.q_net_target_2 = self.policy.q_net_target_2
        if self.use_learnable_smoothing:
            self.C1 = self.policy.C1
            self.C2 = self.policy.C2
        else:
            self.C1 = self.policy.C
            self.C2 = self.policy.C

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net_1.parameters(), self.q_net_target_1.parameters(), self.tau)
            polyak_update(self.q_net_2.parameters(), self.q_net_target_2.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        # Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)

        if not self.use_learnable_smoothing:
            if self.smoothing_decay is not None:
                # Calculate the new C value from the decay schedule
                new_C = self.smoothing_decay(self.num_timesteps, self.total_timesteps)
                # Since self.C1 and self.C2 are references to self.policy.C, this updates all three
                self.policy.C.data.copy_(new_C)

        losses = []
        separation_losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            num_u_values = self.q_net_1.num_u_values
            batch_size = replay_data.observations.shape[0]

            with th.no_grad():
                next_values = self.q_net_target_1(replay_data.next_observations)
                next_q_values = next_values[:, num_u_values:num_u_values + self.policy.num_control_points]
                next_actions = next_values[:, :num_u_values]
                # Reshape next_actions to [batch_size, num_control_points, action_dim]
                next_actions = next_actions.view(next_actions.shape[0], self.policy.num_control_points, -1)
                max_control_point_indices = next_q_values.argmax(dim=1, keepdim=True)
                best_next_action = next_actions.gather(
                    1, max_control_point_indices.unsqueeze(-1).expand(-1, -1, next_actions.size(-1))
                ).squeeze(1)

                noise = th.zeros_like(best_next_action).normal_(0, self.target_policy_noise)
                noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
                noise_added_action = best_next_action + noise
                noise_added_action.clamp_(-1, 1)
                next_q_values_target_1, _, _ = self.calculate_arbitrary_q_value(self.q_net_target_1, replay_data.next_observations, noise_added_action, C=self.C1)
                next_q_values_target_2, _, _ = self.calculate_arbitrary_q_value(self.q_net_target_2, replay_data.next_observations, noise_added_action, C=self.C2)
                Q_targets_next = th.min(next_q_values_target_1, next_q_values_target_2)

            # Compute Q targets for current states
            Q_targets = replay_data.rewards + (self.gamma * Q_targets_next * (1 - replay_data.dones))
            
            loss = 0
            for idx, q_net in enumerate((self.q_net_1, self.q_net_2)):
                C = self.C1 if idx == 0 else self.C2
                Q_expected, network_outputs, _ = self.calculate_arbitrary_q_value(q_net, replay_data.observations, replay_data.actions, C=C)
                loss += 0.5 * F.mse_loss(Q_expected, Q_targets.detach())
                # If separation loss is used, add control point pairwise distances to loss to encourage even
                # distribution of control points
                if self.use_separation_loss:
                    # control points (B, num_control_points, action_size)
                    control_points = network_outputs[:, :num_u_values].view(batch_size, self.policy.num_control_points, -1)
                    # Calculate separation loss based on type of function
                    if self.separation_loss_function == 'min_n_pair_distances':
                        separation_loss = min_n_pairwise_distances_loss(control_points) * self.separation_loss_coefficient
                    elif self.separation_loss_function == 'maximin':
                        separation_loss = maximin_loss(control_points) * self.separation_loss_coefficient
                    elif self.separation_loss_function == 'repulsion_loss':
                        separation_loss = repulsion_loss(control_points) * self.separation_loss_coefficient

                    separation_losses.append(separation_loss.item())
                    loss += 0.5 * separation_loss

                # Store losses for each local network for logging
                losses.append(loss.item())

            # Minimize the loss
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.q_net_1.parameters(), self.max_grad_norm
            )
            th.nn.utils.clip_grad_norm_(
                self.q_net_2.parameters(), self.max_grad_norm
            )
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        if self.use_separation_loss:
            self.logger.record("train/separation_loss", np.mean(separation_losses))
        self.logger.record("train/C", np.mean(self.smoothing_function(self.C1.detach()).cpu().numpy()))

    def learn(
        self: SelfQ3C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Q3C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfQ3C:
        self.total_timesteps = total_timesteps  # Set the total_timesteps
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    def calculate_arbitrary_q_value(self, q_net, obs, act, C):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        num_u_values = q_net.num_u_values
        batch_size = obs.size(0)

        network_outputs = q_net(obs)
        q_values = network_outputs[
            :, num_u_values : num_u_values + self.policy.num_control_points
        ]
        max_q = q_values.max(1)[0].unsqueeze(1)  # detach?

        inverse_distance_weights = th.zeros(
            batch_size, self.policy.num_control_points
        )
        # Reshape control_points_local to be (batch_size, num_control_points, action_size)
        control_points = network_outputs[:, :num_u_values]
        control_points = control_points.view(
            batch_size, self.policy.num_control_points, -1
        )
        actions_expanded = act.unsqueeze(1).expand(
            -1, self.policy.num_control_points, -1
        )

        squared_distances = (
            control_points - actions_expanded
        ) ** 2
        squared_sum_distances = th.sum(squared_distances, dim=2)            

        # Calculate the full inverse distance weight: 1/(dist^2 + eps + (maxk - y) * c)
        q_value_difference = max_q - q_values
        if self.normalize_q_values:
            min_q = q_values.min(1)[0].unsqueeze(1)
            normalized_q_values = (q_values - min_q) / (max_q - min_q + EPS)
            normalized_max_q = normalized_q_values.max(1)[0].unsqueeze(1)
            q_value_difference = normalized_max_q - normalized_q_values
        inverse_distance_weights = 1.0 / (th.clamp(squared_sum_distances, min=EPS) + EPS + (self.smoothing_function(C) * q_value_difference))

        if self.use_knn:
            # Only consider the k nearest neighbours
            _, knn_indices = th.topk(inverse_distance_weights, self.policy.k, largest=True)
            knn_weights = inverse_distance_weights[th.arange(batch_size).unsqueeze(1), knn_indices]
            knn_q_values = q_values[th.arange(batch_size).unsqueeze(1), knn_indices]
            Q_expected = (
                th.sum(knn_q_values * knn_weights, dim=1)
                / th.sum(knn_weights, dim=1)
            ).unsqueeze(1)
        else:
            Q_expected = (
                th.sum(q_values * inverse_distance_weights, dim=1)
                / th.sum(inverse_distance_weights, dim=1)
            ).unsqueeze(1)

        return Q_expected, network_outputs, inverse_distance_weights

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        return super().predict(observation, state, episode_start, deterministic)
