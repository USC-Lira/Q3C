import torch
import torch.nn as nn
import numpy as np

from large_rl.policy.arch.mlp import MLP
from torch.distributions import Normal, Independent


class DDPG_OUNoise(object):
    def __init__(self, dim_action, mu=0.0, theta=0.15, sigma=0.2, env_max_action=1., device="cpu"):
        self.dim_action = dim_action
        self.mu = mu
        self.theta = theta
        self.sigma = sigma * env_max_action
        self.state = torch.ones(self.dim_action, device=device) * self.mu
        self._device = device

    def noise(self, scale):
        x = self.state
        dx = (self.theta * (self.mu - x) + self.sigma * torch.randn(*self.state.shape, device=self._device)) * scale
        self.state = x + dx
        return self.state

    def reset(self, _id=None):
        if _id is not None:
            self.state[_id] = (torch.ones(self.dim_action, device=self._device)[_id]) * self.mu
        else:
            self.state = torch.ones(self.dim_action, device=self._device) * self.mu


class GaussianNoise(object):
    def __init__(self, dim_action, mu=0.0, sigma=0.2, device="cpu", env_max_action=1., **kwargs):
        self.dim_action = dim_action
        self.mu = mu
        self.sigma = sigma * env_max_action
        self._device = device

    def noise(self, scale=1.):
        return torch.normal(mean=self.mu, std=self.sigma, size=self.dim_action, device=self._device) * scale

    def reset(self, id=None):
        pass

class FixedGaussianNoise(GaussianNoise):
    def noise(self, scale=None):
        return super().noise(scale=1.)

class Actor(nn.Module):
    def __init__(self, dim_in=32, dim_hiddens="64_64", dim_out=20, if_init_layer=False, if_norm_each=True,
                 if_norm_final=True, args=None):
        super(Actor, self).__init__()
        self._args = args
        self.model = MLP(dim_in=dim_in, dim_hiddens=dim_hiddens, dim_out=dim_out, type_hidden_act_fn=10,
                         if_init_layer=if_init_layer, if_norm_each=if_norm_each, if_norm_final=if_norm_final)

    def forward(self, inputs, if_joint_update=False, knn_function=None):
        out = self.model(inputs)[:, None, :]
        if self._args["DEBUG_type_activation"] == "tanh":
            out = torch.tanh(out)
            out = out * self._args["env_max_action"]
        elif self._args["DEBUG_type_activation"] == "sigmoid":
            out = torch.sigmoid(out)

        if self._args["WOLP_if_joint_actor"]:  # if update for Wolp-joint then we just output
            if if_joint_update:
                out = out.squeeze(1)  # batch x list-len * dim-action
            else:
                batch, _, mix_dim_action = out.shape
                dim_action = mix_dim_action // self._args["WOLP_cascade_list_len"]
                out = out.reshape(out.shape[0], self._args["WOLP_cascade_list_len"], dim_action)
        return out
    
EPSILON = 1e-6
class GreedyActor(nn.Module):
    def __init__(self, dim_in=32, dim_hiddens="64_64", dim_out=20, if_init_layer=False, if_norm_each=True,
                 if_norm_final=True, args=None):
        super().__init__()
        self._args = args
        self.num_actions = dim_out
        self.dim_out = 2 * dim_out
        self.log_std_min = -np.log(self._args["clip_log_std_threshold"])
        self.log_std_max = np.log(self._args["clip_log_std_threshold"])
        self.action_max = torch.tensor(self._args["env_max_action"]).to(self._args["device"])
        self.action_min = torch.tensor(-self._args["env_max_action"]).to(self._args["device"])
        self.model = MLP(dim_in=dim_in, dim_hiddens=dim_hiddens, dim_out=self.dim_out, type_hidden_act_fn=10,
                         if_init_layer=if_init_layer, if_norm_each=if_norm_each, if_norm_final=if_norm_final)
        
    def forward(self, inputs, if_joint_update=False, knn_function=None):
        mean = self.model(inputs)[:, : self.dim_out // 2]
        log_std = self.model(inputs)[:, self.dim_out // 2:]
        log_std = torch.clamp(log_std,
                                min=self.log_std_min,
                                max=self.log_std_max)
        if self._args["DEBUG_type_activation"] == "tanh":
            mean = torch.tanh(mean)
            mean = mean * self._args["env_max_action"]
        elif self._args["DEBUG_type_activation"] == "sigmoid":
            mean = torch.sigmoid(mean)

        return mean, log_std
    
    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        action = normal.rsample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)
        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        return action, log_prob, mean

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in
        num_samples : int
            The number of actions to sample

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # Non-differentiable
        action = normal.sample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)

        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        # print(action.shape)

        return action, log_prob, mean

    def log_prob(self, states, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        log_prob = normal.log_prob(actions)
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([mean, std], axis=1)[0])

        return log_prob

    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(GreedyActor, self).to(device)
