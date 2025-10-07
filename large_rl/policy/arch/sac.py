import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from copy import deepcopy

from large_rl.policy.arch.mlp import MLP, LayerNorm

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    """ Twin Delayed Critic """

    def __init__(self, args: dict, dim_state: int, dim_action: int, dim_hidden: int = 32):
        super(Critic, self).__init__()

        self._if_dqn = args["agent_type"].lower() == "dqn"

        # State dependent Act Enc for DQN
        if self._if_dqn:
            self.act_enc = MLP(dim_in=dim_state + dim_action,
                               dim_hiddens=args["agent_action_encoder_layers"],
                               dim_out=args["agent_action_encoder_dim_out"],
                               type_hidden_act_fn=10).to(device=args["device"])

        self.Q1 = MLP(
            dim_in=dim_hidden + args["agent_action_encoder_dim_out"] if self._if_dqn else dim_state + dim_action,
            dim_hiddens=args["Qnet_dim_hidden"],
            type_hidden_act_fn=10,  # RELU
            dim_out=1
        ).to(device=args["device"])
        self.Q2 = deepcopy(self.Q1)

        self.apply(weights_init_)

    def forward(self, state, action, if_cache=False):
        # State dependent Action encoding for DQN: (s, a) -> Act Enc -> A
        if self._if_dqn:
            action = self.act_enc(torch.cat(tensors=[state, action], dim=-1))

        _in = torch.cat([state, action], dim=-1)
        q1 = self.Q1(_in, if_cache=if_cache)
        q2 = self.Q2(_in, if_cache=if_cache)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, dim_hidden: int, args: dict):
        super(Actor, self).__init__()

        self._args = args

        self.linear1 = nn.Linear(dim_state, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)

        self.mean_linear = nn.Linear(dim_hidden, dim_action)
        self.log_std_linear = nn.Linear(dim_hidden, dim_action)
        self.apply(weights_init_)

    def forward(self, state):
        # Forward
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        # Sampling
        std = log_std.exp()
        self._mean, self._std = mean.detach().cpu().numpy(), std.detach().cpu().numpy()  # for visualisation purpose
        normal = Normal(mean, std)

        # Reparametrisation trick applies here
        x_t = normal.rsample(sample_shape=torch.tensor([self._args["WOLP_cascade_list_len"]]))  # sample x batch x act
        if self._args["DEBUG_type_activation"] == "tanh":
            action = torch.tanh(x_t)
        elif self._args["DEBUG_type_activation"] == "sigmoid":
            action = torch.sigmoid(x_t)
        else:
            action = x_t
        action = action.permute([1, 0, 2])  # batch x sample x dim-action
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.permute([1, 0, 2])  # batch x sample x dim-action

        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        if self._args["DEBUG_type_activation"] == "tanh":
            mean = torch.tanh(mean)
        elif self._args["DEBUG_type_activation"] == "sigmoid":
            mean = torch.sigmoid(mean)
        return action, log_prob, mean


def _test():
    b = 16
    dim_state = dim_action = 10
    dim_hidden = 16
    for _type in ["tanh", "sigmoid", "none"]:
        args = {"DEBUG_type_activation": _type, "WOLP_cascade_list_len": 5}
        actor = Actor(dim_state=dim_state, dim_action=dim_action, dim_hidden=dim_hidden, args=args)
        state = torch.randn(b, dim_state)
        action, log_prob, mean = actor(state)
        # print(action.shape)
        print(_type, action.min().item(), action.max().item())


if __name__ == '__main__':
    _test()
