""" Cascaded Critics operates on the discrete candidates """

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_ as clip_grad
from copy import deepcopy
import os.path as osp
import os

from large_rl.embedding.base import BaseEmbedding
from large_rl.policy.arch.ddpg import Actor, DDPG_OUNoise
from large_rl.policy.arch.mlp import MLP
from large_rl.policy.agent import Agent
from large_rl.commons.utils import logging


class DDPG(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dim_hidden = self._args.get('dim_hidden', 64)

        # === Actor ===
        self._actor_dim_out = self._args["reacher_action_shape"]
        # before simplifying
        # self.main_actor = Actor(dim_in=self._args["reacher_obs_space"],
        #                         dim_hidden=dim_hidden,
        #                         dim_out=self._actor_dim_out,
        #                         args=self._args, if_norm=self._args["WOLP_if_actor_norm"],
        #                         ).to(device=self._device)
        self.main_actor = Actor(dim_in=self._args["reacher_obs_space"],
                                dim_hiddens=self._args["WOLP_actor_dim_hiddens"],
                                if_init_layer=self._args["WOLP_if_actor_init_layer"],
                                if_norm_each=self._args["WOLP_if_actor_norm_each"],
                                if_norm_final=self._args["WOLP_if_actor_norm_final"],
                                dim_out=self._actor_dim_out,
                                args=self._args
                                ).to(device=self._device)
        self.target_actor = deepcopy(self.main_actor)
        self.opt_actor = torch.optim.Adam(params=self.main_actor.parameters(), lr=self._args["WOLP_actor_lr"])

        # === Critic ===
        self._critic_dim_state = self._args["reacher_obs_space"]
        self._critic_dim_action = self._args["reacher_action_shape"]

        # before simplifying
        # self.main_critic = Critic(args=self._args,
        #                           dim_state=self._critic_dim_state,
        #                           dim_hidden=dim_hidden,
        #                           dim_action=self._critic_dim_action,
        #                           if_norm=self._args["WOLP_if_critic_norm"]
        #                           ).to(device=self._device)
        self.main_critic = MLP(dim_in=self._critic_dim_state + self._critic_dim_action,
                               dim_hiddens=self._args["Qnet_dim_hidden"],
                               dim_out=1,
                               if_norm_each=self._args["WOLP_if_critic_norm_each"],
                               if_norm_final=self._args["WOLP_if_critic_norm_final"],
                               if_init_layer=self._args["WOLP_if_critic_init_layer"],
                               type_hidden_act_fn=10).to(device=self._device)
        self.target_critic = deepcopy(self.main_critic)
        self.opt_critic = torch.optim.Adam(params=self.main_critic.parameters(), lr=self._args["WOLP_critic_lr"])
        self.main_joint_critic = self.target_joint_critic = self.opt_joint_critic = None
        self.main_ar_critic = self.target_ar_critic = self.opt_ar_critic = None

        # === OU Noise ===
        self.noise_sampler = DDPG_OUNoise(dim_action=(self._args["num_envs"], self._actor_dim_out),
                                          device=self._args["device"])
        # self.noise_sampler = OUNoise(dim_action=self._actor_dim_out, device=self._args["device"])

        logging(self.main_actor)
        logging(self.main_critic)

    def _save(self, save_dir):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        logging("Save the agent: {}".format(save_dir))
        ## Actor
        torch.save(self.main_actor.state_dict(), os.path.join(save_dir, f"main_actor.pkl"))
        torch.save(self.target_actor.state_dict(), os.path.join(save_dir, f"target_actor.pkl"))
        torch.save(self.opt_actor.state_dict(), os.path.join(save_dir, f"opt_actor.pkl"))

        ## Critic
        torch.save(self.main_critic.state_dict(), os.path.join(save_dir, f"main_critic.pkl"))
        torch.save(self.target_critic.state_dict(), os.path.join(save_dir, f"target_critic.pkl"))
        torch.save(self.opt_critic.state_dict(), os.path.join(save_dir, f"opt_critic.pkl"))

    def _load(self, load_dir):
        logging("Load the agent: {}".format(load_dir))
        ## Actor
        self.main_actor.load_state_dict(torch.load(os.path.join(load_dir, f"main_actor.pkl")))
        self.target_actor.load_state_dict(torch.load(os.path.join(load_dir, f"target_actor.pkl")))
        self.opt_actor.load_state_dict(torch.load(os.path.join(load_dir, f"opt_actor.pkl")))

        ## Critic
        self.main_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_critic.pkl")))
        self.target_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_critic.pkl")))
        self.opt_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_critic.pkl")))

    def select_random_action(self, batch_size: int, slate_size: int = None, candidate_lists=None):
        assert self._args["env_name"].startswith("mujoco")
        return np.array([self._args["reacher_action_space"].sample() for _ in range(batch_size)])

    def _add_noise(self, mu, **kwargs):
        # Add noise
        if not kwargs.get("if_update", False):
            eps = self.noise_sampler.noise(scale=kwargs["epsilon"]["actor"])
            mu += eps
            if self._args["DEBUG_type_clamp"] == "small":
                mu = mu.clamp(0, 1)  # Constraint the range
            elif self._args["DEBUG_type_clamp"] == "large":
                mu = mu.clamp(-1, 1)  # Constraint the range
        return mu

    def select_action(self, obs: torch.tensor, act_embed_base: BaseEmbedding, epsilon: dict, **kwargs):
        if epsilon["actor"] >= 1.0:
            action_random = self.select_random_action(batch_size=obs.shape[0])
            action = action_random
        else:
            input_dict = {"state_embed": torch.tensor(obs, dtype=torch.float32, device=self._device)}
            action_policy = self._select_action(input_dict=input_dict, epsilon=epsilon, **kwargs)
            action = action_policy

        # Construct the Agent response
        self.res = {"action": np.asarray(action),
                    "max_index": np.zeros(shape=(obs.shape[0],), dtype=np.int32),
                    "max_Q_improvement": np.zeros(shape=(obs.shape[0],), dtype=np.float32),
        }

        return self.res

    def _select_action(self, input_dict: dict, **kwargs):
        # When called from update method, we can switch to Actor Target
        actor = self.target_actor if kwargs.get("if_update", False) else self.main_actor
        critic = self.target_critic if kwargs.get("if_update", False) else self.main_critic

        # Get the proto-action from Actor
        _s = input_dict["state_embed"]
        mu = actor(_s.float()).squeeze(1)  # batch x num_query(or sequence-len) x dim_act
        if not kwargs.get('if_update', False):
            mu = self._add_noise(mu=mu, **kwargs)

        Q_vals = critic(torch.cat(tensors=[_s.float(), mu], dim=-1))
        if kwargs.get("if_get_Q_vals", False):
            return Q_vals.cpu().detach().numpy()

        return mu.cpu().detach().numpy()

    def _update(self,
                input_dict: dict,
                next_input_dict: dict,
                actions: np.ndarray,
                rewards: torch.tensor,
                reversed_dones: torch.tensor,
                act_embed_base: BaseEmbedding,
                **kwargs):

        # === Get next Q-vals
        with torch.no_grad():
            next_Q = self._select_action(input_dict=next_input_dict, epsilon={"actor": 0.0, "critic": 0.0},
                                         if_get_Q_vals=True, if_update=True)

        # ==== Get Taken Q-vals
        Q = self.main_critic(
            torch.cat(tensors=[input_dict['state_embed'].float(),
                               torch.tensor(actions, device=self._device, dtype=torch.float32)], dim=-1))

        # === Bellman Error
        target = rewards + torch.tensor(next_Q, device=self._device) * reversed_dones * self._args["Qnet_gamma"]
        value_loss = torch.nn.functional.mse_loss(Q, target)

        self.opt_critic.zero_grad()
        value_loss.backward()
        if self._args["if_grad_clip"]: clip_grad(self.main_critic.parameters(), 1.0)  # gradient clipping
        self.opt_critic.step()

        # === Update Actor
        _s = input_dict["state_embed"]
        mu = self.main_actor(_s)
        Q = self.main_critic(torch.cat(tensors=[_s[:, None, :].repeat(1, mu.shape[1], 1), mu], dim=-1))
        policy_loss = -Q.sum(dim=-1).mean()  # Average over samples & preserve contribution of slate indices

        self.opt_actor.zero_grad()
        policy_loss.backward()
        if self._args["if_grad_clip"]: clip_grad(self.main_actor.parameters(), 1.0)  # gradient clipping
        self.opt_actor.step()

        self.res = {
            "value_loss": value_loss.item(),  # "next_Q_var": _var, "next_Q_mean": _mean,
            "policy_loss": policy_loss.item() if self.main_actor is not None else 0.0,
            "loss": value_loss.item() + policy_loss.item() if self.main_actor is not None else value_loss.item(),
        }

        return self.res

    def _sync(self, tau: float = 0.0):
        """ when _type is None, we update both actor and critic """
        if tau > 0.0:  # Soft update of params
            for param, target_param in zip(self.main_actor.parameters(),
                                           self.target_actor.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            for param, target_param in zip(self.main_critic.parameters(), self.target_critic.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
        else:
            self.target_critic.load_state_dict(self.main_critic.state_dict())
            self.target_actor.load_state_dict(self.main_actor.state_dict())

    def _train(self):
        self.main_actor.train()
        self.target_actor.train()
        self.main_critic.train()
        self.target_critic.train()

    def _eval(self):
        self.main_actor.eval()
        self.target_actor.eval()
        self.main_critic.eval()
        self.target_critic.eval()

    def _reset(self, **kwargs):
        if "id" in kwargs:
            self.noise_sampler.reset(kwargs["id"])
        else:
            self.noise_sampler.reset()
