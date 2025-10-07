""" SAC based wolpertinger """

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_ as clip_grad
from copy import deepcopy
import os.path as osp
import os

from large_rl.embedding.base import BaseEmbedding
from large_rl.policy.arch.sac import Critic, Actor
from large_rl.policy.wolp import WOLP as Base
from large_rl.commons.utils import logging


class WOLP(Base):
    def __init__(self, **kwargs):
        super(WOLP, self).__init__(**kwargs)

        self._distance_list, self._activate_list = list(), list()
        self._query = self._topk_act = None  # this is for CDDPG

        dim_hidden = self._args.get('dim_hidden', 64)

        # === Actor ===
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self.main_actor_obs_enc = self._instantiate_obs_encoder()
            self.target_actor_obs_enc = deepcopy(self.main_actor_obs_enc)
            self.opt_actor_obs_enc = torch.optim.Adam(self.main_actor_obs_enc.parameters(),
                                                      lr=self._args["mw_obs_enc_lr"])
            logging(self.main_actor_obs_enc)
        else:
            self.main_actor_obs_enc = self.target_actor_obs_enc = self.opt_actor_obs_enc = None

        # Prep the input dim
        if self._args["env_name"].lower().startswith("recsim"):
            self._dim_in = self._args["recsim_dim_embed"]
        elif self._args["env_name"].lower() == "mine":
            if not self._args['mw_obs_flatten']:
                self._dim_in = self.main_actor_obs_enc.dim_out
            else:
                self._dim_in = self._args["mw_obs_length"]

        if self._args["env_name"].lower().startswith("recsim"):
            self._actor_dim_out = self._args["recsim_dim_tsne_embed"] if self._args["recsim_if_tsne_embed"] \
                else self._args["recsim_dim_embed"]
        elif self._args["env_name"].lower() == "mine":
            self._actor_dim_out = self._args["mw_action_dim"]
        self._actor_dim_out += self._args["env_dim_extra"] if not self._args["env_act_emb_tSNE"] else 0
        self.main_actor = Actor(dim_state=self._dim_in,
                                dim_hidden=dim_hidden,
                                dim_action=self._actor_dim_out,
                                args=self._args).to(device=self._device)
        self.target_actor = deepcopy(self.main_actor)
        self.opt_actor = torch.optim.Adam(params=self.main_actor.parameters(), lr=1e-4)

        self.alpha = torch.tensor(0.2, device=self._device)
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self._args["WOLP_if_auto_ent_tune"]:
            self.target_entropy = -torch.prod(torch.Tensor(self._actor_dim_out).to(self._device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4)

        # === Critic ===
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self.main_critic_obs_enc = self._instantiate_obs_encoder()
            self.target_critic_obs_enc = deepcopy(self.main_critic_obs_enc)
            self.opt_critic_obs_enc = torch.optim.Adam(self.main_critic_obs_enc.parameters(),
                                                       lr=self._args["mw_obs_enc_lr"])
            logging(self.main_critic_obs_enc)
        else:
            self.main_critic_obs_enc = self.target_critic_obs_enc = self.opt_critic_obs_enc = None

        if self._args["env_name"].startswith("recsim"):
            self._critic_dim_state = self._args["recsim_dim_embed"]
            self._critic_dim_action = self._actor_dim_out
        elif self._args["env_name"].lower() == "mine":
            if not self._args['mw_obs_flatten']:
                self._critic_dim_state = self.main_critic_obs_enc.dim_out
            else:
                self._critic_dim_state = self._args["mw_obs_length"]
            self._critic_dim_action = self._actor_dim_out
        else:
            self._critic_dim_state = None
            self._critic_dim_action = None

        self.main_critic = Critic(args=self._args,
                                  dim_state=self._critic_dim_state,
                                  dim_hidden=dim_hidden,
                                  dim_action=self._critic_dim_action).to(device=self._device)
        self.target_critic = deepcopy(self.main_critic)
        self.opt_critic = torch.optim.Adam(params=self.main_critic.parameters(), lr=1e-3)

        logging(self.main_actor)
        logging(self.main_critic)

    def _save(self, save_dir):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        logging("Save the agent: {}".format(save_dir))
        # === Actor ===
        if self._args['env_name'] == "mine" and not self._args['mw_obs_flatten']:
            torch.save(self.main_actor_obs_enc.state_dict(), os.path.join(save_dir, f"main_actor_obs_enc.pkl"))
            torch.save(self.target_actor_obs_enc.state_dict(), os.path.join(save_dir, f"target_actor_obs_enc.pkl"))
            torch.save(self.opt_actor_obs_enc.state_dict(), os.path.join(save_dir, f"opt_actor_obs_enc.pkl"))
        torch.save(self.main_actor.state_dict(), os.path.join(save_dir, f"main_actor.pkl"))
        torch.save(self.target_actor.state_dict(), os.path.join(save_dir, f"target_actor.pkl"))
        torch.save(self.opt_actor.state_dict(), os.path.join(save_dir, f"opt_actor.pkl"))

        # === Critic ===
        if self._args['env_name'] == "mine" and not self._args['mw_obs_flatten']:
            torch.save(self.main_critic_obs_enc.state_dict(), os.path.join(save_dir, f"main_critic_obs_enc.pkl"))
            torch.save(self.target_critic_obs_enc.state_dict(), os.path.join(save_dir, f"target_critic_obs_enc.pkl"))
            torch.save(self.opt_critic_obs_enc.state_dict(), os.path.join(save_dir, f"opt_critic_obs_enc.pkl"))
        torch.save(self.main_critic.state_dict(), os.path.join(save_dir, f"main_critic.pkl"))
        torch.save(self.target_critic.state_dict(), os.path.join(save_dir, f"target_critic.pkl"))
        torch.save(self.opt_critic.state_dict(), os.path.join(save_dir, f"opt_critic.pkl"))

    def _load(self, load_dir):
        logging("Load the agent: {}".format(load_dir))
        # === Actor ===
        if self._args['env_name'] == "mine" and not self._args['mw_obs_flatten']:
            self.main_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"main_actor_obs_enc.pkl")))
            self.target_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"target_actor_obs_enc.pkl")))
            self.opt_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"opt_actor_obs_enc.pkl")))
        self.main_actor.load_state_dict(torch.load(os.path.join(load_dir, f"main_actor.pkl")))
        self.target_actor.load_state_dict(torch.load(os.path.join(load_dir, f"target_actor.pkl")))
        self.opt_actor.load_state_dict(torch.load(os.path.join(load_dir, f"opt_actor.pkl")))

        # === Critic ===
        if self._args['env_name'] == "mine" and not self._args['mw_obs_flatten']:
            self.main_critic_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"main_critic_obs_enc.pkl")))
            self.target_critic_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"target_critic_obs_enc.pkl")))
            self.opt_critic_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"opt_critic_obs_enc.pkl")))

        self.main_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_critic.pkl")))
        self.target_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_critic.pkl")))
        self.opt_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_critic.pkl")))

    def _select_action(self, input_dict: dict, **kwargs):
        _batch = input_dict["state_embed"].shape[0]
        # When called from update method, we can switch to Actor Target
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            actor_encoder = self.target_actor_obs_enc if kwargs.get("if_update", False) else self.main_actor_obs_enc
            critic_encoder = self.target_critic_obs_enc if kwargs.get("if_update", False) else self.main_critic_obs_enc
        else:
            actor_encoder = None
            critic_encoder = None

        actor = self.target_actor if kwargs.get("if_update", False) else self.main_actor
        critic = self.target_critic if kwargs.get("if_update", False) else self.main_critic

        # Get the weight from Actor; wolp-sac always samples from the distribution
        if self._args["env_name"].lower() in ["mine", "recsim-data"]:
            if not self._args['mw_obs_flatten']:
                actor_state = actor_encoder(input_dict["state_embed"])
                actor_state = actor_state.squeeze(1)
            else:
                actor_state = input_dict["state_embed"]
        elif self._args["env_name"].lower() == "recsim":
            actor_state = input_dict["state_embed"]
        else:
            actor_state = None
        action, log_pi, _ = actor(actor_state)  # batch_size x dim_act
        self._query = None if kwargs.get("if_update", False) else action.cpu().detach().numpy().astype(np.float32)

        # Compute the scores of all actions and perform k-NN to work out the k-candidates
        topk_act_2D, topk_act_3D = self._perform_kNN(act_embed=input_dict["act_embed"], mu=action,
                                                     _top_k=self._args["WOLP_topK"])

        # Store the topk_act from Actor to construct the element-wise reward in list-action
        self._topk_act = topk_act_3D.cpu().detach().numpy()
        topk_embed = input_dict["act_embed"].gather(dim=1, index=topk_act_2D[..., None].repeat(1, 1, input_dict[
            "act_embed"].shape[-1]))

        ## Critic
        if self._args["env_name"].lower() in ["mine", "recsim-data"]:
            if not self._args['mw_obs_flatten']:
                critic_state = critic_encoder(input_dict["state_embed"])
                critic_state = critic_state.squeeze(1)
            else:
                critic_state = input_dict["state_embed"]
        elif self._args["env_name"].startswith("recsim"):
            critic_state = input_dict["state_embed"]
        else:
            critic_state = None
        _s = critic_state[:, None, :].repeat(1, topk_embed.shape[1], 1)
        Q_vals = critic(_s, topk_embed, if_cache=not kwargs.get("if_update", False))  # Cache oracle act emb

        # if (not self._if_train) and (not kwargs.get("if_update", False)):  # this didn't take that long in eval!
        #     self._candidate_Q_mean = Q_vals.mean(dim=-1).cpu().detach().numpy()
        # else:
        self._candidate_Q_mean = 0.0

        if kwargs.get("if_get_Q_vals", False):
            _topK = torch.topk(torch.min(*Q_vals).squeeze(-1), k=1)  # batch x 1
            log_pi = log_pi.gather(dim=1, index=_topK.indices[..., None]).squeeze(-1)
            return _topK.values - self.alpha * log_pi  # batch x 1
        else:
            res = torch.topk(torch.max(*Q_vals).squeeze(-1), k=self._args["recsim_slate_size"])  # batch x slate_size

            # Get the indices from the original index list
            a = topk_act_2D.gather(dim=-1, index=res.indices).cpu().detach().numpy().astype(np.int64)

            if self._args["WOLP_if_dual_exploration"]:
                topk_act = topk_act_2D.cpu().detach().numpy().astype(np.int64)

                # Eps-decay for Refine-Q
                _mask = self._rng.uniform(low=0.0, high=1.0, size=_batch) < kwargs["epsilon"]["critic"]
                self._selectionQ_mask = _mask
                if sum(_mask) != 0:
                    a[_mask] = self.select_random_action(batch_size=sum(_mask), candidate_lists=topk_act[_mask])
            return a

    def _update(self,
                input_dict: dict,
                next_input_dict: dict,
                actions: np.ndarray,
                rewards: torch.tensor,
                reversed_dones: torch.tensor,
                act_embed_base: BaseEmbedding, **kwargs):

        if kwargs["if_ret"]:
            actions, ind_actions, topk_acts, list_actions, list_rewards, selection_bonus = self._prep_update(
                actions=actions, rewards=rewards)

        if kwargs["if_sel"]:
            # === Get next Q-vals
            with torch.no_grad():
                next_Q = self._select_action(input_dict=next_input_dict, if_get_Q_vals=True, if_update=True)

            # ==== Get Taken Q-vals
            # Get the associated item-embedding for actions(batch x slate_size)
            actions = torch.tensor(actions, device=self._device)  # Actions are in itemId instead of indices of Q-net
            action_embed = act_embed_base.get(index=actions).detach()  # batch x slate_size x dim_act

            assert self._args["recsim_slate_size"] == 1
            slate_embed = action_embed[:, 0, :]

            if self._args["env_name"].lower() in ["mine", "recsim-data"]:
                if not self._args['mw_obs_flatten']:
                    critic_state = self.main_critic_obs_enc(input_dict["state_embed"])
                    critic_state = critic_state.squeeze(1)
                else:
                    critic_state = input_dict["state_embed"]
            elif self._args["env_name"].startswith("recsim"):
                critic_state = input_dict["state_embed"]
            else:
                critic_state = None

            Q1, Q2 = self.main_critic(critic_state, slate_embed)

            # === Bellman Error
            target = rewards + next_Q * reversed_dones * self._args["Qnet_gamma"]
            q1_loss = torch.nn.functional.mse_loss(Q1, target)
            q2_loss = torch.nn.functional.mse_loss(Q2, target)
            value_loss = q1_loss + q2_loss

            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.opt_critic_obs_enc.zero_grad()
            self.opt_critic.zero_grad()

            value_loss.backward()

            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                if self._args["if_grad_clip"]: clip_grad(self.main_critic_obs_enc.parameters(),
                                                         1.0)  # gradient clipping
                self.opt_critic_obs_enc.step()

            if self._args["if_grad_clip"]: clip_grad(self.main_critic.parameters(), 1.0)  # gradient clipping
            self.opt_critic.step()

        if kwargs["if_ret"]:
            # === Update Actor
            if self._args["env_name"].lower() in ["mine", "recsim-data"]:
                if not self._args['mw_obs_flatten']:
                    actor_state = self.main_actor_obs_enc(input_dict["state_embed"])
                    actor_state = actor_state.squeeze(1)
                else:
                    actor_state = input_dict["state_embed"]
            elif self._args["env_name"].startswith("recsim"):
                actor_state = input_dict["state_embed"]
            else:
                actor_state = None
            pi, log_pi, _ = self.main_actor(actor_state)  # Deterministic sampling; batch x sample x action
            _s = actor_state[:, None, :].repeat(1, pi.shape[1], 1)
            Q_vals = self.main_critic(_s, pi)
            policy_loss = ((self.alpha * log_pi) - torch.min(*Q_vals)).mean()  # SAC update of Actor

            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.opt_actor_obs_enc.zero_grad()
            self.opt_actor.zero_grad()

            policy_loss.backward()

            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                if self._args["if_grad_clip"]: clip_grad(self.main_actor_obs_enc.parameters(), 1.0)  # gradient clipping
                self.opt_actor_obs_enc.step()
            if self._args["if_grad_clip"]: clip_grad(self.main_actor.parameters(), 1.0)  # gradient clipping
            self.opt_actor.step()

            if self._args["WOLP_if_auto_ent_tune"]:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.).to(self._device)

        if kwargs["if_sel"]:
            self.res = {"value_loss": value_loss.item(), "loss": value_loss.item()}
        if kwargs["if_ret"]:
            self.res["policy_loss"] = policy_loss.item()
            self.res["loss"] = self.res["value_loss"] + policy_loss.item()

    def _sync(self, tau: float = 0.0):
        if tau > 0.0:  # Soft update of params
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                for param, target_param in zip(self.main_actor_obs_enc.parameters(),
                                               self.target_actor_obs_enc.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                for param, target_param in zip(self.main_critic_obs_enc.parameters(),
                                               self.target_critic_obs_enc.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            for param, target_param in zip(self.main_actor.parameters(), self.target_actor.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            for param, target_param in zip(self.main_critic.parameters(), self.target_critic.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

        else:
            if self.main_critic is not None:
                self.target_critic.load_state_dict(self.main_critic.state_dict())
            if self.main_actor is not None:
                self.target_actor.load_state_dict(self.main_actor.state_dict())
            if self.main_critic_obs_enc is not None:
                self.target_critic_obs_enc.load_state_dict(self.main_critic_obs_enc.state_dict())
            if self.main_actor_obs_enc is not None:
                self.target_actor_obs_enc.load_state_dict(self.main_actor_obs_enc.state_dict())
