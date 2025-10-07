import os.path as osp
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

from large_rl.embedding.base import BaseEmbedding
from large_rl.policy.agent import Agent
from large_rl.commons.utils import logging
from large_rl.policy.arch.mlp import MLP


class DQN(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Init the Q-nets
        self._dim_out = 1

        ## Add encoder
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self.main_obs_enc = self._instantiate_obs_encoder()
            self.target_obs_enc = deepcopy(self.main_obs_enc)
            self._obs_enc_opt = optim.Adam(self.main_obs_enc.parameters(), lr=self._args["mw_obs_enc_lr"])
            logging(self.main_obs_enc)
        else:
            self.main_obs_enc = self.target_obs_enc = self._obs_enc_opt = None

        if self._args["env_name"].startswith("recsim"):
            self._dim_state = self._args["recsim_dim_embed"]
            self._dim_action = self._args["recsim_dim_tsne_embed"] if self._args["recsim_if_tsne_embed"] else \
                self._args["recsim_dim_embed"]
        elif self._args["env_name"].lower() == "mine":
            if not self._args['mw_obs_flatten']:
                self._dim_state, self._dim_action = self.main_obs_enc.dim_out, self._args["mw_action_dim"]
            else:
                self._dim_state, self._dim_action = self._args['mw_obs_length'], self._args["mw_action_dim"]
        self._dim_action += self._args["env_dim_extra"] if not self._args["env_act_emb_tSNE"] else 0

        self.main_Q_net = MLP(dim_in=self._dim_state + self._dim_action,
                              dim_hiddens=self._args["Qnet_dim_hidden"],
                              dim_out=1,
                              if_norm_each=self._args["WOLP_if_critic_norm_each"],
                              if_norm_final=self._args["WOLP_if_critic_norm_final"],
                              if_init_layer=self._args["WOLP_if_critic_init_layer"],
                              type_hidden_act_fn=10).to(device=self._device)
        self.target_Q_net = deepcopy(self.main_Q_net)
        self.optimizer = optim.Adam(self.main_Q_net.parameters(), lr=self._args["lr"])
        logging(self.main_Q_net)

    def _select_action(self, input_dict: dict, **kwargs):
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            input_dict["state_embed"] = self.main_obs_enc(input_dict["state_embed"])
            if self._args["env_name"].lower() == "recsim-data":
                input_dict["state_embed"] = input_dict["state_embed"].view(
                    input_dict["act_embed"].shape[0], self._args["num_all_actions"], self._args["recsim_dim_embed"]
                )

        self.main_Q_net.eval()
        with torch.no_grad():
            # batch x num_candidates
            q_i = self.main_Q_net(torch.cat([input_dict["state_embed"], input_dict["act_embed"]], dim=-1))
        self.main_Q_net.train()
        a = torch.topk(q_i.squeeze(-1), k=self._args["recsim_slate_size"]).indices  # batch_step_size x slate_size
        return a.cpu().detach().numpy().astype(np.int64)

    def _update(self,
                input_dict: dict,
                next_input_dict: dict,
                actions: np.ndarray,
                rewards: torch.tensor,
                reversed_dones: torch.tensor,
                act_embed_base: BaseEmbedding, **kwargs):
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            input_dict["state_embed"] = self.main_obs_enc(input_dict["state_embed"])
            with torch.no_grad():
                next_input_dict["state_embed"] = self.target_obs_enc(next_input_dict["state_embed"])
            if self._args["env_name"].lower() == "recsim-data":
                input_dict["state_embed"] = input_dict["state_embed"][:, None, :]
                next_input_dict["state_embed"] = next_input_dict["state_embed"].view(
                    self._args["batch_size"], self._args["num_all_actions"], self._args["recsim_dim_embed"]
                )

        actions = torch.tensor(actions, device=self._device)
        # Q_all_actions = self.main_Q_net(
        action_embed = act_embed_base.get(index=torch.tensor(actions, device=self._device))  # batch x 1 x dim-act
        Q_vals = self.main_Q_net(torch.cat([input_dict["state_embed"], action_embed], dim=-1)).squeeze(-1)
        # Next Q-vals
        with torch.no_grad():
            next_Q_vals = self.target_Q_net(
                torch.cat([next_input_dict["state_embed"], next_input_dict["act_embed"]], dim=-1))
            next_Q_vals = torch.topk(next_Q_vals.squeeze(-1), k=self._args["recsim_slate_size"]).values
            Q_target = rewards + self._args["Qnet_gamma"] * next_Q_vals * reversed_dones
        bellmann_error = F.mse_loss(Q_vals, Q_target)
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
            'mw_obs_flatten']: self._obs_enc_opt.zero_grad()
        self.optimizer.zero_grad()
        bellmann_error.backward()
        if self._args["if_grad_clip"]: nn.utils.clip_grad_norm_(self.main_Q_net.parameters(), 1.)
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten'] and \
                self._args["if_grad_clip"]:
            nn.utils.clip_grad_norm_(self.main_obs_enc.parameters(), 1.)
        self.optimizer.step()
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
            'mw_obs_flatten']: self._obs_enc_opt.step()

        self.res = {"loss": bellmann_error.item()}

    def _save(self, save_dir):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        logging("Save the agent: {}".format(save_dir))
        if self._args['env_name'] == 'mine' and not self._args['mw_obs_flatten']:
            torch.save(self.main_obs_enc.state_dict(), os.path.join(save_dir, f"main_obs_enc.pkl"))
            torch.save(self.target_obs_enc.state_dict(), os.path.join(save_dir, f"target_obs_enc.pkl"))
            torch.save(self._obs_enc_opt.state_dict(), os.path.join(save_dir, f"obs_enc_opt.pkl"))
        torch.save(self.main_Q_net.state_dict(), os.path.join(save_dir, f"main.pkl"))
        torch.save(self.target_Q_net.state_dict(), os.path.join(save_dir, f"target.pkl"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, f"opt.pkl"))

    def _load(self, load_dir):
        logging("Load the agent: {}".format(load_dir))
        if self._args['env_name'] == 'mine' and not self._args['mw_obs_flatten']:
            self.main_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"main_obs_enc.pkl")))
            self.target_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"target_obs_enc.pkl")))
            self._obs_enc_opt.load_state_dict(torch.load(os.path.join(load_dir, f"obs_enc_opt.pkl")))
        self.main_Q_net.load_state_dict(torch.load(os.path.join(load_dir, f"main.pkl")))
        self.target_Q_net.load_state_dict(torch.load(os.path.join(load_dir, f"target.pkl")))
        self.optimizer.load_state_dict(torch.load(os.path.join(load_dir, f"opt.pkl")))

    def _sync(self, tau: float = 0.0):
        if tau > 0.0:  # Soft update of params
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                for param, target_param in zip(self.main_obs_enc.parameters(), self.target_obs_enc.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
            for param, target_param in zip(self.main_Q_net.parameters(), self.target_Q_net.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
        else:
            self.target_Q_net.load_state_dict(self.main_Q_net.state_dict())
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.target_obs_enc.load_state_dict(self.main_obs_enc.state_dict())

    def _train(self):
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self.main_obs_enc.train()
            self.target_obs_enc.train()
        self.main_Q_net.train()
        self.target_Q_net.train()

    def _eval(self):
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self.main_obs_enc.eval()
            self.target_obs_enc.eval()
        self.main_Q_net.eval()
        self.target_Q_net.eval()
