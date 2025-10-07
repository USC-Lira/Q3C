""" Cascaded Critics operates on the discrete candidates """

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_ as clip_grad
from copy import deepcopy
import os.path as osp
import os

from large_rl.embedding.base import BaseEmbedding
from large_rl.policy.arch.mlp import MLP
from large_rl.policy.arch.ddpg import Actor, DDPG_OUNoise as OUNoise, GaussianNoise, FixedGaussianNoise
from large_rl.policy.arch.arddpg import ARActor, ARCritic
from large_rl.policy.arch.cem import CEM
from large_rl.policy.agent import Agent
from large_rl.policy.policy_utils import compute_penalty
from large_rl.commons.utils import logging


class WOLP(Agent):
    def __init__(self, **kwargs):
        super(WOLP, self).__init__(**kwargs)
        # TODO: Change the names of selection and refinement Q to something simpler and uniform.
        # TODO: Changes the names of retrieval actor and critic to something simpler and uniform.
        self._distance_list, self._activate_list = list(), list()
        self._query = self._topk_act = None  # this is for CDDPG
        self._query_max = None  # this is for CDDPG
        self._max_index = None
        self.mu_list_star = None

        # Prep the input dim
        def _get_dim_cascade_input():
            if self._args["env_name"].lower().startswith("recsim"):
                return self._args["recsim_dim_embed"]
            elif self._args["env_name"].lower() == "mine":
                if not self._args["mw_obs_flatten"]:
                    return self.main_actor_obs_enc.dim_out
                else:
                    return self._args["mw_obs_length"]
            elif self._args["env_name"].lower().startswith("mujoco"):
                return self._args["reacher_obs_space"]
            else:
                raise ValueError

        dim_hidden = self._args.get('dim_hidden', 64)
        WOLP_ar_critic_lr = self._args["WOLP_ar_critic_lr"] if self._args["WOLP_ar_critic_lr"] is not None \
            else self._args["WOLP_critic_lr"]

        # import pudb; pudb.start()
        # === RETRIEVAL ===
        if self._args['env_name'].lower().startswith("mujoco"):
            self._actor_dim_out = self._args["reacher_action_shape"]
            self._critic_dim_state = self._args["reacher_obs_space"]
            self._critic_dim_action = self._args["reacher_action_shape"]

        # === Actor ===
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self._define_obs_encoders(actor=True, critic=False)
        else:
            self.main_actor_obs_enc = self.target_actor_obs_enc = self.opt_actor_obs_enc = None

        if self._args["WOLP_if_joint_actor"] and not self._args["WOLP_if_ar"]:
            if self._args["env_name"].lower().startswith("recsim"):
                _dim = self._args["recsim_dim_tsne_embed"] if self._args["recsim_if_tsne_embed"] \
                    else self._args["recsim_dim_embed"]
            elif self._args["env_name"].lower() == "mine":
                _dim = self._args["mw_action_dim"]
            elif self._args["env_name"].lower().startswith("mujoco"):
                _dim = self._args["reacher_action_shape"]
            else:
                raise ValueError
            _dim += self._args["env_dim_extra"] if not self._args["env_act_emb_tSNE"] else 0
            self._actor_dim_out = _dim * self._args["WOLP_cascade_list_len"]
        else:
            if self._args["env_name"].lower().startswith("recsim"):
                self._actor_dim_out = _dim = self._args["recsim_dim_tsne_embed"] if self._args["recsim_if_tsne_embed"] \
                    else self._args["recsim_dim_embed"]
            elif self._args["env_name"].lower() == "mine":
                self._actor_dim_out = self._args["mw_action_dim"]
            elif self._args["env_name"].lower().startswith("mujoco"):
                self.actor_dim_out = self._args["reacher_action_shape"]
            else:
                raise ValueError
            self._actor_dim_out += self._args["env_dim_extra"] if not self._args["env_act_emb_tSNE"] else 0  # noisy dim

        if self._args["WOLP_if_cem_actor"]:
            self.cem = CEM(seed=self._args["seed"], dim_action=self._actor_dim_out, topK=self._args["CEM_topK"])
            self.opt_actor = self.main_actor = self.target_actor = None
        else:
            self.main_actor, self.target_actor, self.opt_actor = None, None, None

            if self._args["WOLP_if_ar"]:
                self.main_actor = ARActor(dim_in=_get_dim_cascade_input(), dim_hidden=dim_hidden,
                                          dim_memory=self._args["WOLP_slate_dim_out"], dim_out=self._actor_dim_out,
                                          args=self._args).to(self._args["device"])
                self.target_actor = deepcopy(self.main_actor)
                if self._args["WOLP_ar_if_opt_for_list_enc"]:
                    list_enc_params, others_params = list(), list()
                    for _name, _module in self.main_actor.named_parameters():
                        if _name.startswith("list_encoder"):
                            list_enc_params.append(_module)
                        else:
                            others_params.append(_module)
                    self.opt_actor = torch.optim.Adam(params=others_params, lr=self._args["WOLP_actor_lr"])
                    self.opt_actor.add_param_group(param_group={
                        "params": list_enc_params, "lr": self._args["WOLP_list_enc_lr"]})
                else:
                    self.opt_actor = torch.optim.Adam(
                        params=self.main_actor.parameters(), lr=self._args["WOLP_actor_lr"])
            else:
                self.main_actor = Actor(dim_in=_get_dim_cascade_input(),
                                        dim_hiddens=self._args["WOLP_actor_dim_hiddens"],
                                        if_init_layer=self._args["WOLP_if_actor_init_layer"],
                                        if_norm_each=self._args["WOLP_if_actor_norm_each"],
                                        if_norm_final=self._args["WOLP_if_actor_norm_final"],
                                        dim_out=self._actor_dim_out,
                                        args=self._args
                                        ).to(device=self._device)
                self.target_actor = deepcopy(self.main_actor)
                self.opt_actor = torch.optim.Adam(params=self.main_actor.parameters(), lr=self._args["WOLP_actor_lr"])
            logging(self.main_actor)

        # === SELECTION / REFINEMENT ===
        # === Critic ===
        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self._define_obs_encoders(actor=False, critic=True, twin_q=self._args["TwinQ"])
        else:
            self.main_ref_critic_obs_enc = self.target_ref_critic_obs_enc = self.opt_ref_critic_obs_enc = None
            if self._args["TwinQ"]:
                self.main_ref_critic_obs_enc_twin = self.target_ref_critic_obs_enc_twin = self.opt_ref_critic_obs_enc_twin = None

        if self._args["env_name"].lower().startswith("recsim"):
            self._critic_dim_state = self._args["recsim_dim_embed"]
            self._critic_dim_action = self._args["recsim_dim_tsne_embed"] if self._args["recsim_if_tsne_embed"] else \
                self._args["recsim_dim_embed"]
        elif self._args["env_name"].lower() == "mine":
            if not self._args['mw_obs_flatten']:
                self._critic_dim_state = self.main_ref_critic_obs_enc.dim_out
                self._critic_dim_action = self._args["mw_action_dim"]
            else:
                self._critic_dim_state = self._args["mw_obs_length"]
                self._critic_dim_action = self._args["mw_action_dim"]
        elif self._args['env_name'].lower().startswith("mujoco"):
            self._critic_dim_state = self._args["reacher_obs_space"]
            self._critic_dim_action = self._args["reacher_action_shape"]
        else:
            raise ValueError

        self._critic_dim_action += self._args["env_dim_extra"] if not self._args["env_act_emb_tSNE"] else 0

        self.main_ref_critic = MLP(dim_in=self._critic_dim_state + self._critic_dim_action,
                                   dim_hiddens=self._args["Qnet_dim_hidden"],
                                   dim_out=1,
                                   if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                   if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                   if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                   type_hidden_act_fn=10).to(device=self._device)
        self.target_ref_critic = deepcopy(self.main_ref_critic)
        self.opt_ref_critic = torch.optim.Adam(params=self.main_ref_critic.parameters(),
                                               lr=self._args["WOLP_critic_lr"])
        if self._args["TwinQ"]:
            self.main_ref_critic_twin = MLP(dim_in=self._critic_dim_state + self._critic_dim_action,
                                            dim_hiddens=self._args["Qnet_dim_hidden"],
                                            dim_out=1,
                                            if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                            if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                            if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                            type_hidden_act_fn=10).to(device=self._device)
            self.target_ref_critic_twin = deepcopy(self.main_ref_critic_twin)
            self.opt_ref_critic_twin = torch.optim.Adam(params=self.main_ref_critic_twin.parameters(),
                                                        lr=self._args["WOLP_critic_lr"])
        logging(self.main_ref_critic)

        if self._args["REDQ"]:
            assert self._args['env_name'].lower() not in ["mine", "recsim-data"]
            self.main_ref_critic_redq_list = [self.main_ref_critic]
            self.target_ref_critic_redq_list = [self.target_ref_critic]
            for _ in range(self._args["REDQ_num"] - 1):
                self.main_ref_critic_redq_list.append(
                    MLP(dim_in=self._critic_dim_state + self._critic_dim_action,
                        dim_hiddens=self._args["Qnet_dim_hidden"],
                        dim_out=1,
                        if_norm_each=self._args["WOLP_if_critic_norm_each"],
                        if_norm_final=self._args["WOLP_if_critic_norm_final"],
                        if_init_layer=self._args["WOLP_if_critic_init_layer"],
                        type_hidden_act_fn=10).to(device=self._device)
                )
                self.target_ref_critic_redq_list.append(deepcopy(self.main_ref_critic_redq_list[-1]))
            self.opt_ref_critic_redq_list = [torch.optim.Adam(params=_critic.parameters(),
                                                                lr=self._args["WOLP_critic_lr"]) for _critic in
                                                self.main_ref_critic_redq_list]

        # === Extra Critic(Joint / Dual / FLAIR) ===
        if self._args["WOLP_if_dual_critic"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self._define_extra_critic_obs_encoders(twin_q=self._args["TwinQ"])
            else:
                self.main_extra_critic_obs_enc = self.target_extra_critic_obs_enc = self.opt_extra_critic_obs_enc = None
                if self._args["TwinQ"]:
                    self.main_extra_critic_obs_enc_twin = self.target_extra_critic_obs_enc_twin = self.opt_extra_critic_obs_enc_twin = None

            self.main_extra_critic = MLP(dim_in=self._critic_dim_state + self._critic_dim_action,
                                         dim_hiddens=self._args["Qnet_dim_hidden"],
                                         dim_out=1,
                                         if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                         if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                         if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                         type_hidden_act_fn=10).to(device=self._device)
            self.target_extra_critic = deepcopy(self.main_extra_critic)
            self.opt_extra_critic = torch.optim.Adam(params=self.main_extra_critic.parameters(),
                                                     lr=self._args["WOLP_critic_lr"])
            if self._args["TwinQ"]:
                self.main_extra_critic_twin = MLP(dim_in=self._critic_dim_state + self._critic_dim_action,
                                                  dim_hiddens=self._args["Qnet_dim_hidden"],
                                                  dim_out=1,
                                                  if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                                  if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                                  if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                                  type_hidden_act_fn=10).to(device=self._device)
                self.target_extra_critic_twin = deepcopy(self.main_extra_critic_twin)
                self.opt_extra_critic_twin = torch.optim.Adam(params=self.main_extra_critic_twin.parameters(),
                                                              lr=self._args["WOLP_critic_lr"])
            logging(self.main_extra_critic)
        else:
            self.main_extra_critic = self.target_extra_critic = self.opt_extra_critic = None

        if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self._define_joint_critic_obs_encoders(twin_q=self._args["TwinQ"])
            else:
                self.main_joint_critic_obs_enc = self.target_joint_critic_obs_enc = self.opt_joint_critic_obs_enc = \
                    None
                if self._args["TwinQ"]:
                    self.main_joint_critic_obs_enc_twin = self.target_joint_critic_obs_enc_twin = \
                        self.opt_joint_critic_obs_enc_twin = None

            _dim_in = self._critic_dim_state + self._critic_dim_action * self._args["WOLP_cascade_list_len"]
            self.main_joint_critic = MLP(dim_in=_dim_in,
                                         dim_hiddens=self._args["Qnet_dim_hidden"],
                                         dim_out=1,
                                         if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                         if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                         if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                         type_hidden_act_fn=10).to(device=self._device)
            self.target_joint_critic = deepcopy(self.main_joint_critic)
            self.opt_joint_critic = torch.optim.Adam(params=self.main_joint_critic.parameters(),
                                                     lr=self._args["WOLP_critic_lr"])
            if self._args["TwinQ"]:
                self.main_joint_critic_twin = MLP(dim_in=_dim_in,
                                                  dim_hiddens=self._args["Qnet_dim_hidden"],
                                                  dim_out=1,
                                                  if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                                  if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                                  if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                                  type_hidden_act_fn=10).to(device=self._device)
                self.target_joint_critic_twin = deepcopy(self.main_joint_critic_twin)
                self.opt_joint_critic_twin = torch.optim.Adam(params=self.main_joint_critic_twin.parameters(),
                                                              lr=self._args["WOLP_critic_lr"])
            logging(self.main_joint_critic)
        else:
            self.main_joint_critic = self.target_joint_critic = self.opt_joint_critic = None

        if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self._define_ar_critic_obs_encoders(twin_q=self._args["TwinQ"])
            else:
                self.main_ar_critic_obs_enc = self.target_ar_critic_obs_enc = self.opt_ar_critic_obs_enc = None
                if self._args["TwinQ"]:
                    self.main_ar_critic_obs_enc_twin = self.target_ar_critic_obs_enc_twin = \
                        self.opt_ar_critic_obs_enc_twin = None
            if self._args["WOLP_if_joint_critic"]:
                _dim_in = self._critic_dim_state + self._critic_dim_action * self._args["WOLP_cascade_list_len"]
                self.main_ar_critic = MLP(dim_in=_dim_in,
                                          dim_hiddens=self._args["Qnet_dim_hidden"],
                                          dim_out=1,
                                          if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                          if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                          if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                          type_hidden_act_fn=10).to(device=self._device)
                self.target_ar_critic = deepcopy(self.main_ar_critic)
                self.opt_ar_critic = torch.optim.Adam(params=self.main_ar_critic.parameters(),
                                                      lr=WOLP_ar_critic_lr)
                if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                    self.main_ar_critic_twin = MLP(dim_in=_dim_in,
                                                   dim_hiddens=self._args["Qnet_dim_hidden"],
                                                   dim_out=1,
                                                   if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                                   if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                                   if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                                   type_hidden_act_fn=10).to(device=self._device)
                    self.target_ar_critic_twin = deepcopy(self.main_ar_critic_twin)
                    self.opt_ar_critic_twin = torch.optim.Adam(params=self.main_ar_critic_twin.parameters(),
                                                               lr=WOLP_ar_critic_lr)
            else:
                self.main_ar_critic = ARCritic(args=self._args,
                                               dim_state=_get_dim_cascade_input(),
                                               dim_hidden=dim_hidden, dim_memory=self._args["WOLP_slate_dim_out"],
                                               dim_action=self._critic_dim_action).to(device=self._device)
                self.target_ar_critic = deepcopy(self.main_ar_critic)
                if self._args["WOLP_if_0th_ref_critic"]:
                    assert self._args["WOLP_t0_no_list_input"] and not self._args["WOLP_if_ar_critic_share_weight"]
                    self.main_ar_critic.cells[0].Q_net = self.main_ref_critic
                    self.target_ar_critic.cells[0].Q_net = self.target_ref_critic

                if self._args["WOLP_ar_if_opt_for_list_enc"]:
                    list_enc_params, others_params = list(), list()
                    for _name, _module in self.main_ar_critic.named_parameters():
                        if _name.startswith("list_encoder"):
                            list_enc_params.append(_module)
                        else:
                            others_params.append(_module)
                    self.opt_ar_critic = torch.optim.Adam(params=others_params, lr=WOLP_ar_critic_lr)
                    self.opt_ar_critic.add_param_group(param_group={
                        "params": list_enc_params, "lr": self._args["WOLP_list_enc_lr"]})
                else:
                    self.opt_ar_critic = torch.optim.Adam(
                        params=self.main_ar_critic.parameters(), lr=WOLP_ar_critic_lr)

                if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                    self.main_ar_critic_twin = ARCritic(args=self._args,
                                                        dim_state=_get_dim_cascade_input(),
                                                        dim_hidden=dim_hidden,
                                                        dim_memory=self._args["WOLP_slate_dim_out"],
                                                        dim_action=self._critic_dim_action).to(device=self._device)
                    self.target_ar_critic_twin = deepcopy(self.main_ar_critic_twin)
                    if self._args["WOLP_if_0th_ref_critic"]:
                        assert self._args["WOLP_t0_no_list_input"] and not self._args["WOLP_if_ar_critic_share_weight"]
                        self.main_ar_critic_twin.cells[0].Q_net = self.main_ref_critic_twin
                        self.target_ar_critic_twin.cells[0].Q_net = self.target_ref_critic_twin

                    if self._args["WOLP_ar_if_opt_for_list_enc"]:
                        list_enc_params, others_params = list(), list()
                        for _name, _module in self.main_ar_critic_twin.named_parameters():
                            if _name.startswith("list_encoder"):
                                list_enc_params.append(_module)
                            else:
                                others_params.append(_module)
                        self.opt_ar_critic_twin = torch.optim.Adam(params=others_params,
                                                                   lr=WOLP_ar_critic_lr)
                        self.opt_ar_critic_twin.add_param_group(param_group={
                            "params": list_enc_params, "lr": self._args["WOLP_list_enc_lr"]})
                    else:
                        self.opt_ar_critic_twin = torch.optim.Adam(
                            params=self.main_ar_critic_twin.parameters(), lr=WOLP_ar_critic_lr)
            logging(self.main_ar_critic)
        else:
            self.main_ar_critic = self.target_ar_critic = self.opt_ar_critic = None
            if self._args["TwinQ"]:
                self.main_ar_critic_twin = self.target_ar_critic_twin = self.opt_ar_critic_twin = None

        # === OU Noise ===
        if (self._args["WOLP_if_ar"] or self._args["WOLP_if_joint_actor"]) \
              and self._args["WOLP_if_noise_postQ"]:
            self.dim_action = (self._args["num_envs"], 1, self._critic_dim_action)
        elif self._args["WOLP_if_ar"] and self._args["WOLP_cascade_list_len"] > 1:
            self.dim_action = (self._args["num_envs"], self._args["WOLP_cascade_list_len"], self._actor_dim_out)
        elif self._args["WOLP_if_joint_actor"]:
            assert not self._args["WOLP_if_noise_postQ"]
            self.dim_action = (self._args["num_envs"], self._args["WOLP_cascade_list_len"],
                            self._actor_dim_out // self._args["WOLP_cascade_list_len"])
        else:
            self.dim_action = (self._args["num_envs"], 1, self._actor_dim_out)

        if (self._args["WOLP_total_dual_exploration"] and self._args["WOLP_preturb_with_fixed_Gaussian"]):
            self.noise_sampler = FixedGaussianNoise(dim_action=self.dim_action, mu=0,
                                               sigma=self._args["WOLP_noise_expl_sigma"],
                                               env_max_action=self._args["env_max_action"],
                                               device=self._args["device"])
        elif self._args["WOLP_noise_type"] == "normal":
            self.noise_sampler = GaussianNoise(dim_action=self.dim_action, mu=0,
                                               sigma=self._args["WOLP_noise_expl_sigma"],
                                               env_max_action=self._args["env_max_action"],
                                               device=self._args["device"])
        elif self._args["WOLP_noise_type"] == "ou":
            self.noise_sampler = OUNoise(dim_action=self.dim_action,
                                         env_max_action=self._args["env_max_action"],
                                         device=self._args["device"])
        else:
            raise NotImplementedError

        if self._args["env_name"].lower().startswith('mujoco') and \
            (self._args["WOLP_total_dual_exploration"] or self._args["WOLP_if_noise_postQ"]):
            # Total adds noise to only the final selected action as opposed to list_len actions.
            if self._args["WOLP_noise_type"] == "ou":
                self.noise_sampler_total = OUNoise(dim_action=(self._args["num_envs"], 1, self._actor_dim_out),
                                                   env_max_action=self._args["env_max_action"],
                                                   device=self._args["device"])
            else:
                self.noise_sampler_total = GaussianNoise(dim_action=(self._args["num_envs"], 1, self._actor_dim_out),
                                                        mu=0, sigma=self._args["WOLP_noise_expl_sigma"],
                                                        env_max_action=self._args["env_max_action"],
                                                        device=self._args["device"])
    def reinitialize_layers(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def reinitialize_out_layers(module):
            if module is None:
                return
            for name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear) and name == 'out':
                    sub_module.apply(init_weights)
                else:
                    reinitialize_out_layers(sub_module)

        # Re-initialize layers of actors
        if hasattr(self, 'main_actor'):
            reinitialize_out_layers(self.main_actor)
        if hasattr(self, 'target_actor'):
            reinitialize_out_layers(self.target_actor)

        # Re-initialize layers of critics
        if hasattr(self, 'main_ref_critic'):
            reinitialize_out_layers(self.main_ref_critic)
        if hasattr(self, 'target_ref_critic'):
            reinitialize_out_layers(self.target_ref_critic)
        if hasattr(self, 'main_ref_critic_twin'):
            reinitialize_out_layers(self.main_ref_critic_twin)
        if hasattr(self, 'target_ref_critic_twin'):
            reinitialize_out_layers(self.target_ref_critic_twin)
        if hasattr(self, 'main_extra_critic'):
            reinitialize_out_layers(self.main_extra_critic)
        if hasattr(self, 'target_extra_critic'):
            reinitialize_out_layers(self.target_extra_critic)
        if hasattr(self, 'main_extra_critic_twin'):
            reinitialize_out_layers(self.main_extra_critic_twin)
        if hasattr(self, 'target_extra_critic_twin'):
            reinitialize_out_layers(self.target_extra_critic_twin)
        if hasattr(self, 'main_joint_critic'):
            reinitialize_out_layers(self.main_joint_critic)
        if hasattr(self, 'target_joint_critic'):
            reinitialize_out_layers(self.target_joint_critic)
        if hasattr(self, 'main_joint_critic_twin'):
            reinitialize_out_layers(self.main_joint_critic_twin)
        if hasattr(self, 'target_joint_critic_twin'):
            reinitialize_out_layers(self.target_joint_critic_twin)
        if hasattr(self, 'main_ar_critic'):
            reinitialize_out_layers(self.main_ar_critic)
        if hasattr(self, 'target_ar_critic'):
            reinitialize_out_layers(self.target_ar_critic)
        if hasattr(self, 'main_ar_critic_twin'):
            reinitialize_out_layers(self.main_ar_critic_twin)
        if hasattr(self, 'target_ar_critic_twin'):
            reinitialize_out_layers(self.target_ar_critic_twin)

        # Skip re-initialization for optimizer attributes
        # `opt_ref_critic`, `opt_ref_critic_twin`, `opt_extra_critic`, `opt_extra_critic_twin`, `opt_joint_critic`,
        # `opt_joint_critic_twin`, `opt_ar_critic`, `opt_ar_critic_twin` are optimizers and should not be reinitialized


    def _cem_get_query(self, state, q_net, **kwargs):
        # CEM based policy to work out the optimised query
        self.cem.initialise(batch_size=state.shape[0])
        state = state[:, None, :].repeat(1, self._args["CEM_num_samples"], 1)

        for i in range(self._args["CEM_num_iter"]):
            actions = self.cem.sample(self._args["CEM_num_samples"],
                                      self._args["CEM_rescale_actions"],
                                      self._args["env_max_action"])  # b x num_samples x dim_action

            with torch.no_grad():
                # b x num_samples
                Q_vals = q_net(torch.cat([state, torch.tensor(actions, device=self._device)], -1)).squeeze(-1)
            if i < (self._args["CEM_num_iter"] - 1):
                idx = torch.topk(Q_vals, k=self._args["CEM_topK"]).indices.cpu().detach().numpy()  # b x topK
                elites = np.take_along_axis(arr=actions, indices=idx[..., None], axis=1)  # b x topK x dim
                self.cem.update(elite_samples=elites)

        max_idx = torch.topk(Q_vals, k=self._args["WOLP_cascade_list_len"]).indices.cpu().detach().numpy()
        mu = np.take_along_axis(arr=actions, indices=max_idx[..., None], axis=1)  # b x topK x dim_action
        if not self._args["env_name"].lower().startswith("mujoco"):
            # No randomness here is okay for continuous environments
            # because we are affording the Q-function a chance to select any of the topK actions.
            mu = self._add_noise(mu=torch.tensor(mu, device=self._device), **kwargs)
            return mu
        else:
            return torch.tensor(mu, device=self._device)

    def _add_noise(self, mu, total=False, **kwargs):
        if not kwargs.get("if_update", False):
            if total:
                eps = self.noise_sampler_total.noise(scale=kwargs["epsilon"]["actor"])
            else:
                eps = self.noise_sampler.noise(scale=kwargs["epsilon"]["actor"])
            if self._args["WOLP_0th_no_perturb"] and \
                not (self._args["WOLP_total_dual_exploration"] and total) and \
                not self._args["WOLP_if_noise_postQ"]:
                # Don't remove 0th action perturbation if
                ## 1. Total dual exploration's total exploration phase
                ## 2. Post-Q exploration (because that is the only exploration step)
                eps[:, 0, :] = 0
            mu += eps
            if self._args["DEBUG_type_clamp"].lower() == "small":
                mu = mu.clamp(0, self._args['env_max_action'])  # Constraint the range
            elif self._args["DEBUG_type_clamp"].lower() == "large":
                mu = mu.clamp(-self._args['env_max_action'], self._args['env_max_action'])  # Constraint the range
        return mu

    def _continuous_kNN(self, mu, _top_k):
        """
        In the continuous action space of the environment, find neighbors in a close neighborhood of the original
        continuous action.
        """
        topk_act = mu.repeat([1, _top_k, 1])
        topk_act[:,1:] += self._args['continuous_kNN_sigma'] * \
            torch.randn([mu.shape[0], _top_k - 1, mu.shape[2] ], device=self._device)
        return topk_act

    def _perform_kNN(self, act_embed, mu, _top_k):
        """
         Args:
             act_embed (torch.tensor): batch_size x num_candidates x dim_act
             mu (torch.tensor): batch_size x num_query x dim_act
         Returns:
             batch_size x num_candidates
         """
        # score = - torch.cdist(x1=mu, x2=act_embed, p=2.0).squeeze(-1)  # batch x list_len x num_act
        score = - torch.cdist(x1=mu, x2=act_embed, p=2.0, compute_mode="donot_use_mm_for_euclid_dist").squeeze(-1)
        topk_act = list()
        if self._args['WOLP_allow_kNN_duplicate']:
            topk_act_3D = torch.topk(score, k=_top_k, dim=-1).indices  # batch x list-len x top-K
        else:
            _mask = torch.zeros(mu.shape[0], act_embed.shape[1], requires_grad=False, device=self._device)  # b x num-act
            _temp = torch.tensor(np.asarray([np.arange(mu.shape[0]) for _ in range(_top_k)]).T, device=self._device)
            _v = torch.tensor(- 1000000., device=self._device)
            for _ind in range(mu.shape[1]):  # for loop along the queries
                _score = score[:, _ind, :]
                _score += _mask  # Avoid the duplicate candidate actions

                _topk_act = torch.topk(_score, k=_top_k, dim=-1).indices  # batch x topK
                if mu.shape[1] > 1: _mask = _mask.index_put_((_temp, _topk_act), _v)  # Update Mask w/h selected indices
                topk_act.append(_topk_act)
            topk_act_3D = torch.stack(topk_act, dim=1)  # batch x num_query x top-K

        topk_act_2D = topk_act_3D.view(topk_act_3D.size(0), topk_act_3D.size(1) * topk_act_3D.size(2))  # b x (L * K)
        topk_embed = act_embed.gather(dim=1, index=topk_act_2D[..., None].repeat(1, 1, act_embed.shape[-1]))
        return topk_act_2D, topk_act_3D, topk_embed  # batch x num_query or batch x topK

    def _select_action(self, input_dict: dict, naive_eval=False, **kwargs):
        _batch = input_dict["state_embed"].shape[0]
        # === 1. When called from update method, we can switch to Actor Target === #
        actor = self.target_actor if kwargs.get("if_update", False) else self.main_actor
        # Note: Action is always selected using the refinement critic, because that's the maximization objective
        critic = self.target_ref_critic if kwargs.get("if_update", False) else self.main_ref_critic
        if (kwargs.get("if_get_Q_vals", False) and self._args["TwinQ"]) or \
            (self._args["WOLP_use_conservative_Q_max"] and not self._args["WOLP_use_main_ref_critic_for_action_selection"]):
            critic_twin = self.target_ref_critic_twin if kwargs.get("if_update", False) else self.main_ref_critic_twin

        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            obs_actor_enc = self.target_actor_obs_enc if kwargs.get("if_update", False) else self.main_actor_obs_enc
            obs_critic_enc = self.target_ref_critic_obs_enc if kwargs.get("if_update",
                                                                          False) else self.main_ref_critic_obs_enc
            if kwargs.get("if_get_Q_vals", False) and self._args["TwinQ"]:
                obs_critic_enc_twin = self.target_ref_critic_obs_enc_twin if kwargs.get("if_update",
                                                                                        False) else self.main_ref_critic_obs_enc_twin
        else:
            obs_actor_enc = None
            obs_critic_enc = None
            obs_critic_enc_twin = None

        # === 2. CEM Actor === #
        if self._args["WOLP_if_cem_actor"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                _s = obs_actor_enc(input_dict["state_embed"]).squeeze(1)
            else:
                _s = input_dict["state_embed"]
            mu = self._cem_get_query(state=_s, q_net=critic, **kwargs)
            self.mu_list_star = deepcopy(mu.detach())
            self.mu_list_eps = deepcopy(mu.detach())
        else:
            # === 3. Get the proto-action from Actor #
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                _s = obs_actor_enc(input_dict["state_embed"]).squeeze(1)
            else:
                _s = input_dict["state_embed"]
            if self._args["WOLP_if_ar"]:
                _s = _s[:, None, :]

            if self._args["WOLP_if_noise_postQ"]:
                # Post-Q Exploration: Do not add noise to actor output, but instead, after Q-value is computed
                # TODO: Thus, set WOLP_if_ar_noise_before_cascade to False.
                mu = actor(_s)
                self.mu_list_star = deepcopy(mu.detach())
                self.mu_list_eps = deepcopy(mu.detach())
            elif self._args["WOLP_if_ar"] and self._args["WOLP_if_ar_noise_before_cascade"]:
                if not kwargs.get("if_update", False):
                    eps = self.noise_sampler.noise(scale=kwargs["epsilon"]["actor"])
                    if self._args["WOLP_0th_no_perturb"]:
                        # Don't remove 0th action perturbation if
                        ## 1. Total dual exploration's total exploration phase
                        ## 2. Post-Q exploration (because that is the only exploration step)
                        eps[:, 0, :] = 0
                else:
                    eps = None

                def knn_action_function(x):
                    if self._args["WOLP_topK"] > 1:
                        with torch.no_grad():
                            topk_embed = self._perform_kNN(input_dict["act_embed"], x, self._args["WOLP_topK"])[2]
                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                _s = obs_critic_enc(input_dict["state_embed"])
                                if self._args["env_name"].lower() == "mine":
                                    _s = _s.repeat(1, topk_embed.shape[1], 1)
                                elif self._args["env_name"].lower() == "recsim-data":
                                    _s = _s[:, None, :].repeat(1, topk_embed.shape[1], 1)
                            else:
                                _s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                            Q_vals = critic(torch.cat([_s, topk_embed], dim=-1)).squeeze(-1)
                            # Find the best action from the topK actions
                            return topk_embed.gather(dim=1, index=Q_vals.argmax(dim=1)[:, None, None].repeat(1, 1, topk_embed.shape[-1]))
                    else:
                        return self._perform_kNN(input_dict["act_embed"], x, 1)[2]

                mu = actor(_s, eps=eps,
                            knn_function=knn_action_function if \
                            (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                and not self._args["env_name"].lower().startswith("mujoco")) \
                            else None)
                self.mu_list_star = deepcopy(mu.detach())
                self.mu_list_eps = deepcopy(mu.detach())
            else:
                mu = actor(_s)
                self.mu_list_star = deepcopy(mu.detach())
                if not kwargs.get('if_update', False) and not self._args["WOLP_if_noise_postQ"]:
                    mu = self._add_noise(mu=mu, **kwargs)
                    self.mu_list_eps = deepcopy(mu.detach())

        # Store the query from Actor
        if kwargs.get("if_update", False):
            self._query = None
        elif self._args['env_name'].lower().startswith("mujoco"):
            assert self.mu_list_star is not None, "mu_list_star is None"
            self._query = torch.cat([self.mu_list_eps, self.mu_list_star], dim=-1).cpu().detach().numpy().astype(np.float32)
        else:
            self._query = self.mu_list_eps.cpu().detach().numpy().astype(np.float32)
        if self._args["WOLP_ar_use_mu_star"]:
            assert self.mu_list_star is not None, "mu_list_star is None"
            mu = deepcopy(self.mu_list_star)

        if kwargs.get("if_update", False) and self._args["WOLP_use_0th_for_target_action_selection"]:
            mu = mu[:, :1]

        if self._args["env_name"].lower().startswith("mujoco"):
            # TODO: Change this generally for continuous action space environments; OR just rename the new env to mujoco-env_name
            # === 4. For Wolpertinger, a replacement for k-NN ~ sample a few actions randomly nearby the action === #
            if self._args['WOLP_topK'] > 1 and not self._args['WOLP_if_cem_actor']:
                topk_embed = self._continuous_kNN(mu, self._args["WOLP_topK"])  # batch x (num_query * topK)
            else:
                topk_embed = mu
            topk_act_2D = topk_embed.view(topk_embed.size(0), -1)
        else:
            # k-NN based on the action-id so that we can let critic directly work on this candidate-set
            topk_act_2D, topk_act_3D, topk_embed = self._perform_kNN(input_dict["act_embed"], mu, self._args["WOLP_topK"])
            if not kwargs.get("if_update", False): self._prev_kNN_candidate_ind = topk_act_2D

            # Store the topk_act from Actor to construct the element-wise reward in list-action
            self._topk_act = topk_act_3D.cpu().detach().numpy()

        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            _s = obs_critic_enc(input_dict["state_embed"])
            if self._args["env_name"].lower() == "mine":
                _s = _s.repeat(1, topk_embed.shape[1], 1)
            elif self._args["env_name"].lower() == "recsim-data":
                _s = _s[:, None, :].repeat(1, topk_embed.shape[1], 1)
        else:
            _s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
        # NOTE w.r.t TwinQ: While selecting the action from the actor, we only use the primary critic.
        # NOTE: max action should still be selected using Q1.
        # However, when we compute the final Q-value, we use both the primary and the twin critic over this selected action.
        if self._args["WOLP_use_main_ref_critic_for_action_selection"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                _s = self.main_ref_critic_obs_enc(input_dict["state_embed"])
                if self._args["env_name"].lower() == "mine":
                    _s = _s.repeat(1, topk_embed.shape[1], 1)
                elif self._args["env_name"].lower() == "recsim-data":
                    _s = _s[:, None, :].repeat(1, topk_embed.shape[1], 1)
            else:
                _s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
            Q_vals = self.main_ref_critic(torch.cat([_s, topk_embed], dim=-1)).squeeze(-1)
            if self._args["WOLP_use_conservative_Q_max"]:
                assert False
                raise NotImplementedError
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    twin_s = self.main_ref_critic_obs_enc_twin(input_dict["state_embed"])
                    if self._args["env_name"].lower() == "mine":
                        twin_s = twin_s.repeat(1, topk_embed.shape[1], 1)
                    elif self._args["env_name"].lower() == "recsim-data":
                        twin_s = twin_s[:, None, :].repeat(1, topk_embed.shape[1], 1)
                else:
                    twin_s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                Q_vals_twin = self.main_ref_critic_twin(torch.cat([twin_s, topk_embed], dim=-1)).squeeze(-1)
                Q_vals = torch.min(Q_vals, Q_vals_twin)
        else:
            Q_vals = critic(torch.cat([_s, topk_embed], dim=-1)).squeeze(-1)
            if self._args["WOLP_use_conservative_Q_max"]:
                assert False
                raise NotImplementedError
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    twin_s = obs_critic_enc_twin(input_dict["state_embed"])
                    if self._args["env_name"].lower() == "mine":
                        twin_s = twin_s.repeat(1, topk_embed.shape[1], 1)
                    elif self._args["env_name"].lower() == "recsim-data":
                        twin_s = twin_s[:, None, :].repeat(1, topk_embed.shape[1], 1)
                else:
                    twin_s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                Q_vals_twin = critic_twin(torch.cat([twin_s, topk_embed], dim=-1)).squeeze(-1)
                Q_vals = torch.min(Q_vals, Q_vals_twin)

        self._max_Q_improvement = (Q_vals.max(dim=1).values - Q_vals[:, 0]).cpu().detach().numpy()
        self._max_Q_improvement_percent = 100 * self._max_Q_improvement / Q_vals[:, 0].cpu().detach().numpy()
        if (not kwargs.get("if_get_Q_vals", False)) and (not kwargs.get("if_update", False)):
            self._candidate_Q_mean = Q_vals.mean(dim=-1).cpu().detach().numpy()
        else:
            self._candidate_Q_mean = None

        _if_boltz = self._args["WOLP_selection_if_boltzmann"] and not kwargs.get("if_update", False) and self._if_train
        if _if_boltz:
            action_dist = torch.distributions.categorical.Categorical(logits=Q_vals / kwargs["epsilon"]["critic"])
            res = action_dist.sample()[:, None]
        else:
            res = torch.topk(Q_vals, k=self._args["recsim_slate_size"])  # batch x slate_size

        if kwargs.get("if_get_Q_vals", False):
            if _if_boltz:
                return res, Q_vals.mean(dim=-1)
            else:
                if self._args["REDQ"]:
                    # Out of 0 to REDQ_num - 1, select REDQ_num_random (defaults to 2) indices randomly
                    # and then select the best action from these indices.
                    _s = input_dict["state_embed"]
                    _rand = self._rng.choice(self._args["REDQ_num"], size=self._args['REDQ_num_random'], replace=False)
                    target_a = topk_embed.gather(dim=1, index=res.indices[:, :, None].repeat(1, 1, topk_embed.shape[-1]))
                    Q_vals = []
                    for ind in _rand:
                        Q_vals.append(self.target_ref_critic_redq_list[ind](torch.cat([_s, target_a.squeeze(1)], dim=-1)))
                    return torch.min(*Q_vals), torch.stack(Q_vals, dim=-1).mean(dim=-1)

                if self._args["TwinQ"]:
                    if self._args["TD3_target_policy_smoothing"]:
                        target_a = topk_embed.gather(dim=1, index=res.indices[:, :, None].repeat(1, 1, topk_embed.shape[-1]))
                        target_noise = (torch.randn_like(target_a) * self._args['TD3_policy_noise']
                                        ).clamp(-self._args['TD3_noise_clip'], self._args['TD3_noise_clip'])
                        target_a = (target_a + target_noise).clamp(-self._args["env_max_action"], self._args["env_max_action"])
                        if self._args["WOLP_discrete_kNN_target_smoothing"] and not self._args['env_name'].lower().startswith("mujoco"):
                            target_a = self._perform_kNN(input_dict["act_embed"], target_a, self._args["WOLP_topK"])[2]
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            _s = obs_critic_enc(input_dict["state_embed"])
                            _s_twin = obs_critic_enc_twin(input_dict["state_embed"])
                            if self._args["env_name"].lower() == "mine":
                                _s = _s[:, 0]
                                _s_twin = _s_twin[:, 0]
                            # elif self._args["env_name"].lower() == "recsim-data":
                            #     _s = _s
                            #     _s_twin = _s_twin
                        else:
                            _s = input_dict["state_embed"]
                            _s_twin = input_dict["state_embed"]
                        Q_val = torch.min(
                            critic(torch.cat([_s, target_a.squeeze(1)], dim=-1)),
                            critic_twin(torch.cat([_s_twin, target_a.squeeze(1)], dim=-1))
                        )
                    else:
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            _s = obs_critic_enc(input_dict["state_embed"])
                            _s_twin = obs_critic_enc_twin(input_dict["state_embed"])
                            if self._args["env_name"].lower() == "mine":
                                _s = _s[:, 0]
                                _s_twin = _s_twin[:, 0]
                            # elif self._args["env_name"].lower() == "recsim-data":
                            #     _s = _s
                            #     _s_twin = _s_twin
                        else:
                            _s = input_dict["state_embed"]
                            _s_twin = input_dict["state_embed"]
                        target_a = topk_embed.gather(dim=1, index=res.indices[:, :, None].repeat(1, 1, topk_embed.shape[-1]))
                        Q_vals_twin = critic_twin(torch.cat([_s_twin, target_a.squeeze(1)], dim=-1))
                        if self._args["WOLP_use_main_ref_critic_for_action_selection"]:
                            Q_val = torch.min(
                                critic(torch.cat([_s, target_a.squeeze(1)], dim=-1)),
                                Q_vals_twin
                            )
                        else:
                            Q_val = torch.min(
                                Q_vals.gather(dim=1, index=res.indices),
                                Q_vals_twin
                            )
                    return Q_val, Q_vals.mean(dim=-1)
                else:
                    return res.values, Q_vals.mean(dim=-1)
        else:
            if naive_eval:
                if self._args["env_name"].lower().startswith("mujoco"):
                    a = topk_embed[:, 0].cpu().detach().numpy()
                else:
                    a = topk_act_2D[:, 0].cpu().detach().numpy().astype(np.int64)
            else:
                ind = res.indices[:, :, None].repeat(1, 1, topk_embed.shape[-1])
                self._query_max = topk_embed.gather(dim=1, index=ind).cpu().detach().numpy().squeeze(1)
                # Get the indices from the original index list
                if self._args["env_name"].lower().startswith("mujoco"):
                    if _if_boltz:
                        raise NotImplementedError
                    else:
                        # ind = res.indices[:, :, None].repeat(1, 1, mu.shape[-1])
                        self._max_index = res.indices.squeeze(1).cpu().detach().numpy()
                        a = topk_embed.gather(dim=1, index=ind)

                        if self._args["WOLP_if_noise_postQ"] or self._args["WOLP_total_dual_exploration"]:
                            a = self._add_noise(mu=a, total=True, **kwargs)
                            a = a.cpu().detach().numpy().squeeze(1)
                        elif self._args["WOLP_if_dual_exploration"] and not _if_boltz:
                            a = a.cpu().detach().numpy().squeeze(1)
                            # Eps-decay for Refine-Q
                            _mask = self._rng.uniform(low=0.0, high=1.0, size=mu.shape[0]) < kwargs["epsilon"]["critic"]
                            self._selectionQ_mask = _mask
                            if sum(_mask) != 0:
                                '''
                                # The problem with this could be that now you would always take a random action
                                # based on what the retriever sampled for the selector.
                                # The final decision is still greedy based on the selection Q-function.
                                # Ideally, we should be able to take completely random actions in this epsilon greedy way. - Not sure
                                # a[_mask] = self.select_random_action(batch_size=sum(_mask), candidate_lists=None)
                                '''
                                a[_mask] = topk_embed.cpu().detach().numpy()[_mask, self._rng.randint(
                                    topk_embed.shape[1], size=sum(_mask)
                                )]
                        else:
                            a = a.cpu().detach().numpy().squeeze(1)
                else:
                    # assert not self._args["WOLP_if_noise_postQ"]
                    if _if_boltz:
                        a = topk_act_2D.gather(dim=-1, index=res).cpu().detach().numpy().astype(np.int64)
                    else:
                        a = topk_act_2D.gather(dim=-1, index=res.indices).cpu().detach().numpy().astype(np.int64)

                    if self._args["WOLP_if_noise_postQ"] or self._args["WOLP_total_dual_exploration"]:
                        # Take an action from the entire action space with epsilon probability
                        # topk_act = topk_act_2D.cpu().detach().numpy().astype(np.int64)
                        # Eps-decay for Refine-Q
                        _mask = self._rng.uniform(low=0.0, high=1.0, size=_batch) < kwargs["epsilon"]["critic"]
                        self._selectionQ_mask = _mask
                        if sum(_mask) != 0:
                            a[_mask] = self.select_random_action(batch_size=sum(_mask), candidate_lists=None)
                    elif self._args["WOLP_if_dual_exploration"] and not _if_boltz:
                        topk_act = topk_act_2D.cpu().detach().numpy().astype(np.int64)
                        # Eps-decay for Refine-Q
                        _mask = self._rng.uniform(low=0.0, high=1.0, size=_batch) < kwargs["epsilon"]["critic"]
                        self._selectionQ_mask = _mask
                        if sum(_mask) != 0:
                            a[_mask] = self.select_random_action(batch_size=sum(_mask), candidate_lists=topk_act[_mask])
            return a

    def _prep_update(self, actions, rewards):
        """
            Prepares the actions and rewards for the update method.
            - For discrete actions:
                topk_acts: [batch_size, num_queries (i.e. K), num_NNs (i.e. 1)] → int
                list_actions: [batch_size, num_queries, dim_action] → float
                actions: [batch_size, 1] → int: index of the action taken in the total action space
                ind_actions: [batch_size, 1] → int: index of the action taken in the number of queries
                query_max: [batch_size, dim_action] → float: best nearest neighbor “true” action representation while action was taken.
            - For continuous actions:
                TODO: Add the continuous action space support
        """
        list_rewards = None
        if self._args['env_name'].lower().startswith("mujoco"):
            # NOTE: query is responsible for both mu_list_eps and mu_list_star
            actions, query, query_max = actions[:, :self._critic_dim_action], \
                actions[:, self._critic_dim_action:-self._critic_dim_action], \
                actions[:, -self._critic_dim_action:]
            # if (self._args["WOLP_if_ar"]) or (self._args["WOLP_if_joint_actor"]):
            # cddpg / auto-regressive actor / wolp-joint
            query = query.reshape([query.shape[0], self._args['WOLP_cascade_list_len'], -1])

        if not self._args['env_name'].lower().startswith("mujoco") and \
                (not self._args["WOLP_if_ar"]) and (not self._args["WOLP_if_joint_critic"]) and \
                (not self._args["WOLP_if_dual_critic"]) and self._args["agent_type"] != "wolp-sac":  # wolp
            topk_acts = actions[:, :, 2:self._args["WOLP_topK"] + 2].astype(np.int)  # batch x query x topK
            ind_actions = actions[:, :, 1].astype(np.int)
            actions = actions[:, :, 0].astype(np.int)
            query_max = None
            list_actions = None
            # actions, ind_actions, topk_acts = actions[:, 0, 0], actions[:, 0, 1], actions[:, 0, 2: -self._critic_dim_action]
            # actions = actions[:, None].astype(np.int)
            # ind_actions = ind_actions[:, None].astype(np.int)
            # topk_acts = topk_acts.astype(np.int)
        else:  # cddpg / auto-regressive actor / wolp-joint / wolp-dual-critic
            if not self._args['env_name'].lower().startswith("mujoco"):
                topk_acts = actions[:, :, 2:self._args["WOLP_topK"] + 2]  # batch x query x topK
                list_actions = actions[:, :, self._args["WOLP_topK"] + 2: -self._critic_dim_action]
                query_max = actions[:, 0, -self._critic_dim_action:].astype(np.float32)
                actions, ind_actions = actions[:, :, 0], actions[:, :, 1]
                actions = actions[:, 0][:, None].astype(np.int)  # See _add_buffer in main.py
                ind_actions = ind_actions[:, 0][:, None].astype(np.int)  # See _add_buffer in main.py
                topk_acts = topk_acts.astype(np.int)
                list_actions = list_actions.astype(np.float32)

            if self._args["WOLP_cascade_list_len"] > 1:
                if self._args["WOLP_cascade_type_list_reward"].lower() == "elementwise":
                    # Create the element-wise reward
                    if self._args['env_name'].lower().startswith("mujoco"):
                        raise NotImplementedError
                    list_rewards = list()
                    _rewards = rewards.detach().cpu().numpy()
                    for r, _topk_act, _action in zip(_rewards, topk_acts, actions):  # for each sample in batch
                        _r = np.zeros(self._args["WOLP_cascade_list_len"])
                        if r != 0.0:
                            for _index in range(self._args["WOLP_cascade_list_len"]):
                                if _action in _topk_act[_index, :].tolist():
                                    _r[_index] = r  # just use the env reward
                                    break
                        list_rewards.append(_r)
                    list_rewards = torch.tensor(np.asarray(list_rewards).astype(np.float32), device=self._device)
                elif self._args["WOLP_cascade_type_list_reward"].lower() == "last":
                    # Add the env reward only to the final list index
                    list_rewards = torch.zeros(rewards.size(0), self._args["WOLP_cascade_list_len"]).to(self._device)
                    list_rewards[(rewards != 0.0).view(-1), -1] = rewards[(rewards != 0.0).view(-1), -1]
        if self._args["WOLP_if_ar_selection_bonus"]:
            selection_bonus = list()
            for _topk_act, _action in zip(topk_acts, actions):  # for each sample in batch
                _r = np.zeros(self._args["WOLP_cascade_list_len"])
                for _index in range(self._args["WOLP_cascade_list_len"]):
                    if _action in _topk_act[_index, :].tolist():
                        _r[_index] = self._args["WOLP_ar_selection_bonus"]
                        break
                selection_bonus.append(_r)
        else:
            selection_bonus = None
        selection_bonus = torch.tensor(np.asarray(selection_bonus).astype(np.float32), device=self._device)

        if self._args['env_name'].lower().startswith("mujoco"):
            return actions, None, None, query, list_rewards, selection_bonus, query_max
        else:
            return actions, ind_actions, topk_acts, list_actions, list_rewards, selection_bonus, query_max

    def _update(self,
                input_dict: dict,
                next_input_dict: dict,
                actions: np.ndarray,
                rewards: torch.tensor,
                reversed_dones: torch.tensor,
                act_embed_base: BaseEmbedding, **kwargs):

        if self._args["WOLP_if_pairwise_distance_bonus"]:
            dist_bonus = rewards[:, 1][:, None]
            rewards = rewards[:, 0][:, None]

        actions, ind_actions, topk_acts, list_actions_all, list_rewards, selection_bonus, query_max = self._prep_update(
            actions=actions, rewards=rewards)

        if self._args['env_name'].lower().startswith("mujoco"):
            if list_actions_all is not None:  # wolp don't need this
                # Split list_actions
                list_actions_eps = list_actions_all[:, :, :self._critic_dim_action]
                list_actions_star = list_actions_all[:, :, self._critic_dim_action:]
                if self._args["WOLP_ar_use_star_for_update_conditioning"]:
                    list_actions = list_actions_star
                else:
                    list_actions = list_actions_eps
        else:
            list_actions = list_actions_all

        if kwargs["if_sel"]:
            # === Get next Q-vals
            if self._args["WOLP_if_original_wolp_target_compute"]:
                raise NotImplementedError # NOTE: Not used in current version.
                if self._args["WOLP_if_cem_actor"]:
                    mu = self._cem_get_query(state=input_dict["state_embed"], q_net=self.target_ref_critic,
                                             if_update=True, epsilon={"actor": 0.0, "critic": 0.0})
                else:
                    # Get the weight from Actor
                    if self._args["WOLP_if_ar"]:
                        _s = input_dict["state_embed"][:, None, :]
                    else:
                        _s = input_dict["state_embed"]
                    def knn_action_function(x):
                        if self._args["WOLP_topK"] > 1:
                            with torch.no_grad():
                                topk_embed = self._perform_kNN(input_dict["act_embed"], x, self._args["WOLP_topK"])[2]
                                # assert self._args["env_name"].lower() != 'recsim-data'
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                    _s = self.main_ref_critic_obs_enc(input_dict["state_embed"]).repeat(1, topk_embed.shape[1], 1)
                                else:
                                    _s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                                Q_vals = self.target_ref_critic(torch.cat([_s, topk_embed], dim=-1)).squeeze(-1)
                                # Find the best action from the topK actions
                                return topk_embed.gather(dim=1, index=Q_vals.argmax(dim=1)[:, None, None].repeat(1, 1, topk_embed.shape[-1]))
                        else:
                            return self._perform_kNN(input_dict["act_embed"], x, 1)[2]
                    mu = self.target_actor(_s,
                                        knn_function=knn_action_function if \
                                        (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                         and not self._args["env_name"].lower().startswith("mujoco")) \
                                        else None) # batch x num_query(or sequence-len) x dim_act
                assert self._args["env_name"].lower() != 'recsim-data'
                next_Q = self.target_ref_critic(
                    input_dict["state_embed"][:, None, :].repeat(1, mu.shape[1], 1), mu).squeeze(-1)
                if self._args["TwinQ"]:
                    next_Q_twin = self.target_ref_critic_twin(
                        input_dict["state_embed"][:, None, :].repeat(1, mu.shape[1], 1), mu).squeeze(-1)
                    next_Q = torch.min(next_Q, next_Q_twin)
            else:
                with torch.no_grad():
                    # Note for TwinQ: The _select_action function accounts for TwinQ
                    next_Q, _mean = self._select_action(input_dict=next_input_dict, if_get_Q_vals=True,
                                                        if_update=True, epsilon={"actor": 0.0, "critic": 0.0})
                    next_select_Q_max, next_select_Q_mean = deepcopy(next_Q), _mean

            # ==== Get Taken Q-vals
            # Get the associated item-embedding for candidates for Refine-Q(batch x num-candidates)
            if self._args["WOLP_if_refineQ_single_action_update"]:
                if self._args["env_name"].lower().startswith("mujoco"):
                    action_embed = torch.tensor(actions, device=self._device)
                else:
                    action_embed = act_embed_base.get(index=torch.tensor(actions, device=self._device))  # b x 1 x dim
            else:
                assert not self._args["env_name"].lower().startswith("mujoco")
                if self._args["WOLP_if_ar"] or self._args["WOLP_if_joint_critic"] or self._args["WOLP_if_dual_critic"]:
                    topk_acts = topk_acts.reshape(topk_acts.shape[0],
                                                  topk_acts.shape[1] * topk_acts.shape[2])  # b x candidate
                action_embed = act_embed_base.get(index=topk_acts).detach()  # batch x topk x dim_act

            if self._args["WOLP_if_refineQ_single_action_update"]:
                if self._args["env_name"].lower().startswith("mujoco"):
                    if self._args["REDQ"]:
                        Q_vals = []
                        for i in range(self._args["REDQ_num"]):
                            Q_vals.append(
                                self.main_ref_critic_redq_list[i](torch.cat([input_dict["state_embed"], action_embed], dim=-1))
                            )
                        Q_vals = torch.stack(Q_vals, dim=-1)
                    else:
                        Q = self.main_ref_critic(torch.cat([input_dict["state_embed"], action_embed], dim=-1))
                        if self._args["TwinQ"]:
                            Q_twin = self.main_ref_critic_twin(torch.cat([input_dict["state_embed"], action_embed], dim=-1))
                else:
                    assert not self._args["REDQ"]
                    if self._args["env_name"].lower() == "recsim-data":
                        _s = self.main_ref_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
                    else:
                        _s = input_dict["state_embed"]
                    Q = self.main_ref_critic(torch.cat([_s, action_embed[:, 0, :]], dim=-1))
                    if self._args["TwinQ"]:
                        if self._args["env_name"].lower() == "recsim-data":
                            _s_twin = self.main_ref_critic_obs_enc_twin(input_dict["state_embed"]).squeeze(1)
                        else:
                            _s_twin = input_dict["state_embed"]
                        Q_twin = self.main_ref_critic_twin(torch.cat([_s_twin, action_embed[:, 0, :]], dim=-1))
            else:
                assert not self._args["env_name"].lower().startswith("mujoco")
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    _s = self.main_ref_critic_obs_enc(input_dict["state_embed"]).repeat(1, topk_acts.shape[1], 1)
                else:
                    _s = input_dict["state_embed"][:, None, :].repeat(1, topk_acts.shape[1], 1)
                q = self.main_ref_critic(torch.cat([_s, action_embed], dim=-1)).squeeze(-1)  # batch x candidate
                assert min(ind_actions) >= 0
                Q = q.gather(dim=-1, index=torch.tensor(ind_actions).to(device=self._device))
                if self._args["TwinQ"]:
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s_twin = self.main_ref_critic_obs_enc_twin(input_dict["state_embed"]).repeat(1, topk_acts.shape[1], 1)
                    else:
                        _s_twin = input_dict["state_embed"][:, None, :].repeat(1, topk_acts.shape[1], 1)
                    q_twin = self.main_ref_critic_twin(torch.cat([_s_twin, action_embed], dim=-1)).squeeze(
                        -1)  # batch x candidate
                    Q_twin = q_twin.gather(dim=-1, index=torch.tensor(ind_actions).to(device=self._device))
                    q = torch.min(q, q_twin)

                with torch.no_grad():
                    # Note: q is Q-vals on candidate-set whereas Q is Q-val of the taken action!!
                    select_Q_max, select_Q_mean = q.max(dim=-1).values, q.mean(dim=-1)

            # === Bellman Error
            target = rewards + next_Q * reversed_dones * self._args["Qnet_gamma"]
            if self._args["REDQ"]:
                value_loss = torch.nn.functional.mse_loss(Q_vals, target.unsqueeze(-1).repeat(1, 1, self._args["REDQ_num"]))
            else:
                value_loss = torch.nn.functional.mse_loss(Q, target)

            if self.opt_ref_critic_obs_enc is not None: self.opt_ref_critic_obs_enc.zero_grad()

            if self._args["REDQ"]:
                for i in range(self._args["REDQ_num"]):
                    self.opt_ref_critic_redq_list[i].zero_grad()
                value_loss.backward()
                if self._args["if_grad_clip"]:
                    [clip_grad(_critic.parameters(), 1.0) for _critic in self.main_ref_critic_redq_list]
                [self.opt_ref_critic_redq_list[i].step() for i in range(self._args["REDQ_num"])]
            else:
                self.opt_ref_critic.zero_grad()
                value_loss.backward()
                if self._args["if_grad_clip"]: clip_grad(self.main_ref_critic.parameters(), 1.0)  # gradient clipping
                if self.opt_ref_critic_obs_enc is not None and self._args["if_grad_clip"]:
                    clip_grad(self.main_ref_critic_obs_enc.parameters(), 1.0)
                self.opt_ref_critic.step()
                if self.opt_ref_critic_obs_enc is not None: self.opt_ref_critic_obs_enc.step()

                if self._args["TwinQ"]:
                    value_loss_twin = torch.nn.functional.mse_loss(Q_twin, target)
                    if self.opt_ref_critic_obs_enc_twin is not None: self.opt_ref_critic_obs_enc_twin.zero_grad()
                    self.opt_ref_critic_twin.zero_grad()
                    value_loss_twin.backward()
                    if self._args["if_grad_clip"]: clip_grad(self.main_ref_critic_twin.parameters(),
                                                            1.0)  # gradient clipping
                    if self.opt_ref_critic_obs_enc_twin is not None and self._args["if_grad_clip"]:
                        clip_grad(self.main_ref_critic_obs_enc_twin.parameters(), 1.0)
                    self.opt_ref_critic_twin.step()
                    if self.opt_ref_critic_obs_enc_twin is not None: self.opt_ref_critic_obs_enc_twin.step()

        if kwargs["if_ret"]:
            # ============ Update Extra Critic
            # import pudb; pudb.start()
            # Handle bonuses
            if self._args["WOLP_if_new_list_obj"][0] == "r":
                pass
            elif self._args["WOLP_if_new_list_obj"][0] == "max":
                assert not self._args["env_name"].lower().startswith("mujoco")
                rewards = select_Q_max[:, None]
                list_rewards = select_Q_max[:, None]
            elif self._args["WOLP_if_new_list_obj"][0] == "ave":
                assert not self._args["env_name"].lower().startswith("mujoco")
                rewards = select_Q_mean[:, None]
                list_rewards = select_Q_mean[:, None]

            if self._args["WOLP_if_ar_selection_bonus"]:
                rewards += selection_bonus
                list_rewards += selection_bonus
            if self._args["WOLP_if_pairwise_distance_bonus"]:
                rewards += dist_bonus
                list_rewards += dist_bonus

            if self._args["WOLP_if_dual_critic"]:
                with torch.no_grad():
                    if self._args["WOLP_if_dual_critic_imitate"]:
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                            'mw_obs_flatten']:
                            _s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                        else:
                            _s = input_dict["state_embed"]
                        if not self._args["env_name"].lower().startswith("mujoco") and self._args["WOLP_if_dual_critic_kNN_target"]:
                            knn_list_actions = self._perform_kNN(input_dict["act_embed"],
                                                                 torch.tensor(list_actions, device=self._device),
                                                                 self._args["WOLP_topK"])[2]
                            _s = _s[:, None, :].repeat(1, knn_list_actions.shape[1], 1)
                            _target = self.target_ref_critic(torch.cat([_s, knn_list_actions], dim=-1)).squeeze(-1)
                            if self._args["TwinQ"] and self._args["WOLP_twin_target"]:
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                else:
                                    _s_twin = input_dict["state_embed"]
                                _s_twin = _s_twin[:, None, :].repeat(1, knn_list_actions.shape[1], 1)
                                _target_twin = self.target_ref_critic_twin(
                                    torch.cat([_s_twin, knn_list_actions], dim=-1)).squeeze(-1)
                                _target = torch.min(_target, _target_twin)
                            target = _target.max(dim=-1).values.unsqueeze(-1)
                        else:
                            target = self.target_ref_critic(
                                torch.cat([_s, torch.tensor(query_max, device=self._device)], dim=-1))
                            if self._args["TwinQ"]:
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                else:
                                    _s_twin = input_dict["state_embed"]
                                target_twin = self.target_ref_critic_twin(
                                    torch.cat([_s_twin, torch.tensor(query_max, device=self._device)], dim=-1))
                                target = torch.min(target, target_twin)
                    else:
                        # Calculate next_Q to be used for target later
                        if self._args["WOLP_refineQ_target"]:
                            next_Q = next_select_Q_max
                        else:
                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                'mw_obs_flatten']:
                                _s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                            else:
                                if self._args["WOLP_if_ar"]:
                                    _s = next_input_dict["state_embed"][:, None, :]
                                else:
                                    _s = next_input_dict["state_embed"]
                            # Find target action at the next state
                            raise NotImplementedError
                            # TODO: Just need to specify the correct knn_action_function if needed
                            mu = self.target_actor(_s,
                                knn_function=(lambda x: self._perform_kNN(input_dict["act_embed"], x, 1)[2]) if \
                                (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                    and not self._args["env_name"].lower().startswith("mujoco")) \
                                else None).squeeze(1)  # bat x dim_act

                            if self._args["WOLP_if_ar"]:
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s = self.target_extra_critic_obs_enc(next_input_dict["state_embed"])
                                    _s = _s.repeat(1, mu.shape[1], 1)
                                    if self._args["TwinQ"]:
                                        _s_twin = self.target_extra_critic_obs_enc_twin(next_input_dict["state_embed"])
                                        _s_twin = _s_twin.repeat(1, mu.shape[1], 1)
                                else:
                                    _s = next_input_dict["state_embed"][:, None, :].repeat(1, mu.shape[1], 1)
                                    if self._args["TwinQ"]:
                                        _s_twin = next_input_dict["state_embed"][:, None, :].repeat(1, mu.shape[1], 1)
                            else:
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s = self.target_extra_critic_obs_enc(next_input_dict["state_embed"])
                                    if self._args["TwinQ"]:
                                        _s_twin = self.target_extra_critic_obs_enc_twin(next_input_dict["state_embed"])
                                else:
                                    _s = next_input_dict["state_embed"]
                                    if self._args["TwinQ"]:
                                        _s_twin = next_input_dict["state_embed"]
                            next_Q = self.target_extra_critic(torch.cat([_s, mu], dim=-1))  # b x 1
                            if self._args["TwinQ"]:
                                next_Q_twin = self.target_extra_critic_twin(torch.cat([_s_twin, mu], dim=-1))  # b x 1
                                next_Q = torch.min(next_Q, next_Q_twin)
                        # === Bellman Error
                        if self._args["WOLP_if_ar"]:
                            next_Q = next_Q.view(Q.shape[0], Q.shape[1])
                        target = rewards + next_Q * reversed_dones * self._args["retrieve_Qnet_gamma"]

                if not self._args["WOLP_if_ar"]:
                    batch, list_len, dim_action = list_actions.shape
                    list_actions = list_actions.reshape(batch, list_len * dim_action)
                mu = torch.tensor(list_actions, device=self._device)

                # import pudb; pudb.start()
                if self._args["WOLP_if_ar"]:
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        # _s = self.main_extra_critic_obs_enc(input_dict["state_embed"]).repeat(1, mu.shape[1], 1)
                        _s = self.main_extra_critic_obs_enc(input_dict["state_embed"])[:, None, :].repeat(1, mu.shape[1], 1)
                    else:
                        _s = input_dict["state_embed"][:, None, :].repeat(1, mu.shape[1], 1)
                else:
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_extra_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
                    else:
                        _s = input_dict["state_embed"]
                Q = self.main_extra_critic(torch.cat([_s, mu], dim=-1))
                if self._args["TwinQ"]:
                    if self._args["WOLP_if_ar"]:
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            # _s_twin = self.main_extra_critic_obs_enc_twin(input_dict["state_embed"]).repeat(1, mu.shape[1], 1)
                            _s_twin = self.main_extra_critic_obs_enc_twin(input_dict["state_embed"])[:, None, :].repeat(1, mu.shape[1], 1)
                        else:
                            _s_twin = input_dict["state_embed"][:, None, :].repeat(1, mu.shape[1], 1)
                    else:
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            _s_twin = self.main_extra_critic_obs_enc_twin(input_dict["state_embed"]).squeeze(1)
                        else:
                            _s_twin = input_dict["state_embed"]
                    Q_twin = self.main_extra_critic_twin(torch.cat([_s_twin, mu], dim=-1))
                # === Bellman Error
                if self._args["WOLP_if_ar"]:
                    Q = Q.view(Q.shape[0], Q.shape[1])
                    if self._args["TwinQ"]:
                        Q_twin = Q_twin.view(Q_twin.shape[0], Q_twin.shape[1])

                extra_value_loss = torch.nn.functional.mse_loss(Q, target)

                self.opt_extra_critic.zero_grad()
                if self.opt_extra_critic_obs_enc is not None: self.opt_extra_critic_obs_enc.zero_grad()
                extra_value_loss.backward()
                if self._args["if_grad_clip"]:
                    clip_grad(self.main_extra_critic.parameters(), 1.0)
                    if self.main_extra_critic_obs_enc is not None:
                        clip_grad(self.main_extra_critic_obs_enc.parameters(), 1.0)
                self.opt_extra_critic.step()
                if self.opt_extra_critic_obs_enc is not None: self.opt_extra_critic_obs_enc.step()

                if self._args["TwinQ"]:
                    extra_value_loss_twin = torch.nn.functional.mse_loss(Q_twin, target)

                    self.opt_extra_critic_twin.zero_grad()
                    if self.opt_extra_critic_obs_enc_twin is not None: self.opt_extra_critic_obs_enc_twin.zero_grad()
                    extra_value_loss_twin.backward()
                    if self._args["if_grad_clip"]:
                        clip_grad(self.main_extra_critic_twin.parameters(), 1.0)
                        if self.main_extra_critic_obs_enc_twin is not None:
                            clip_grad(self.main_extra_critic_obs_enc_twin.parameters(), 1.0)
                    self.opt_extra_critic_twin.step()
                    if self.opt_extra_critic_obs_enc_twin is not None: self.opt_extra_critic_obs_enc_twin.step()

            if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
                if self._args["WOLP_if_ar_imitate"]:
                    # NOTE: This requires k-NN search before computing the refineQ values. - query_max does that
                    with torch.no_grad():
                        # assert not self._args['env_name'].lower() == "recsim-data", "Not implemented yet!"
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            _s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                        else:
                            _s = input_dict["state_embed"]
                        _target = self.target_ref_critic(
                            torch.cat([_s, torch.tensor(query_max, device=self._device)], dim=-1))
                        # TODO: Add fresh_update here: basically, first compute the max and then use TwinQ regardless
                        if self._args["TwinQ"] and self._args["WOLP_twin_target"]:
                            _target_twin = self.target_ref_critic_twin(
                                torch.cat([_s, torch.tensor(query_max, device=self._device)], dim=-1))
                            _target = torch.min(_target, _target_twin)
                        # NOTE: Compute the target based on the maximum of IDs upto i.
                        next_Q = _target
                elif self._args["WOLP_refineQ_target"]:
                    next_Q = next_select_Q_max
                else:
                    # import pudb; pudb.start()
                    with torch.no_grad():
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                            'mw_obs_flatten']:
                            _s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                        else:
                            _s = next_input_dict["state_embed"]

                        mu = self.target_actor(_s, if_joint_update=True)  # bat x list * dim_act

                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                            'mw_obs_flatten']:
                            _s = self.target_joint_critic_obs_enc(next_input_dict["state_embed"])
                        else:
                            _s = next_input_dict["state_embed"]
                        next_Q = self.target_joint_critic(torch.cat([_s, mu], dim=-1))  # b x 1
                        if self._args["TwinQ"]:
                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                'mw_obs_flatten']:
                                _s_twin = self.target_joint_critic_obs_enc_twin(next_input_dict["state_embed"])
                            else:
                                _s_twin = next_input_dict["state_embed"]
                            next_Q_twin = self.target_joint_critic_twin(torch.cat([_s_twin, mu], dim=-1))
                            next_Q = torch.min(next_Q, next_Q_twin)

                # Avoid backprop to Actor
                batch, list_len, dim_action = list_actions.shape
                list_actions = list_actions.reshape(batch, list_len * dim_action)
                mu = torch.tensor(list_actions, device=self._device)

                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    _s = self.main_joint_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
                else:
                    _s = input_dict["state_embed"]
                Q = self.main_joint_critic(torch.cat([_s, mu], dim=-1))
                if self._args["TwinQ"]:
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s_twin = self.main_joint_critic_obs_enc_twin(input_dict["state_embed"]).squeeze(1)
                    else:
                        _s_twin = input_dict["state_embed"]
                    Q_twin = self.main_joint_critic_twin(torch.cat([_s_twin, mu], dim=-1))

                # === Bellman Error
                if self._args["WOLP_if_ar_imitate"]:
                    target = next_Q
                else:
                    target = rewards + next_Q * reversed_dones * self._args["retrieve_Qnet_gamma"]
                joint_value_loss = torch.nn.functional.mse_loss(Q, target)

                if self.opt_joint_critic_obs_enc is not None: self.opt_joint_critic_obs_enc.zero_grad()
                self.opt_joint_critic.zero_grad()
                joint_value_loss.backward()
                if self._args["if_grad_clip"]: clip_grad(self.main_joint_critic.parameters(), 1.0)
                if self.opt_joint_critic_obs_enc is not None and self._args["if_grad_clip"]:
                    clip_grad(self.main_joint_critic_obs_enc.parameters(), 1.0)
                self.opt_joint_critic.step()
                if self.opt_joint_critic_obs_enc is not None: self.opt_joint_critic_obs_enc.step()

                if self._args["TwinQ"]:
                    joint_value_loss_twin = torch.nn.functional.mse_loss(Q_twin, target)

                    if self.opt_joint_critic_obs_enc_twin is not None: self.opt_joint_critic_obs_enc_twin.zero_grad()
                    self.opt_joint_critic_twin.zero_grad()
                    joint_value_loss_twin.backward()
                    if self._args["if_grad_clip"]: clip_grad(self.main_joint_critic_twin.parameters(), 1.0)
                    if self.opt_joint_critic_obs_enc_twin is not None and self._args["if_grad_clip"]:
                        clip_grad(self.main_joint_critic_obs_enc_twin.parameters(), 1.0)
                    self.opt_joint_critic_twin.step()
                    if self.opt_joint_critic_obs_enc_twin is not None: self.opt_joint_critic_obs_enc_twin.step()

            if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
                # NOTE: no critic other than refinement critic needs training when WOLP_no_ar_critics is True
                if self._args["WOLP_if_joint_critic"]:
                    if self._args["WOLP_refineQ_target"]:
                        next_Q = next_select_Q_max
                    else:
                        raise NotImplementedError
                        # TODO: Just need to specify the correct knn_action_function if needed
                        with torch.no_grad():
                            if self._args["WOLP_cascade_list_len"] > 1:
                                # === Next-Q on Current list: Proto-action
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                                else:
                                    _s = next_input_dict["state_embed"][:, None, :]

                                # Get the weight from Actor
                                mu = self.target_actor(_s,
                                        knn_function=(lambda x: self._perform_kNN(input_dict["act_embed"], x, 1)[2]) if \
                                        (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                         and not self._args["env_name"].lower().startswith("mujoco")) \
                                        else None) # batch x sequence-len x dim_act
                                mu = mu.view(mu.shape[0], mu.shape[1] * mu.shape[2])
                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                'mw_obs_flatten']:
                                _s = self.target_ar_critic_obs_enc(next_input_dict["state_embed"]).squeeze(1)
                            else:
                                _s = next_input_dict["state_embed"]
                            next_Q = self.target_ar_critic(torch.cat([_s, mu], dim=-1))  # b x 1
                            if self._args["TwinQ"]:
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s_twin = self.target_ar_critic_obs_enc_twin(next_input_dict["state_embed"]).squeeze(1)
                                else:
                                    _s_twin = next_input_dict["state_embed"]
                                next_Q_twin = self.target_ar_critic_twin(torch.cat([_s_twin, mu], dim=-1))
                                next_Q = torch.min(next_Q, next_Q_twin)

                    # Reshape list input
                    batch, list_len, dim_action = list_actions.shape
                    list_actions = list_actions.reshape(batch, list_len * dim_action)
                    mu = torch.tensor(list_actions, device=self._device)

                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_ar_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
                    else:
                        _s = input_dict["state_embed"]
                    Q = self.main_ar_critic(torch.cat([_s, mu], dim=-1))
                    if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            _s_twin = self.main_ar_critic_obs_enc_twin(input_dict["state_embed"]).squeeze(1)
                        else:
                            _s_twin = input_dict["state_embed"]
                        Q_twin = self.main_ar_critic_twin(torch.cat([_s_twin, mu], dim=-1))
                else:
                    if self._args["WOLP_ar_if_use_full_next_Q"]:  # for wolp-MultiAgent
                        raise NotImplementedError
                        # TODO:
                        # 1. Just need to specify the correct knn_action_function if needed
                        # 2. Need to correct the objective for training to align with imitate
                        if self._args["WOLP_refineQ_target"]:
                            next_Q = next_select_Q_max
                        else:
                            with torch.no_grad():
                                # === Next-Q on Next List: Proto-action
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                                else:
                                    _s = next_input_dict["state_embed"][:, None, :]

                                # Get the weight from Actor
                                mu = self.target_actor(_s,
                                        knn_function=(lambda x: self._perform_kNN(input_dict["act_embed"], x, 1)[2]) if \
                                        (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                         and not self._args["env_name"].lower().startswith("mujoco")) \
                                        else None)  # batch x sequence-len x dim_act

                                # === Next-Q: Critic
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                    'mw_obs_flatten']:
                                    _s = self.target_ar_critic_obs_enc(next_input_dict["state_embed"])
                                    if self._args["env_name"].lower() == "recsim-data":
                                        _s = _s[:, None, :]
                                else:
                                    _s = next_input_dict["state_embed"][:, None, :]
                                next_Q = self.target_ar_critic(state=_s, action_seq=mu)  # b x seq
                                if self._args["TwinQ"]:
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                        'mw_obs_flatten']:
                                        _s_twin = self.target_ar_critic_obs_enc_twin(next_input_dict["state_embed"])
                                        if self._args["env_name"].lower() == "recsim-data":
                                            _s_twin = _s_twin[:, None, :]
                                    else:
                                        _s_twin = next_input_dict["state_embed"][:, None, :]
                                    next_Q_twin = self.target_ar_critic_twin(state=_s_twin, action_seq=mu)
                                    next_Q = torch.min(next_Q, next_Q_twin)
                    else:  # for all other ARDDPGs
                        if self._args["WOLP_refineQ_target"] and not self._args["WOLP_if_ar_imitate"]:
                            next_Q = next_select_Q_max.repeat(1, list_actions.shape[1])
                        elif not self._args["WOLP_if_ar_imitate"]:
                            raise NotImplementedError # We only deal with `WOLP_if_ar_imitate == True` now.
                            # TODO: Just need to specify the correct knn_action_function if needed
                            with torch.no_grad():
                                if self._args["WOLP_cascade_list_len"] > 1:
                                    if self._args["WOLP_ar_type_listwise_update"].lower() == "next-ts-same-index":
                                        state = next_input_dict["state_embed"]
                                    else:
                                        state = input_dict["state_embed"]

                                    # === Next-Q on Current list: Proto-action
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                        'mw_obs_flatten']:
                                        _s = self.target_actor_obs_enc(state)
                                        if self._args["env_name"].lower() == "recsim-data":
                                            _s = _s[:, None, :]
                                    else:
                                        _s = state[:, None, :]

                                    # Get the weight from Actor
                                    mu = self.target_actor(_s,
                                        knn_function=(lambda x: self._perform_kNN(input_dict["act_embed"], x, 1)[2]) if \
                                        (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                         and not self._args["env_name"].lower().startswith("mujoco")) \
                                        else None)  # batch x sequence-len x dim_act

                                    # === Next-Q on Current list: Critic
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                        'mw_obs_flatten']:
                                        _s = self.target_ar_critic_obs_enc(state)
                                        if self._args["env_name"].lower() == "recsim-data":
                                            _s = _s[:, None, :]
                                    else:
                                        _s = state[:, None, :]
                                    next_Q = self.target_ar_critic(state=_s, action_seq=mu)  # b x seq
                                    if self._args["TwinQ"]:
                                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                            'mw_obs_flatten']:
                                            _s_twin = self.target_ar_critic_obs_enc_twin(state)
                                            if self._args["env_name"].lower() == "recsim-data":
                                                _s_twin = _s_twin[:, None, :]
                                        else:
                                            _s_twin = state[:, None, :]
                                        next_Q_twin = self.target_ar_critic_twin(state=_s_twin, action_seq=mu)
                                        next_Q = torch.min(next_Q, next_Q_twin)

                                if self._args["WOLP_ar_type_listwise_update"].lower() != "next-ts-same-index":
                                    # === Next-Q on Next list: Proto-action
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                        'mw_obs_flatten']:
                                        _s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                                        if self._args["env_name"].lower() == "recsim-data":
                                            _s = _s[:, None, :]
                                    else:
                                        _s = next_input_dict["state_embed"][:, None, :]

                                    # Get the weight from Actor
                                    mu = self.target_actor(_s, if_next_list_Q=True,
                                        knn_function=(lambda x: self._perform_kNN(input_dict["act_embed"], x, 1)[2]) if \
                                        (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                         and not self._args["env_name"].lower().startswith("mujoco")) \
                                        else None)  # batch x 1 x dim_act

                                    # === Next-Q on Next list: Critic
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                        'mw_obs_flatten']:
                                        _s = self.target_ar_critic_obs_enc(next_input_dict["state_embed"])
                                        if self._args["env_name"].lower() == "recsim-data":
                                            _s = _s[:, None, :]
                                    else:
                                        _s = next_input_dict["state_embed"][:, None, :]
                                    last_next_Q = self.target_ar_critic(state=_s, action_seq=mu, if_next_list_Q=True)
                                    if self._args["TwinQ"]:
                                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                            'mw_obs_flatten']:
                                            _s_twin = self.target_ar_critic_obs_enc_twin(next_input_dict["state_embed"])
                                            if self._args["env_name"].lower() == "recsim-data":
                                                _s_twin = _s_twin[:, None, :]
                                        else:
                                            _s_twin = next_input_dict["state_embed"][:, None, :]
                                        last_next_Q_twin = self.target_ar_critic_twin(state=_s_twin, action_seq=mu,
                                                                                      if_next_list_Q=True)
                                        last_next_Q = torch.min(last_next_Q, last_next_Q_twin)

                            if self._args["WOLP_cascade_list_len"] > 1 and self._args["WOLP_if_ar_critic_cascade"]:
                                if self._args["WOLP_ar_type_listwise_update"].lower() == "next-list-index":
                                    next_Q = torch.cat([next_Q[:, 1:], last_next_Q], dim=-1)
                                elif self._args["WOLP_ar_type_listwise_update"].lower() == "0-th-next-ts":
                                    next_Q = last_next_Q.repeat(1, list_actions.shape[1])
                                elif self._args["WOLP_ar_type_listwise_update"].lower() == "next-ts-same-index":
                                    pass
                            else:  # others + full / contextual
                                next_Q = last_next_Q

                            # if replace the last list index w/h reward then it's bandit
                            # if all R and don't change the last list index then RL
                            if self._args["WOLP_if_new_list_obj"][1] == "bandit":
                                # empty the next-Q val on the last index for the ave-Q-reward as env reward
                                next_Q[:, -1] = 0

                            if self._args["WOLP_if_ar_contextual_prop"]:  # Only use the last entry and rely on BPTT!
                                next_Q, Q = next_Q[:, -1][:, None], Q[:, -1][:, None]
                                if self._args["WOLP_cascade_list_len"] > 1 and \
                                        self._args["WOLP_cascade_type_list_reward"] in ["elementwise", "last"]:
                                    list_rewards = list_rewards[:, -1][:, None]

                    # === Q on Taken list-action: Critic
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_ar_critic_obs_enc(input_dict["state_embed"])
                        if self._args["env_name"].lower() == "recsim-data":
                            _s = _s[:, None, :]
                    else:
                        _s = input_dict["state_embed"][:, None, :]
                    if self._args["TwinQ"]:
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            _s_twin = self.main_ar_critic_obs_enc_twin(input_dict["state_embed"])
                            if self._args["env_name"].lower() == "recsim-data":
                                _s_twin = _s_twin[:, None, :]
                        else:
                            _s_twin = input_dict["state_embed"][:, None, :]
                    if self._args["WOLP_Q_on_true_action"] and not self._args["WOLP_if_ar_imitate"]:
                        raise NotImplementedError # We only deal with `WOLP_if_ar_imitate == True` now.
                        # NOTE: Just need to specify the correct knn_action_function if needed
                        if self._args['env_name'].lower().startswith("mujoco"):
                            action_embed = torch.tensor(actions[:, None, :], device=self._device)

                        Q = self.main_ar_critic(state=_s,
                                                action_seq=torch.tensor(list_actions, device=self._device),
                                                true_action=action_embed,
                                                knn_function=(lambda x: self._perform_kNN(input_dict["act_embed"], x, 1)[2]) if \
                                                (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                                    and not self._args["env_name"].lower().startswith("mujoco")) \
                                                else None)
                        if self._args["TwinQ"]:
                            Q_twin = self.main_ar_critic_twin(state=_s_twin,
                                                              action_seq=torch.tensor(list_actions,
                                                                                      device=self._device),
                                                              true_action=action_embed,
                                                            knn_function=(lambda x: self._perform_kNN(input_dict["act_embed"], x, 1)[2]) if \
                                                            (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                                                and not self._args["env_name"].lower().startswith("mujoco")) \
                                                            else None)
                    else:
                        if self._args["WOLP_if_ar"] and \
                            self._args["WOLP_knn_inside_cascade"] and \
                                    not self._args["env_name"].lower().startswith("mujoco"):
                            if self._args["WOLP_topK"] > 1:
                                with torch.no_grad():
                                    # TODO: Check if the reshaped version is correct
                                    topk_embed = self._perform_kNN(input_dict["act_embed"], torch.tensor(list_actions, device=self._device), self._args["WOLP_topK"])[2]
                                    topk_embed_reshape = topk_embed.reshape(topk_embed.shape[0],
                                                                            self._args["WOLP_cascade_list_len"],
                                                                            self._args["WOLP_topK"],
                                                                            topk_embed.shape[-1])
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                        ref_s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                                        if self._args["env_name"].lower() == "mine":
                                            ref_s = ref_s[:, :, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                        elif self._args["env_name"].lower() == "recsim-data":
                                            ref_s = ref_s[:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                    else:
                                        ref_s = input_dict["state_embed"][:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                    Q_vals = self.target_ref_critic(torch.cat([ref_s, topk_embed_reshape], dim=-1)).squeeze(-1)
                                    # TODO: Add fresh_update here: basically, first compute the max and then use TwinQ regardless
                                    if self._args["TwinQ"] and self._args["WOLP_twin_target"]:
                                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                            ref_s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                            if self._args["env_name"].lower() == "mine":
                                                ref_s_twin = ref_s_twin[:, :, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                            elif self._args["env_name"].lower() == "recsim-data":
                                                ref_s_twin = ref_s_twin[:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                        else:
                                            ref_s_twin = input_dict["state_embed"][:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                        Q_vals_twin = self.target_ref_critic_twin(torch.cat([ref_s_twin, topk_embed_reshape], dim=-1)).squeeze(-1)
                                        Q_vals = torch.min(Q_vals, Q_vals_twin)
                                    # Find the best action from the topK actions for each query in the list
                                    knn_list_actions = topk_embed_reshape.gather(dim=2, index=Q_vals.argmax(dim=2)[:, :, None, None].repeat(1, 1, 1, topk_embed_reshape.shape[-1])).squeeze(2)
                            else:
                                knn_list_actions = self._perform_kNN(input_dict["act_embed"],
                                                                    torch.tensor(list_actions, device=self._device),
                                                                    self._args["WOLP_topK"])[2]
                        else:
                            knn_list_actions = None
                        if self._args['env_name'].lower().startswith("mujoco") and \
                            (self._args["WOLP_ar_use_taken_action"] or self._args['WOLP_ar_critic_taken_action_update']):
                            action_embed = action_embed[:, None, :]
                        if self._args['WOLP_ar_critic_eps_action']:
                            # action_seq only used for true_action in this case.
                            assert not self._args["WOLP_ar_use_taken_action"]
                            action_seq = torch.tensor(list_actions_eps, device=self._device)
                        else:
                            action_seq = torch.tensor(list_actions, device=self._device)

                        if self._args['WOLP_ar_conditioning_star']:
                            assert self._args['env_name'].lower().startswith('mujoco')
                            alternate_conditioning = torch.tensor(list_actions_star, device=self._device)
                        else:
                            # NOTE: If none, then condition on action_seq itself
                            alternate_conditioning = knn_list_actions

                        Q = self.main_ar_critic(state=_s,
                                action_seq=action_seq,
                                alternate_conditioning=alternate_conditioning,
                                true_action=action_embed if self._args["WOLP_ar_use_taken_action"] else None)
                        if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                            Q_twin = self.main_ar_critic_twin(state=_s_twin,
                                action_seq=action_seq,
                                alternate_conditioning=alternate_conditioning,
                                true_action=action_embed if self._args["WOLP_ar_use_taken_action"] else None)

                        if self._args['WOLP_ar_critic_taken_action_update'] and self._args['WOLP_if_ar_imitate']:
                            if self._args["WOLP_if_0th_ref_critic"]:
                                start_index = 1
                            else:
                                start_index = 0
                            if start_index < self._args['WOLP_cascade_list_len']:
                                Q_taken = self.main_ar_critic(state=_s,
                                    action_seq=action_seq,
                                    alternate_conditioning=alternate_conditioning,
                                    true_action=action_embed,
                                    start_index=start_index)
                                Q = torch.cat([Q, Q_taken], dim=1)
                        elif self._args['WOLP_ar_critic_scaled_num_updates'] and self._args['WOLP_if_ar_imitate']:
                            # First we already computed the Q-values for the epsilon-added noisy actions: Qk(s, mu_k_eps | mu_1_star, ..., mu_k-1_star)
                            if self._args['WOLP_ar_critic_eps_action'] and self._args['WOLP_ar_conditioning_star']:
                                assert not self._args['WOLP_ar_use_taken_action']
                                # Now, iteratively compute the Q-values for Qk(s, mu_j_star | mu_1_star, ..., mu_k-1_star) for j = 1, ..., k-1
                                for j_ind in range(1, self._args['WOLP_cascade_list_len']):
                                    Q_j = self.main_ar_critic(state=_s,
                                                                action_seq=None,
                                                                alternate_conditioning=alternate_conditioning,
                                                                true_action=torch.tensor(list_actions_star[:, j_ind-1:j_ind, :], device=self._device), # NOTE: This is the action taken at the j_ind-1 index
                                                                start_index=j_ind,
                                                                )
                                    # Concatenate to the list-len dimension
                                    Q = torch.cat([Q, Q_j], dim=1)
                            else:
                                for j_ind in range(1, self._args['WOLP_cascade_list_len']):
                                    Q_j = self.main_ar_critic(state=_s,
                                                                action_seq=None,
                                                                alternate_conditioning=torch.tensor(list_actions, device=self._device),
                                                                true_action=torch.tensor(list_actions[:, j_ind-1:j_ind, :], device=self._device), # NOTE: This is the action taken at the j_ind-1 index
                                                                start_index=j_ind,
                                                                )
                                    # Concatenate to the list-len dimension
                                    Q = torch.cat([Q, Q_j], dim=1)

                # === Bellman Error
                if self._args["WOLP_if_ar_imitate"]:
                    with torch.no_grad():
                        # assert not self._args['env_name'].lower() == "recsim-data", "Not implemented yet!"
                        if self._args["WOLP_ar_use_query_max"]:
                            assert not self._args['env_name'].lower() == "recsim-data", "Not implemented yet!"
                            raise NotImplementedError
                            # NOTE: This is correct, but unused
                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                _s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                                raise NotImplementedError #Check dimensions and whether we need to expand
                            else:
                                _s = input_dict["state_embed"]
                            _target = self.target_ref_critic(
                                torch.cat([_s, torch.tensor(query_max, device=self._device)], dim=-1))
                            if self._args["TwinQ"]:
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                    _s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                    raise NotImplementedError #Check dimensions and whether we need to expand
                                else:
                                    _s_twin = input_dict["state_embed"]
                                _target_twin = self.target_ref_critic_twin(
                                    torch.cat([_s_twin, torch.tensor(query_max, device=self._device)], dim=-1))
                                _target = torch.min(_target, _target_twin)
                            target = _target.repeat(1, list_actions.shape[1])
                        else:
                            with torch.no_grad():
                                if self._args["WOLP_ar_conditioning_star"]:
                                    if self._args["WOLP_ar_critic_taken_action_update"]:
                                        assert not self._args["WOLP_ar_critic_eps_action"]
                                        modified_list_actions = torch.cat([
                                            torch.tensor(list_actions_star, device=self._device),
                                            action_embed,
                                        ], dim=1)
                                    elif self._args["WOLP_ar_use_taken_action"]:
                                        modified_list_actions = torch.cat([action_embed,
                                                                            torch.tensor(list_actions_star[:, :-1, :], device=self._device)],
                                                                            dim=1)
                                    elif self._args["WOLP_ar_critic_eps_action"]:
                                        # Q(s, mu_k_eps | mu_1_star, ..., mu_k-1_star)
                                        # = max[ Q*(s, mu_k_eps), max[ Q*(s, mu_1_star), ..., Q(s, mu_k-1_star)) ] ]
                                        modified_list_actions = torch.cat([
                                            torch.tensor(list_actions_star[:, :-1, :], device=self._device),
                                            torch.tensor(list_actions_eps, device=self._device),
                                            ], dim=1)
                                    else:
                                        modified_list_actions = torch.tensor(list_actions_star, device=self._device)
                                else:
                                    if self._args["WOLP_ar_critic_taken_action_update"]:
                                        modified_list_actions = torch.cat([
                                            torch.tensor(list_actions, device=self._device),
                                            action_embed,
                                        ], dim=1)
                                    elif self._args["WOLP_ar_use_taken_action"]:
                                        # NOTE: action_embed's shape would already be adjusted for mujoco above
                                        modified_list_actions = torch.cat([action_embed,
                                                                            torch.tensor(list_actions[:, :-1, :], device=self._device)],
                                                                            dim=1)
                                    else:
                                        modified_list_actions = torch.tensor(list_actions, device=self._device)
                                if self._args["WOLP_use_main_ref_critic_for_target"]:
                                    if self._args["env_name"].lower().startswith("mujoco"):
                                        topk_embed = modified_list_actions
                                    else:
                                        topk_embed = self._perform_kNN(input_dict["act_embed"],
                                            modified_list_actions, 1)[2]
                                    # Get Q-values for each of the topK actions
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                        ref_s = self.main_ref_critic_obs_enc(input_dict["state_embed"])
                                        if self._args["env_name"].lower() == "mine":
                                            ref_s = ref_s.repeat(1, topk_embed.shape[1], 1)
                                        elif self._args["env_name"].lower() == "recsim-data":
                                            ref_s = ref_s[:, None, :].repeat(1, topk_embed.shape[1], 1)
                                    else:
                                        ref_s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                                    if self._args['REDQ'] and self._args["REDQ_SAVO"]:
                                        _targets = []
                                        for i in range(self._args["REDQ_num"]):
                                            _targets.append(self.main_ref_critic_redq_list[i](torch.cat([ref_s, topk_embed], dim=-1)).squeeze(-1))
                                        _targets = torch.stack(_targets, dim=1)
                                        _target = _targets.mean(dim=1)
                                    else:
                                        _target = self.main_ref_critic(torch.cat([ref_s, topk_embed], dim=-1)).squeeze(-1)
                                        if self._args["TwinQ"] and self._args["WOLP_twin_target"] and self._args["WOLP_twin_main_for_target"]:
                                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                                ref_s_twin = self.main_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                                if self._args["env_name"].lower() == "mine":
                                                    ref_s_twin = ref_s_twin.repeat(1, topk_embed.shape[1], 1)
                                                elif self._args["env_name"].lower() == "recsim-data":
                                                    ref_s_twin = ref_s_twin[:, None, :].repeat(1, topk_embed.shape[1], 1)
                                            else:
                                                ref_s_twin = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                                            _target_twin = self.main_ref_critic_twin(torch.cat([ref_s_twin, topk_embed], dim=-1)).squeeze(-1)
                                            _target = torch.min(_target, _target_twin)
                                elif self._args["WOLP_ignore_knn_for_target"]:
                                    topk_embed = self._perform_kNN(input_dict["act_embed"],
                                        modified_list_actions, 1)[2]
                                    # Get Q-values for each of the topK actions
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                        ref_s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                                        if self._args["env_name"].lower() == "mine":
                                            ref_s = ref_s.repeat(1, topk_embed.shape[1], 1)
                                        elif self._args["env_name"].lower() == "recsim-data":
                                            ref_s = ref_s[:, None, :].repeat(1, topk_embed.shape[1], 1)
                                    else:
                                        ref_s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                                    _target = self.target_ref_critic(torch.cat([ref_s, topk_embed], dim=-1)).squeeze(-1)
                                    if self._args["TwinQ"]:
                                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                            ref_s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                            if self._args["env_name"].lower() == "mine":
                                                ref_s_twin = ref_s_twin.repeat(1, topk_embed.shape[1], 1)
                                            elif self._args["env_name"].lower() == "recsim-data":
                                                ref_s_twin = ref_s_twin[:, None, :].repeat(1, topk_embed.shape[1], 1)
                                        else:
                                            ref_s_twin = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                                        _target_twin = self.target_ref_critic_twin(torch.cat([ref_s_twin, topk_embed], dim=-1)).squeeze(-1)
                                        _target = torch.min(_target, _target_twin)
                                else:
                                    if not self._args["WOLP_ar_knn_action_rep"] or self._args["env_name"].lower().startswith("mujoco"):
                                        topk_embed_reshape = modified_list_actions[:, :, None, :]
                                    else:
                                        if self._args["WOLP_topK"] <= 1:
                                            topk_embed_reshape = self._perform_kNN(input_dict["act_embed"],
                                                modified_list_actions,
                                                self._args["WOLP_topK"])[2][:, :, None, :]
                                        else:
                                                topk_embed = self._perform_kNN(input_dict["act_embed"], modified_list_actions, self._args["WOLP_topK"])[2]
                                                topk_embed_reshape = topk_embed.reshape(topk_embed.shape[0],
                                                                                        self._args["WOLP_cascade_list_len"],
                                                                                        self._args["WOLP_topK"],
                                                                                        topk_embed.shape[-1])
                                    # Get Q-values for each of the topK actions
                                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                        ref_s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                                        if self._args["env_name"].lower() == "mine":
                                            ref_s = ref_s[:, :, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                        elif self._args["env_name"].lower() == "recsim-data":
                                            ref_s = ref_s[:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                    else:
                                        ref_s = input_dict["state_embed"][:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                    if self._args["WOLP_ar_critic_target_smoothing"]:
                                        target_noise = (torch.randn_like(topk_embed_reshape) * self._args['TD3_policy_noise']
                                                        ).clamp(-self._args['TD3_noise_clip'], self._args['TD3_noise_clip'])
                                        topk_embed_reshape = (topk_embed_reshape + target_noise).clamp(-self._args["env_max_action"], self._args["env_max_action"])
                                    _target = self.target_ref_critic(torch.cat([ref_s, topk_embed_reshape], dim=-1)).squeeze(-1)
                                    if self._args["WOLP_ar_fresh_update"]:
                                        max_actions = _target.max(dim=2)[1]
                                        target_a = topk_embed_reshape.gather(dim=2, index=max_actions[:, :, None, None].repeat(1, 1, 1, topk_embed_reshape.shape[-1])).squeeze(2)
                                        # TODO: Now use this target_a to compute the Q and TwinQ values with optional Target policy smoothing (because this Q-function is a DDPG function).
                                        if self._args["TwinQ"] and self._args["WOLP_ar_TD3_target_policy_smoothing"]:
                                            target_noise = (torch.randn_like(target_a) * self._args['TD3_policy_noise']
                                                            ).clamp(-self._args['TD3_noise_clip'], self._args['TD3_noise_clip'])
                                            target_a = (target_a + target_noise).clamp(-self._args["env_max_action"], self._args["env_max_action"])
                                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                                ref_s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                                if self._args["env_name"].lower() == "mine":
                                                    ref_s_twin = ref_s_twin.repeat(1, target_a.shape[1], 1)
                                                elif self._args["env_name"].lower() == "recsim-data":
                                                    assert False, "Needs dimensional verification!"
                                                    ref_s_twin = ref_s_twin[:, None, :].repeat(1, topk_embed_reshape.shape[1], 1)
                                            else:
                                                ref_s_twin = input_dict["state_embed"][:, None, :].repeat(1, target_a.shape[1], 1)
                                            _target = torch.min(
                                                self.target_ref_critic(torch.cat([ref_s[:, :, 0], target_a], dim=-1)),
                                                self.target_ref_critic_twin(torch.cat([ref_s_twin, target_a], dim=-1))
                                            ).squeeze(-1)
                                        elif self._args["TwinQ"]:
                                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                                ref_s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                                if self._args["env_name"].lower() == "mine":
                                                    ref_s_twin = ref_s_twin.repeat(1, target_a.shape[1], 1)
                                                elif self._args["env_name"].lower() == "recsim-data":
                                                    assert False, "Needs dimensional verification!"
                                                    ref_s_twin = ref_s_twin[:, None, :].repeat(1, topk_embed_reshape.shape[1], 1)
                                            else:
                                                ref_s_twin = input_dict["state_embed"][:, None, :].repeat(1, target_a.shape[1], 1)
                                            Q_vals_twin = self.target_ref_critic_twin(torch.cat([ref_s_twin, target_a], dim=-1))
                                            _target = torch.min(
                                                _target.gather(dim=2, index=max_actions[:, :, None]),
                                                Q_vals_twin
                                            ).squeeze(-1)
                                        else:
                                            _target = _target.gather(dim=2, index=max_actions[:, :, None]).squeeze(-1)
                                    else:
                                        # Get twin Q-values for each of the topK actions if needed
                                        if self._args["TwinQ"] and self._args["WOLP_twin_target"]:
                                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                                ref_s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                                                if self._args["env_name"].lower() == "mine":
                                                    ref_s_twin = ref_s_twin[:, :, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                                elif self._args["env_name"].lower() == "recsim-data":
                                                    ref_s_twin = ref_s_twin[:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                            else:
                                                ref_s_twin = input_dict["state_embed"][:, None, None, :].repeat(1, topk_embed_reshape.shape[1], topk_embed_reshape.shape[2], 1)
                                            _target_twin = self.target_ref_critic_twin(torch.cat([ref_s_twin, topk_embed_reshape], dim=-1)).squeeze(-1)
                                            _target = torch.min(_target, _target_twin)
                                        _target = _target.max(dim=2)[0]
                                        # NOTE: Compute the target based on the maximum of IDs upto i.
                                if self._args["WOLP_ar_conditioning_star"] and self._args["WOLP_ar_critic_eps_action"]:
                                    # Q(s, mu_k_eps | mu_1_star, ..., mu_k-1_star)
                                    # = max[ Q*(s, mu_k_eps), max[ Q*(s, mu_1_star), ..., Q(s, mu_k-1_star)) ] ]
                                    star_cummax = torch.cummax(_target[:, :list_actions_star.shape[1] - 1], dim=1)[0] # bs x K-1
                                    # Prepend a large negative value to star_cummax at the 1st dimension in the 0th position
                                    star_cummax_filled = torch.cat([torch.full((star_cummax.shape[0], 1), -1e8, device=self._device), star_cummax], dim=1)
                                    eps_target_vals = _target[:, (list_actions_star.shape[1]-1):] # bs x K
                                    if self._args['WOLP_ar_improvement_as_target']:
                                        star_cummax_0th_zero = torch.cat([torch.full((star_cummax.shape[0], 1), 0, device=self._device), star_cummax], dim=1)
                                        target = eps_target_vals - star_cummax_0th_zero
                                        if self._args["WOLP_if_min_improvement_0"]:
                                            target = torch.max(target, torch.zeros_like(target))
                                    else:
                                        # Take the max of the two
                                        target = torch.max(star_cummax_filled, eps_target_vals)
                                elif self._args['WOLP_ar_critic_taken_action_update']:
                                    if self._args["WOLP_ar_use_ref_next_Q"]:
                                        _target[:, -1] = next_select_Q_max.squeeze(-1)
                                    target_cummax = torch.cummax(_target[:, :-1], dim=1)[0] # bs x K

                                    _taken_target_vals = (_target[:, -1:]).repeat(1, self._args['WOLP_cascade_list_len']) # bs x K
                                    if self._args['WOLP_ar_improvement_as_target']:
                                        target_cummax_zero_filled = torch.cat([torch.full((target_cummax.shape[0], 1), 0, device=self._device), target_cummax[:, :-1]], dim=1)
                                        target_1 = _target[:, :-1] - target_cummax_zero_filled
                                        target_2 = _taken_target_vals - target_cummax_zero_filled
                                        if self._args["WOLP_if_0th_ref_critic"]:
                                            target = torch.cat([target_1, target_2[:, 1:]], dim=1) # bs x 2K-1
                                        else:
                                            target = torch.cat([target_1, target_2], dim=1) # bs x 2K
                                        if self._args["WOLP_if_min_improvement_0"]:
                                            target = torch.max(target, torch.zeros_like(target))
                                    else:
                                        target_cummax_filled = torch.cat([torch.full((target_cummax.shape[0], 1), -1e8, device=self._device), target_cummax[:, :-1]], dim=1)
                                        taken_target_vals = torch.max(_taken_target_vals, target_cummax_filled) # bs x K
                                        if self._args["WOLP_if_0th_ref_critic"]:
                                            target = torch.cat([target_cummax, taken_target_vals[:, 1:]], dim=1) # bs x 2K-1
                                        else:
                                            target = torch.cat([target_cummax, taken_target_vals], dim=1) # bs x 2K
                                else:
                                    if self._args["WOLP_ar_use_ref_next_Q"]:
                                        _target[:, 0] = next_select_Q_max.squeeze(-1)
                                    target = torch.cummax(_target, dim=1)[0]
                                    if self._args['WOLP_ar_improvement_as_target']:
                                        target = _target - torch.cat([torch.full((target.shape[0], 1), 0, device=self._device), target[:, :-1]], dim=1)
                                        if self._args["WOLP_if_min_improvement_0"]:
                                            target = torch.max(target, torch.zeros_like(target))
                                if self._args['WOLP_ar_critic_scaled_num_updates']:
                                    assert not self._args['WOLP_ar_critic_taken_action_update']
                                    if not (self._args["WOLP_ar_conditioning_star"] and self._args["WOLP_ar_critic_eps_action"]):
                                        star_cummax = torch.cummax(_target[:, :-1], dim=1)[0] # bs x K-1
                                    # Qk(s, mu_j_star | mu_1_star, ..., mu_k-1_star) = max{Q*(s, mu_1_star),..., Q*(s, mu_k-1_star)} for j = 1, ..., k-1
                                    for j_ind in range(1, self._args['WOLP_cascade_list_len']):
                                        if self._args['WOLP_ar_improvement_as_target']:
                                            # Qk(s, mu_j_star) - star_cummax
                                            star_targets = _target[:, (j_ind-1):j_ind] - star_cummax[:, (j_ind-1):]
                                            if self._args["WOLP_if_min_improvement_0"]:
                                                star_targets = torch.max(star_targets, torch.zeros_like(star_targets))
                                            target = torch.cat([target, star_targets], dim=1)
                                        else:
                                            target = torch.cat([target, star_cummax[:, (j_ind-1):]], dim=1)
                            # else:
                            #     if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            #         _s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                            #     else:
                            #         _s = input_dict["state_embed"]
                            #     _s = _s[:,None,:].repeat(1, knn_list_actions.shape[1], 1)
                            #     # _s = input_dict["state_embed"][:,None,:].repeat(1, knn_list_actions.shape[1], 1)
                            #     _target = self.target_ref_critic(
                            #         torch.cat([_s, torch.tensor(knn_list_actions, device=self._device)], dim=-1)).squeeze(-1)
                            #     if self._args["TwinQ"]:
                            #         if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                            #             _s_twin = self.target_ref_critic_obs_enc_twin(input_dict["state_embed"])
                            #         else:
                            #             _s_twin = input_dict["state_embed"]
                            #         _s_twin = _s_twin[:,None,:].repeat(1, knn_list_actions.shape[1], 1)
                            #         _target_twin = self.target_ref_critic_twin(
                            #             torch.cat([_s_twin, torch.tensor(knn_list_actions, device=self._device)], dim=-1)).squeeze(-1)
                            #         _target = torch.min(_target, _target_twin)
                            #     # NOTE: Compute the target based on the maximum of IDs upto i.
                            #     target = torch.cummax(_target, dim=1)[0]
                else:
                    if self._args["WOLP_discounted_cascading"]:
                        list_len = self._args["WOLP_cascade_list_len"]
                        discount_exponent = torch.tensor(list_len - np.arange(list_len),
                                                        dtype=torch.float32,
                                                        device=self._device)[None, :]
                        target = torch.pow(self._args["retrieve_Qnet_gamma"], discount_exponent - 1) * rewards + \
                                torch.pow(self._args["retrieve_Qnet_gamma"], discount_exponent) * next_Q * reversed_dones
                    elif self._args["WOLP_cascade_list_len"] > 1 and \
                            self._args["WOLP_cascade_type_list_reward"] in ["elementwise", "last"]:
                        target = list_rewards + next_Q * reversed_dones * self._args["retrieve_Qnet_gamma"]
                    else:
                        target = rewards + next_Q * reversed_dones * self._args["retrieve_Qnet_gamma"]
                if self._args["WOLP_if_ar"] and self._args["WOLP_if_0th_ref_critic"]:
                    # Don't train the 0-th ref critic again
                    ar_value_loss = torch.nn.functional.mse_loss(Q[:,1:], target[:,1:])
                    if self._args["WOLP_ar_value_loss_if_sum"]:
                        ar_value_loss *= (target.shape[1] - 1)
                else:
                    ar_value_loss = torch.nn.functional.mse_loss(Q, target)
                    if self._args["WOLP_ar_value_loss_if_sum"]:
                        ar_value_loss *= target.shape[1]

                if not torch.isnan(ar_value_loss):
                    # When len(list) == 1, the ar_value_loss is nan
                    if self.opt_ar_critic_obs_enc is not None: self.opt_ar_critic_obs_enc.zero_grad()
                    self.opt_ar_critic.zero_grad()
                    ar_value_loss.backward()
                    if self._args["if_grad_clip"]: clip_grad(self.main_ar_critic.parameters(), 1.0)
                    if self.opt_ar_critic_obs_enc is not None and self._args["if_grad_clip"]:
                        clip_grad(self.main_ar_critic_obs_enc.parameters(), 1.0)
                    # print([sum_of_grad(net=cell) for cell in self.main_ar_critic.cells])
                    self.opt_ar_critic.step()
                    if self.opt_ar_critic_obs_enc is not None: self.opt_ar_critic_obs_enc.step()

                if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                    if self._args["WOLP_if_ar"] and self._args["WOLP_if_0th_ref_critic"]:
                        # Don't train the 0-th ref twin critic again
                        ar_value_loss_twin = torch.nn.functional.mse_loss(Q_twin[:,1:], target[:,1:])
                    else:
                        ar_value_loss_twin = torch.nn.functional.mse_loss(Q_twin, target)

                    if not torch.isnan(ar_value_loss_twin):
                        if self.opt_ar_critic_obs_enc_twin is not None: self.opt_ar_critic_obs_enc_twin.zero_grad()
                        self.opt_ar_critic_twin.zero_grad()
                        ar_value_loss_twin.backward()
                        if self._args["if_grad_clip"]: clip_grad(self.main_ar_critic_twin.parameters(), 1.0)
                        if self.opt_ar_critic_obs_enc_twin is not None and self._args["if_grad_clip"]:
                            clip_grad(self.main_ar_critic_obs_enc_twin.parameters(), 1.0)
                        # print([sum_of_grad(net=cell) for cell in self.main_ar_critic.cells])
                        self.opt_ar_critic_twin.step()
                        if self.opt_ar_critic_obs_enc_twin is not None: self.opt_ar_critic_obs_enc_twin.step()

            # === Update Actor
            train_actor = (
                    self.main_actor is not None and \
                    self._args['global_ts'] > self._args['delayed_actor_training'])
            train_actor = train_actor and self.update_count % self._args['TD3_policy_delay'] == 0
            if train_actor:
                if self._args["WOLP_if_ar"]:
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_actor_obs_enc(input_dict["state_embed"])
                        if self._args["env_name"].lower() == "recsim-data":
                            _s = _s[:, None, :]
                    else:
                        _s = input_dict["state_embed"][:, None, :]
                    def knn_action_function(x):
                        if self._args["WOLP_topK"] > 1:
                            with torch.no_grad():
                                topk_embed = self._perform_kNN(input_dict["act_embed"], x, self._args["WOLP_topK"])[2]
                                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                                    _s = self.target_ref_critic_obs_enc(input_dict["state_embed"])
                                    if self._args["env_name"].lower() == "mine":
                                        _s = _s.repeat(1, topk_embed.shape[1], 1)
                                    elif self._args["env_name"].lower() == "recsim-data":
                                        _s = _s[:, None, :].repeat(1, topk_embed.shape[1], 1)
                                else:
                                    _s = input_dict["state_embed"][:, None, :].repeat(1, topk_embed.shape[1], 1)
                                Q_vals = self.target_ref_critic(torch.cat([_s, topk_embed], dim=-1)).squeeze(-1)
                                # Find the best action from the topK actions
                                return topk_embed.gather(dim=1, index=Q_vals.argmax(dim=1)[:, None, None].repeat(1, 1, topk_embed.shape[-1]))
                        else:
                            return self._perform_kNN(input_dict["act_embed"], x, 1)[2]
                    if self._args['WOLP_ar_actor_no_conditioning']:
                        alternate_conditioning = None
                    elif self._args['WOLP_ar_conditioning_star']:
                        assert self._args['env_name'].lower().startswith('mujoco')
                        alternate_conditioning = torch.tensor(list_actions_star, device=self._device)
                    elif self._args["WOLP_ar_use_taken_action"]:
                        alternate_conditioning = torch.tensor(list_actions, device=self._device)
                    else:
                        alternate_conditioning = None
                    mu = self.main_actor(_s,
                                        knn_function=knn_action_function if \
                                        (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                         and not self._args["env_name"].lower().startswith("mujoco")) \
                                        else None,
                                        alternate_conditioning=alternate_conditioning)

                    if self._args["WOLP_if_dual_critic"] and not self._args["WOLP_if_joint_actor"] \
                        and not(self._args["WOLP_if_ar"] and self._args["WOLP_no_ar_critics"]):
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                            'mw_obs_flatten']:
                            _s = self.main_extra_critic_obs_enc(input_dict["state_embed"])
                        else:
                            _s = input_dict["state_embed"][:, None, :]
                        _s = _s.repeat(1, mu.shape[1], 1)
                        Q = self.main_extra_critic(torch.cat([_s, mu], dim=-1))
                    elif not self._args["WOLP_if_dual_critic"] and self._args["WOLP_if_joint_critic"]:
                        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                            'mw_obs_flatten']:
                            _s = self.main_ar_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
                        else:
                            _s = input_dict["state_embed"]
                        mu = mu.view(mu.shape[0], mu.shape[1] * mu.shape[2])
                        Q = self.main_ar_critic(torch.cat([_s, mu], dim=-1))
                    else:
                        if self._args["WOLP_no_ar_critics"]:
                            # Use the ref critic to train the linked actors
                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                'mw_obs_flatten']:
                                _s = self.main_ref_critic_obs_enc(input_dict["state_embed"])
                                if self._args["env_name"].lower() == "recsim-data":
                                    _s = _s[:, None, :]
                            if self._args["WOLP_if_dual_critic"]:
                                # This smooths the effect of unknown discreteness in discrete action space tasks.
                                Q = self.main_extra_critic(torch.cat([_s.repeat(1, mu.shape[1], 1), mu], dim=-1))
                            else:
                                Q = self.main_ref_critic(torch.cat([_s.repeat(1, mu.shape[1], 1), mu], dim=-1))
                            if self._args["WOLP_threshold_Q"]:
                                if self._args["WOLP_threshold_Q_direct_cummax"]:
                                    assert False, "Likely incorrect!"
                                    Q = torch.cummax(Q, dim=1)[0]
                                else:
                                    Q_new = Q[:, :1]
                                    for i in range(1, Q.shape[1]):
                                        Q_no_grad = Q[:, i-1].detach()
                                        Q_new = torch.cat([Q_new, torch.max(Q[:, i], Q_no_grad)[:, None, :]], dim=1)
                                    Q = Q_new
                        else:
                            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args[
                                'mw_obs_flatten']:
                                _s = self.main_ar_critic_obs_enc(input_dict["state_embed"])
                                if self._args["env_name"].lower() == "recsim-data":
                                    _s = _s[:, None, :]
                            if self._args["REDQ"] and self._args["WOLP_cascade_list_len"] == 1:
                                Q_vals = []
                                for i in range(self._args["REDQ_num"]):
                                    Q_vals.append(self.main_ref_critic_redq_list[i](
                                        torch.cat([_s.repeat(1, mu.shape[1], 1), mu], dim=-1)))
                                Q = torch.stack(Q_vals, dim=1).mean(dim=1)
                            else:
                                Q = self.main_ar_critic(state=_s, action_seq=mu,
                                    knn_function=knn_action_function if \
                                    (self._args["WOLP_if_ar"] and self._args["WOLP_knn_inside_cascade"] and self._args["WOLP_if_ar_noise_before_cascade"] \
                                        and not self._args["env_name"].lower().startswith("mujoco")) \
                                    else None,
                                    alternate_conditioning=alternate_conditioning
                                    )

                    if self._args["WOLP_if_ar_contextual_prop"]:  # Use the last entry to rely on backprop thru time
                        Q = Q[:, -1]  # batch x 1
                elif self._args["WOLP_if_joint_actor"]:

                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_actor_obs_enc(input_dict["state_embed"])
                    else:
                        _s = input_dict["state_embed"][:, None, :]
                    mu = self.main_actor(_s, if_joint_update=True)  # batch x list-len * dim-action

                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_joint_critic_obs_enc(input_dict["state_embed"])
                    Q = self.main_joint_critic(torch.cat([_s, mu], dim=-1))
                elif self._args["WOLP_if_dual_critic"]:
                    # WOLP_dual, where actor loss comes from extra critic
                    # extra_critic is trained with diverse action embs as inputs, while ref_critic only GT action embs
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_actor_obs_enc(input_dict["state_embed"])
                    else:
                        _s = input_dict["state_embed"][:, None, :]
                    mu = self.main_actor(_s).squeeze(1)  # batch x list-len * dim-action
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_extra_critic_obs_enc(input_dict["state_embed"])
                    Q = self.main_extra_critic(torch.cat([_s, mu], dim=-1))
                else:
                    # WOLP original, where actor loss comes from ref critic
                    assert self._args["method_name"].lower() == "wolp"
                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_actor_obs_enc(input_dict["state_embed"]).squeeze(1)
                    else:
                        _s = input_dict["state_embed"]
                    mu = self.main_actor(_s)

                    if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                        _s = self.main_ref_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
                    Q = self.main_ref_critic(torch.cat([_s[:, None, :].repeat(1, mu.shape[1], 1), mu], dim=-1))

                # Average over samples & preserve contribution of slate indices, equal to total mean for non-listwise RL
                if self._args["WOLP_mean_policy_loss"]:
                    policy_loss = -Q.mean()
                else:
                    policy_loss = -Q.sum(dim=-1).mean()

                if self._args["WOLP_actor_penalty"]:
                    base_policy_loss = policy_loss
                    penalty_loss = compute_penalty(
                        mu,
                        self._args["WOLP_actor_penalty_sigma"],
                        self._args["WOLP_actor_penalty_weight"]
                    )
                    policy_loss += penalty_loss

                if self.opt_actor_obs_enc is not None: self.opt_actor_obs_enc.zero_grad()
                self.opt_actor.zero_grad()
                policy_loss.backward()
                # print(check_grad(self.main_actor.named_parameters()))
                if self._args["if_grad_clip"]: clip_grad(self.main_actor.parameters(), 1.0)  # gradient clipping
                if self.opt_actor_obs_enc is not None and self._args["if_grad_clip"]:
                    clip_grad(self.main_actor_obs_enc.parameters(), 1.0)
                self.opt_actor.step()
                if self.opt_actor_obs_enc is not None: self.opt_actor_obs_enc.step()

        if kwargs["if_sel"]:
            self.res = {"value_loss": value_loss.item(), "loss": value_loss.item()}
            if self._args["TwinQ"] and not self._args["REDQ"]:
                self.res["value_loss_twin"] = value_loss_twin.item()
                self.res["loss_twin"] = value_loss_twin.item()

        if kwargs["if_ret"]:
            self.res["policy_loss"] = policy_loss.item() if train_actor else 0.0
            self.res["loss"] = self.res["value_loss"] + policy_loss.item() if train_actor else self.res["value_loss"]
            if (self._args["WOLP_if_ar"] or (self._args["WOLP_if_ar"] and self._args["WOLP_if_joint_critic"])) and not \
                self._args["WOLP_no_ar_critics"]:
                self.res["ar_value_loss"] = ar_value_loss.item()
                if self._args["TwinQ"] and not self._args['WOLP_if_ar_imitate']:
                    self.res["ar_value_loss_twin"] = ar_value_loss_twin.item()

            if self._args["WOLP_if_joint_actor"]:
                self.res["joint_value_loss"] = joint_value_loss.item()
                if self._args["TwinQ"]:
                    self.res["joint_value_loss_twin"] = joint_value_loss_twin.item()

            if self._args["WOLP_if_dual_critic"]:
                self.res["extra_value_loss"] = extra_value_loss
                if self._args["TwinQ"]:
                    self.res["extra_value_loss_twin"] = extra_value_loss_twin

            if self._args["WOLP_actor_penalty"]:
                self.res["actor_penalty_loss"] = penalty_loss.item() if train_actor else 0.0
                self.res["policy_loss"] = base_policy_loss.item() if train_actor else 0.0
                self.res["total_actor_loss"] = policy_loss.item() if train_actor else 0.0

    def _sync(self, tau: float = 0.0):
        """ when _type is None, we update both actor and critic """
        if tau > 0.0:  # Soft update of params
            # ========== Actor ==========
            if self.main_actor is not None:
                for param, target_param in zip(self.main_actor.parameters(),
                                               self.target_actor.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                if self.main_actor_obs_enc is not None:
                    for param, target_param in zip(self.main_actor_obs_enc.parameters(),
                                                   self.target_actor_obs_enc.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            # ========== Selection Critic ==========
            for param, target_param in zip(self.main_ref_critic.parameters(), self.target_ref_critic.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["REDQ"]:
                for i in range(self._args["REDQ_num"]):
                    for param, target_param in zip(self.main_ref_critic_redq_list[i].parameters(),
                                                   self.target_ref_critic_redq_list[i].parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["TwinQ"]:
                for param, target_param in zip(self.main_ref_critic_twin.parameters(),
                                               self.target_ref_critic_twin.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self.main_ref_critic_obs_enc is not None:
                for param, target_param in zip(self.main_ref_critic_obs_enc.parameters(),
                                               self.target_ref_critic_obs_enc.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                if self._args["TwinQ"]:
                    for param, target_param in zip(self.main_ref_critic_obs_enc_twin.parameters(),
                                                   self.target_ref_critic_obs_enc_twin.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            # ========== Extra Critic ==========
            if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
                for param, target_param in zip(self.main_ar_critic.parameters(), self.target_ar_critic.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                    for param, target_param in zip(self.main_ar_critic_twin.parameters(),
                                                   self.target_ar_critic_twin.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                if self.main_ar_critic_obs_enc is not None:
                    for param, target_param in zip(self.main_ar_critic_obs_enc.parameters(),
                                                   self.target_ar_critic_obs_enc.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                    if self._args["TwinQ"]:
                        for param, target_param in zip(self.main_ar_critic_obs_enc_twin.parameters(),
                                                       self.target_ar_critic_obs_enc_twin.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
                for param, target_param in zip(self.main_joint_critic.parameters(),
                                               self.target_joint_critic.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                if self._args["TwinQ"]:
                    for param, target_param in zip(self.main_joint_critic_twin.parameters(),
                                                   self.target_joint_critic_twin.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                if self.main_joint_critic_obs_enc is not None:
                    for param, target_param in zip(self.main_joint_critic_obs_enc.parameters(),
                                                   self.target_joint_critic_obs_enc.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                    if self._args["TwinQ"]:
                        for param, target_param in zip(self.main_joint_critic_obs_enc_twin.parameters(),
                                                       self.target_joint_critic_obs_enc_twin.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["WOLP_if_dual_critic"]:
                for param, target_param in zip(self.main_extra_critic.parameters(),
                                               self.target_extra_critic.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                if self._args["TwinQ"]:
                    for param, target_param in zip(self.main_extra_critic_twin.parameters(),
                                                   self.target_extra_critic_twin.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                if self.main_extra_critic_obs_enc is not None:
                    for param, target_param in zip(self.main_extra_critic_obs_enc.parameters(),
                                                   self.target_extra_critic_obs_enc.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                    if self._args["TwinQ"]:
                        for param, target_param in zip(self.main_extra_critic_obs_enc_twin.parameters(),
                                                       self.target_extra_critic_obs_enc_twin.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
        else:
            if self.main_actor is not None:
                self.target_ref_critic.load_state_dict(self.main_ref_critic.state_dict())
                if self._args["TwinQ"]:
                    self.target_ref_critic_twin.load_state_dict(self.main_ref_critic_twin.state_dict())
                self.target_actor.load_state_dict(self.main_actor.state_dict())

            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.target_actor_obs_enc.load_state_dict(self.main_actor_obs_enc.state_dict())
                self.target_ref_critic_obs_enc.load_state_dict(self.main_ref_critic_obs_enc.state_dict())
                if self._args["TwinQ"]:
                    self.target_ref_critic_obs_enc_twin.load_state_dict(self.main_ref_critic_obs_enc_twin.state_dict())

            if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
                self.target_ar_critic.load_state_dict(self.main_ar_critic.state_dict())
                if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                    self.target_ar_critic_twin.load_state_dict(self.main_ar_critic_twin.state_dict())

    def _train(self):
        if self.main_actor is not None:
            self.main_actor.train()
            self.target_actor.train()

        self.main_ref_critic.train()
        self.target_ref_critic.train()
        if self._args["REDQ"]:
            for i in range(self._args["REDQ_num"]):
                self.main_ref_critic_redq_list[i].train()
                self.target_ref_critic_redq_list[i].train()
        if self._args["TwinQ"]:
            self.main_ref_critic_twin.train()
            self.target_ref_critic_twin.train()

        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            if self.opt_actor_obs_enc is not None:
                self.main_actor_obs_enc.train()
                self.target_actor_obs_enc.train()
                self.main_ref_critic_obs_enc.train()
                self.target_ref_critic_obs_enc.train()
                if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
                    if self.main_ar_critic_obs_enc is not None:
                        self.main_ar_critic_obs_enc.train()
                        self.target_ar_critic_obs_enc.train()
                if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
                    if self.main_joint_critic_obs_enc is not None:
                        self.main_joint_critic_obs_enc.train()
                        self.target_joint_critic_obs_enc.train()
                if self._args["WOLP_if_dual_critic"]:
                    if self.main_extra_critic_obs_enc is not None:
                        self.main_extra_critic_obs_enc.train()
                        self.target_extra_critic_obs_enc.train()
                # TwinQ observation encoders
                if self._args["TwinQ"]:
                    self.main_ref_critic_obs_enc_twin.train()
                    self.target_ref_critic_obs_enc_twin.train()
                    if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
                        if self.main_ar_critic_obs_enc_twin is not None:
                            self.main_ar_critic_obs_enc_twin.train()
                            self.target_ar_critic_obs_enc_twin.train()
                    if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
                        if self.main_joint_critic_obs_enc_twin is not None:
                            self.main_joint_critic_obs_enc_twin.train()
                            self.target_joint_critic_obs_enc_twin.train()
                    if self._args["WOLP_if_dual_critic"]:
                        if self.main_extra_critic_obs_enc_twin is not None:
                            self.main_extra_critic_obs_enc_twin.train()
                            self.target_extra_critic_obs_enc_twin.train()

        if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
            self.main_ar_critic.train()
            self.target_ar_critic.train()
            if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                self.main_ar_critic_twin.train()
                self.target_ar_critic_twin.train()

        if self._args["WOLP_if_dual_critic"]:
            self.main_extra_critic.train()
            self.target_extra_critic.train()
            if self._args["TwinQ"]:
                self.main_extra_critic_twin.train()
                self.target_extra_critic_twin.train()

        if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
            self.main_joint_critic.train()
            self.target_joint_critic.train()
            if self._args["TwinQ"]:
                self.main_joint_critic_twin.train()
                self.target_joint_critic_twin.train()

    def _eval(self):
        if self.main_actor is not None:
            self.main_actor.eval()
            self.target_actor.eval()

        self.main_ref_critic.eval()
        self.target_ref_critic.eval()
        if self._args["REDQ"]:
            for i in range(self._args["REDQ_num"]):
                self.main_ref_critic_redq_list[i].eval()
                self.target_ref_critic_redq_list[i].eval()
        if self._args["TwinQ"]:
            self.main_ref_critic_twin.eval()
            self.target_ref_critic_twin.eval()

        if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            if self.opt_actor_obs_enc is not None:
                self.main_actor_obs_enc.eval()
                self.target_actor_obs_enc.eval()
                self.main_ref_critic_obs_enc.eval()
                self.target_ref_critic_obs_enc.eval()
                if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
                    if self.main_ar_critic_obs_enc is not None:
                        self.main_ar_critic_obs_enc.eval()
                        self.target_ar_critic_obs_enc.eval()
                if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
                    if self.main_joint_critic_obs_enc is not None:
                        self.main_joint_critic_obs_enc.eval()
                        self.target_joint_critic_obs_enc.eval()
                if self._args["WOLP_if_dual_critic"]:
                    if self.main_extra_critic_obs_enc is not None:
                        self.main_extra_critic_obs_enc.eval()
                        self.target_extra_critic_obs_enc.eval()
                # TwinQ observation encoders
                if self._args["TwinQ"]:
                    self.main_ref_critic_obs_enc_twin.eval()
                    self.target_ref_critic_obs_enc_twin.eval()
                    if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
                        if self.main_ar_critic_obs_enc_twin is not None:
                            self.main_ar_critic_obs_enc_twin.eval()
                            self.target_ar_critic_obs_enc_twin.eval()
                    if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
                        if self.main_joint_critic_obs_enc_twin is not None:
                            self.main_joint_critic_obs_enc_twin.eval()
                            self.target_joint_critic_obs_enc_twin.eval()
                    if self._args["WOLP_if_dual_critic"]:
                        if self.main_extra_critic_obs_enc_twin is not None:
                            self.main_extra_critic_obs_enc_twin.eval()
                            self.target_extra_critic_obs_enc_twin.eval()

        if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
            self.main_ar_critic.eval()
            self.target_ar_critic.eval()
            if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                self.main_ar_critic_twin.eval()
                self.target_ar_critic_twin.eval()

        if self._args["WOLP_if_dual_critic"]:
            self.main_extra_critic.eval()
            self.target_extra_critic.eval()
            if self._args["TwinQ"]:
                self.main_extra_critic_twin.eval()
                self.target_extra_critic_twin.eval()

        if self._args["WOLP_if_joint_critic"] and not self._args["WOLP_if_ar"]:
            self.main_joint_critic.eval()
            self.target_joint_critic.eval()
            if self._args["TwinQ"]:
                self.main_joint_critic_twin.eval()
                self.target_joint_critic_twin.eval()

    def _reset(self, **kwargs):
        if "id" in kwargs:
            self.noise_sampler.reset(kwargs["id"])
        else:
            self.noise_sampler.reset()

    def _save(self, save_dir):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        logging("Save the agent: {}".format(save_dir))
        ## Actor
        if self._args['env_name'] in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            torch.save(self.main_actor_obs_enc.state_dict(), os.path.join(save_dir, f"main_actor_obs_enc.pkl"))
            torch.save(self.target_actor_obs_enc.state_dict(), os.path.join(save_dir, f"target_actor_obs_enc.pkl"))
            torch.save(self.opt_actor_obs_enc.state_dict(), os.path.join(save_dir, f"opt_actor_obs_enc.pkl"))
        torch.save(self.main_actor.state_dict(), os.path.join(save_dir, f"main_actor.pkl"))
        torch.save(self.target_actor.state_dict(), os.path.join(save_dir, f"target_actor.pkl"))
        torch.save(self.opt_actor.state_dict(), os.path.join(save_dir, f"opt_actor.pkl"))

        ## Critic
        if self._args['env_name'] in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            torch.save(self.main_ref_critic_obs_enc.state_dict(),
                       os.path.join(save_dir, f"main_ref_critic_obs_enc.pkl"))
            torch.save(self.target_ref_critic_obs_enc.state_dict(),
                       os.path.join(save_dir, f"target_ref_critic_obs_enc.pkl"))
            torch.save(self.opt_ref_critic_obs_enc.state_dict(), os.path.join(save_dir, f"opt_ref_critic_obs_enc.pkl"))
        torch.save(self.main_ref_critic.state_dict(), os.path.join(save_dir, f"main_ref_critic.pkl"))
        torch.save(self.target_ref_critic.state_dict(), os.path.join(save_dir, f"target_ref_critic.pkl"))
        torch.save(self.opt_ref_critic.state_dict(), os.path.join(save_dir, f"opt_ref_critic.pkl"))

        if self._args["REDQ"]:
            for i in range(self._args["REDQ_num"]):
                torch.save(self.main_ref_critic_redq_list[i].state_dict(),
                           os.path.join(save_dir, f"main_ref_critic_redq_{i}.pkl"))
                torch.save(self.target_ref_critic_redq_list[i].state_dict(),
                           os.path.join(save_dir, f"target_ref_critic_redq_{i}.pkl"))
                torch.save(self.opt_ref_critic_redq_list[i].state_dict(),
                           os.path.join(save_dir, f"opt_ref_critic_redq_{i}.pkl"))

        if self._args["TwinQ"]:
            if self._args['env_name'] in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                torch.save(self.main_ref_critic_obs_enc_twin.state_dict(),
                           os.path.join(save_dir, f"main_ref_critic_obs_enc_twin.pkl"))
                torch.save(self.target_ref_critic_obs_enc_twin.state_dict(),
                           os.path.join(save_dir, f"target_ref_critic_obs_enc_twin.pkl"))
                torch.save(self.opt_ref_critic_obs_enc_twin.state_dict(),
                           os.path.join(save_dir, f"opt_ref_critic_obs_enc_twin.pkl"))
            torch.save(self.main_ref_critic_twin.state_dict(), os.path.join(save_dir, f"main_ref_critic_twin.pkl"))
            torch.save(self.target_ref_critic_twin.state_dict(), os.path.join(save_dir, f"target_ref_critic_twin.pkl"))
            torch.save(self.opt_ref_critic_twin.state_dict(), os.path.join(save_dir, f"opt_ref_critic_twin.pkl"))

        if self._args["WOLP_if_joint_critic"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                torch.save(self.main_joint_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"main_joint_critic_obs_enc.pkl"))
                torch.save(self.target_joint_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"target_joint_critic_obs_enc.pkl"))
                torch.save(self.opt_joint_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"opt_joint_critic_obs_enc.pkl"))

            torch.save(self.main_joint_critic.state_dict(), os.path.join(save_dir, f"main_joint_critic.pkl"))
            torch.save(self.target_joint_critic.state_dict(), os.path.join(save_dir, f"target_joint_critic.pkl"))
            torch.save(self.opt_joint_critic.state_dict(), os.path.join(save_dir, f"opt_joint_critic.pkl"))
            if self._args["TwinQ"]:
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    torch.save(self.main_joint_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"main_joint_critic_obs_enc_twin.pkl"))
                    torch.save(self.target_joint_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"target_joint_critic_obs_enc_twin.pkl"))
                    torch.save(self.opt_joint_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"opt_joint_critic_obs_enc_twin.pkl"))
                torch.save(self.main_joint_critic_twin.state_dict(),
                           os.path.join(save_dir, f"main_joint_critic_twin.pkl"))
                torch.save(self.target_joint_critic_twin.state_dict(),
                           os.path.join(save_dir, f"target_joint_critic_twin.pkl"))
                torch.save(self.opt_joint_critic_twin.state_dict(),
                           os.path.join(save_dir, f"opt_joint_critic_twin.pkl"))

        if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                torch.save(self.main_ar_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"main_ar_critic_obs_enc.pkl"))
                torch.save(self.target_ar_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"target_ar_critic_obs_enc.pkl"))
                torch.save(self.opt_ar_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"opt_ar_critic_obs_enc.pkl"))
            torch.save(self.main_ar_critic.state_dict(), os.path.join(save_dir, f"main_ar_critic.pkl"))
            torch.save(self.target_ar_critic.state_dict(), os.path.join(save_dir, f"target_ar_critic.pkl"))
            torch.save(self.opt_ar_critic.state_dict(), os.path.join(save_dir, f"opt_ar_critic.pkl"))
            if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    torch.save(self.main_ar_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"main_ar_critic_obs_enc_twin.pkl"))
                    torch.save(self.target_ar_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"target_ar_critic_obs_enc_twin.pkl"))
                    torch.save(self.opt_ar_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"opt_ar_critic_obs_enc_twin.pkl"))
                torch.save(self.main_ar_critic_twin.state_dict(), os.path.join(save_dir, f"main_ar_critic_twin.pkl"))
                torch.save(self.target_ar_critic_twin.state_dict(),
                           os.path.join(save_dir, f"target_ar_critic_twin.pkl"))
                torch.save(self.opt_ar_critic_twin.state_dict(), os.path.join(save_dir, f"opt_ar_critic_twin.pkl"))

        if self._args["WOLP_if_dual_critic"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                torch.save(self.main_extra_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"main_extra_critic_obs_enc.pkl"))
                torch.save(self.target_extra_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"target_extra_critic_obs_enc.pkl"))
                torch.save(self.opt_extra_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"opt_extra_critic_obs_enc.pkl"))
            torch.save(self.main_extra_critic.state_dict(), os.path.join(save_dir, f"main_extra_critic.pkl"))
            torch.save(self.target_extra_critic.state_dict(), os.path.join(save_dir, f"target_extra_critic.pkl"))
            torch.save(self.opt_extra_critic.state_dict(), os.path.join(save_dir, f"opt_extra_critic.pkl"))
            if self._args["TwinQ"]:
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    torch.save(self.main_extra_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"main_extra_critic_obs_enc_twin.pkl"))
                    torch.save(self.target_extra_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"target_extra_critic_obs_enc_twin.pkl"))
                    torch.save(self.opt_extra_critic_obs_enc_twin.state_dict(),
                               os.path.join(save_dir, f"opt_extra_critic_obs_enc_twin.pkl"))
                torch.save(self.main_extra_critic_twin.state_dict(),
                           os.path.join(save_dir, f"main_extra_critic_twin.pkl"))
                torch.save(self.target_extra_critic_twin.state_dict(),
                           os.path.join(save_dir, f"target_extra_critic_twin.pkl"))
                torch.save(self.opt_extra_critic_twin.state_dict(),
                           os.path.join(save_dir, f"opt_extra_critic_twin.pkl"))

    def _load(self, load_dir):
        logging("Load the agent: {}".format(load_dir))
        ## Actor
        if self._args['env_name'] in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self.main_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"main_actor_obs_enc.pkl")))
            self.target_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"target_actor_obs_enc.pkl")))
            self.opt_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"opt_actor_obs_enc.pkl")))
        self.main_actor.load_state_dict(torch.load(os.path.join(load_dir, f"main_actor.pkl")))
        self.target_actor.load_state_dict(torch.load(os.path.join(load_dir, f"target_actor.pkl")))
        self.opt_actor.load_state_dict(torch.load(os.path.join(load_dir, f"opt_actor.pkl")))

        ## Critic
        if self._args['env_name'] in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
            self.main_ref_critic_obs_enc.load_state_dict(
                torch.load(os.path.join(load_dir, f"main_ref_critic_obs_enc.pkl")))
            self.target_ref_critic_obs_enc.load_state_dict(
                torch.load(os.path.join(load_dir, f"target_ref_critic_obs_enc.pkl")))
            self.opt_ref_critic_obs_enc.load_state_dict(
                torch.load(os.path.join(load_dir, f"opt_ref_critic_obs_enc.pkl")))

        self.main_ref_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_ref_critic.pkl")))
        self.target_ref_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_ref_critic.pkl")))
        self.opt_ref_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_ref_critic.pkl")))

        if self._args["REDQ"]:
            for i in range(self._args["REDQ_num"]):
                self.main_ref_critic_redq_list[i].load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_ref_critic_redq_{i}.pkl")))
                self.target_ref_critic_redq_list[i].load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_ref_critic_redq_{i}.pkl")))
                self.opt_ref_critic_redq_list[i].load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_ref_critic_redq_{i}.pkl")))

        if self._args["TwinQ"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.main_ref_critic_obs_enc_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_ref_critic_obs_enc_twin.pkl")))
                self.target_ref_critic_obs_enc_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_ref_critic_obs_enc_twin.pkl")))
                self.opt_ref_critic_obs_enc_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_ref_critic_obs_enc_twin.pkl")))
            self.main_ref_critic_twin.load_state_dict(torch.load(os.path.join(load_dir, f"main_ref_critic_twin.pkl")))
            self.target_ref_critic_twin.load_state_dict(
                torch.load(os.path.join(load_dir, f"target_ref_critic_twin.pkl")))
            self.opt_ref_critic_twin.load_state_dict(torch.load(os.path.join(load_dir, f"opt_ref_critic_twin.pkl")))

        if self._args["WOLP_if_joint_critic"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.main_extra_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_joint_critic_obs_enc.pkl")))
                self.target_joint_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_joint_critic_obs_enc.pkl")))
                self.opt_joint_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_joint_critic_obs_enc.pkl")))
            self.main_joint_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_joint_critic.pkl")))
            self.target_joint_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_joint_critic.pkl")))
            self.opt_joint_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_joint_critic.pkl")))
            if self._args["TwinQ"]:
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    self.main_joint_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"main_joint_critic_obs_enc_twin.pkl")))
                    self.target_joint_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"target_joint_critic_obs_enc_twin.pkl")))
                    self.opt_joint_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"opt_joint_critic_obs_enc_twin.pkl")))
                self.main_joint_critic_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_joint_critic_twin.pkl")))
                self.target_joint_critic_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_joint_critic_twin.pkl")))
                self.opt_joint_critic_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_joint_critic_twin.pkl")))

        if self._args["WOLP_if_ar"] and not self._args["WOLP_no_ar_critics"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.main_ar_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_ar_critic_obs_enc.pkl")))
                self.target_ar_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_ar_critic_obs_enc.pkl")))
                self.opt_ar_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_ar_critic_obs_enc.pkl")))
            self.main_ar_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_ar_critic.pkl")))
            self.target_ar_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_ar_critic.pkl")))
            self.opt_ar_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_ar_critic.pkl")))
            if self._args["TwinQ"] and not self._args["WOLP_if_ar_imitate"]:
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    self.main_ar_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"main_ar_critic_obs_enc_twin.pkl")))
                    self.target_ar_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"target_ar_critic_obs_enc_twin.pkl")))
                    self.opt_ar_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"opt_ar_critic_obs_enc_twin.pkl")))
                self.main_ar_critic_twin.load_state_dict(torch.load(os.path.join(load_dir, f"main_ar_critic_twin.pkl")))
                self.target_ar_critic_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_ar_critic_twin.pkl")))
                self.opt_ar_critic_twin.load_state_dict(torch.load(os.path.join(load_dir, f"opt_ar_critic_twin.pkl")))

        if self._args["WOLP_if_dual_critic"]:
            if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                self.main_extra_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_extra_critic_obs_enc.pkl")))
                self.target_extra_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_extra_critic_obs_enc.pkl")))
                self.opt_extra_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_extra_critic_obs_enc.pkl")))
            self.main_extra_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_extra_critic.pkl")))
            self.target_extra_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_extra_critic.pkl")))
            self.opt_extra_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_extra_critic.pkl")))
            if self._args["TwinQ"]:
                if self._args["env_name"].lower() in ["mine", "recsim-data"] and not self._args['mw_obs_flatten']:
                    self.main_extra_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"main_extra_critic_obs_enc_twin.pkl")))
                    self.target_extra_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"target_extra_critic_obs_enc_twin.pkl")))
                    self.opt_extra_critic_obs_enc_twin.load_state_dict(
                        torch.load(os.path.join(load_dir, f"opt_extra_critic_obs_enc_twin.pkl")))
                self.main_extra_critic_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_extra_critic_twin.pkl")))
                self.target_extra_critic_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_extra_critic_twin.pkl")))
                self.opt_extra_critic_twin.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_extra_critic_twin.pkl")))
