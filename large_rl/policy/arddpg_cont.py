""" Cascaded Critics operates on the discrete candidates """

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_ as clip_grad
from copy import deepcopy
import os.path as osp
import os

from large_rl.embedding.base import BaseEmbedding
from large_rl.policy.arch.mlp import MLP
from large_rl.policy.arch.ddpg import Actor, DDPG_OUNoise as OUNoise
from large_rl.policy.arch.arddpg import ARActor, ARCritic
from large_rl.policy.arch.cem import CEM
from large_rl.policy.agent import Agent
from large_rl.commons.utils import logging
from large_rl.policy.wolp import WOLP

LARGE_NEG = -1000000


class ARDDPG_CONT(WOLP, Agent):
    def __init__(self, **kwargs):
        super(WOLP, self).__init__(**kwargs)

        self._query = self._topk_act = None  # this is for CDDPG

        # Prep the input dim
        def _get_dim_cascade_input():
            if self._args["env_name"].startswith("mujoco"):
                return self._args["reacher_obs_space"]
            else:
                raise ValueError

        dim_hidden = self._args.get('dim_hidden', 64)

        # === RETRIEVAL ===
        if self._args['env_name'].startswith("mujoco"):
            self._actor_dim_out = self._args["reacher_action_shape"]
            self._critic_dim_state = self._args["reacher_obs_space"]
            self._critic_dim_action = self._args["reacher_action_shape"]
        else:
            raise NotImplementedError

        # 1. Observation Encoder
        if self._args["env_name"].startswith("mujoco"):
            self.main_actor_obs_enc = self.target_actor_obs_enc = self.opt_actor_obs_enc = None
        else:
            self._define_obs_encoders(actor=True, critic=False)
            raise NotImplementedError('Check if you need observation encoder for this env')

        # 2. Joint Actor-Critic for Retrieval
        if self._args["WOLP_if_joint_actor"]:
            if self._args["env_name"].startswith("mujoco"):
                self.main_joint_critic_obs_enc = self.target_joint_critic_obs_enc = self.opt_joint_critic_obs_enc = None
                self._actor_dim_out = (self._args["reacher_action_shape"] *
                                       self._args['WOLP_cascade_list_len'])
            else:
                self._define_joint_critic_obs_encoders()
                self._actor_dim_out = None
                raise NotImplementedError

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
            logging(self.main_joint_critic)
        else:
            self.main_joint_critic = self.target_joint_critic = self.opt_joint_critic = None

        self.main_actor, self.target_actor, self.opt_actor = None, None, None
        self.main_ar_critic = self.target_ar_critic = self.opt_ar_critic = None
        if self._args["WOLP_if_cem_actor"]:
            # 3. CEM Actor
            self.cem = CEM(seed=self._args["seed"], dim_action=self._actor_dim_out, topK=self._args["CEM_topK"])
        elif self._args["WOLP_if_ar"]:
            # 4. ARDDGP and its variants
            assert not self._args['WOLP_if_joint_actor']
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
            if self._args["env_name"].startswith("mujoco"):
                self.main_ar_critic_obs_enc = self.target_ar_critic_obs_enc = self.opt_ar_critic_obs_enc = None
            else:
                self._define_ar_critic_obs_encoders()

            self.main_ar_critic = ARCritic(args=self._args,
                                           dim_state=_get_dim_cascade_input(),
                                           dim_hidden=dim_hidden, dim_memory=self._args["WOLP_slate_dim_out"],
                                           dim_action=self._critic_dim_action,
                                           ).to(device=self._device)
            self.target_ar_critic = deepcopy(self.main_ar_critic)
            self.opt_ar_critic = torch.optim.Adam(params=self.main_ar_critic.parameters(),
                                                  lr=self._args["WOLP_critic_lr"])
            logging(self.main_ar_critic)
        else:
            # 5. Base Wolpertinger
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

            # === Another Critic for Retrieval for Base Wolpertinger ===
            if self._args["WOLP_if_dual_critic"]:
                if self._args["env_name"].startswith("mujoco"):
                    self.main_extra_critic_obs_enc = self.target_extra_critic_obs_enc = self.opt_extra_critic_obs_enc = None
                else:
                    self.main_ar_critic_obs_enc = self.target_ar_critic_obs_enc = self.opt_ar_critic_obs_enc = None

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
                logging(self.main_extra_critic)
            else:
                self.main_extra_critic = self.target_extra_critic = self.opt_extra_critic = None
        logging(self.main_actor)

        # === REFINEMENT ===
        # and then inherit those initialization functions. Only keep the environment specific code to be separate
        if self._args["env_name"].startswith("mujoco"):
            self.main_ref_critic_obs_enc = self.target_ref_critic_obs_enc = self.opt_ref_critic_obs_enc = None
        else:
            self._define_obs_encoders(actor=False, critic=True)
            raise NotImplementedError

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
        logging(self.main_ref_critic)

        # === OU Noise ===
        dim_action = self._actor_dim_out
        if self._args["WOLP_if_ar"]:
            self.dim_action = (self._args["num_envs"], self._args["WOLP_cascade_list_len"], dim_action)
        elif self._args["WOLP_if_joint_actor"]:
            self.dim_action = (self._args["num_envs"], self._args["WOLP_cascade_list_len"],
                               dim_action // self._args["WOLP_cascade_list_len"])
        else:
            self.dim_action = (self._args["num_envs"], 1, dim_action)
        self.noise_sampler = OUNoise(dim_action=self.dim_action, device=self._args["device"])

    def _cem_get_query(self, state, q_net, **kwargs):
        # CEM based policy to work out the optimised query
        self.cem.initialise(batch_size=state.shape[0])
        state = state[:, None, :].repeat(1, self._args["CEM_num_samples"], 1)

        for i in range(self._args["CEM_num_iter"]):
            actions = self.cem.sample(self._args["CEM_num_samples"])  # b x num_samples x dim_action

            with torch.no_grad():
                # b x num_samples
                Q_vals = q_net(torch.cat([state, torch.tensor(actions, device=self._device)], -1)).squeeze(-1)
            if i < (self._args["CEM_num_iter"] - 1):
                idx = torch.topk(Q_vals, k=self._args["CEM_topK"]).indices.cpu().detach().numpy()  # b x topK
                elites = np.take_along_axis(arr=actions, indices=idx[..., None], axis=1)  # b x topK x dim
                self.cem.update(elite_samples=elites)

        max_idx = torch.topk(Q_vals, k=self._args["WOLP_cascade_list_len"]).indices.cpu().detach().numpy()
        mu = np.take_along_axis(arr=actions, indices=max_idx[..., None], axis=1)  # b x topK x dim_action
        # No randomness here is okay because we are affording the Q-function a chance to select any of the topK actions.
        # mu = self._add_noise(mu=torch.tensor(mu, device=self._device), **kwargs)
        return torch.tensor(mu, device=self._device)

    def _add_noise(self, mu, **kwargs):
        if not kwargs.get("if_update", False):
            eps = self.noise_sampler.noise(scale=kwargs["epsilon"]["actor"])
            mu += eps
            if self._args["DEBUG_type_clamp"] == "small":
                mu = mu.clamp(0, 1)  # Constraint the range
            elif self._args["DEBUG_type_clamp"] == "large":
                mu = mu.clamp(-1, 1)  # Constraint the range
        return mu

    def _continuous_kNN(self, mu, _top_k):
        """
        In the continuous action space of the environment, find neighbors in a close neighborhood of the original
        continuous action.
        """
        topk_act = mu.repeat([1, _top_k, 1])
        topk_act += self._args['continuous_kNN_sigma'] * torch.randn(topk_act.shape, device=self._device)
        return topk_act

    def select_random_action(self, batch_size: int, slate_size: int = None, candidate_lists=None):
        assert self._args["env_name"].startswith("mujoco")
        return np.array([self._args["reacher_action_space"].sample() for _ in range(batch_size)])

    def select_action(self, obs: torch.tensor, act_embed_base: BaseEmbedding, epsilon: dict, **kwargs):
        """
            This function is largely different from the wolp.py because of the continuous action v/s discrete action difference
        """
        policy_obs = torch.tensor(obs, dtype=torch.float32, device=self._device)

        # Prep input for agent
        input_dict = {"state_embed": policy_obs}

        # Main inference
        action, query, refinement_index = self._select_action(input_dict=input_dict, epsilon=epsilon, **kwargs)

        # Construct the Agent response
        res = {"action": np.asarray(action)}

        if self._args["agent_type"].startswith('arddpg_cont'):
            res["query"] = query  # Query from Actor in Wolp / LIRD
            res["refinement_index"] = refinement_index
        return res

    def _select_action(self, input_dict: dict, **kwargs):
        # === 1. When called from update method, we can switch to Actor Target === #
        actor = self.target_actor if kwargs.get("if_update", False) else self.main_actor
        critic = self.target_ref_critic if kwargs.get("if_update", False) else self.main_ref_critic

        if self._args["obs_enc_apply"]:
            obs_actor_enc = self.target_actor_obs_enc if kwargs.get("if_update", False) else self.main_actor_obs_enc
            obs_critic_enc = self.target_ref_critic_obs_enc if kwargs.get("if_update",
                                                                          False) else self.main_ref_critic_obs_enc
        else:
            obs_actor_enc = None
            obs_critic_enc = None

        # === 2. CEM Actor === #
        if self._args["WOLP_if_cem_actor"]:
            mu = self._cem_get_query(state=input_dict["state_embed"], q_net=critic, **kwargs)
        else:
            # === 3. Get the proto-action from Actor #
            if self._args["obs_enc_apply"]:
                _s = obs_actor_enc(input_dict["state_embed"])
            else:
                _s = input_dict["state_embed"]
            if self._args["WOLP_if_ar"]:
                _s = _s[:, None, :]
            if self._args["WOLP_if_ar"] and self._args["WOLP_if_ar_noise_before_cascade"]:
                if not kwargs.get("if_update", False):
                    eps = self.noise_sampler.noise(scale=kwargs["epsilon"]["actor"])
                else:
                    eps = None
                mu = actor(_s, if_next_Q=False, eps=eps)
            else:
                mu = actor(_s)
                if not kwargs.get('if_update', False):
                    mu = self._add_noise(mu=mu, **kwargs)

        # Store the query from Actor
        _query = None if kwargs.get("if_update", False) else mu.cpu().detach().numpy().astype(np.float32)

        # === 4. For Wolpertinger, a replacement for k-NN ~ sample a few actions randomly nearby the action === #
        if self._args['WOLP_topK'] > 1 and not self._args['WOLP_if_cem_actor']:
            mu = self._continuous_kNN(mu, self._args["WOLP_topK"])  # batch x (num_query * topK)
        else:
            mu = mu

        # === 5. Critic to Action Refinement === #
        # topk_embed = input_dict["act_embed"].gather(
        #     dim=1, index=topk_act[..., None].repeat(1, 1, input_dict["act_embed"].shape[-1]))
        if self._args["obs_enc_apply"]:
            _s = obs_critic_enc(input_dict["state_embed"])[:, None, :].repeat(1, mu.shape[1], 1)
        else:
            _s = input_dict["state_embed"][:, None, :].repeat(1, mu.shape[1], 1)

        Q_vals = critic(torch.cat([_s, mu], dim=-1))
        if (not self._if_train) and (not kwargs.get("if_update", False)):  # this didn't take that long in eval!
            self._candidate_Q_mean = Q_vals.mean().cpu().detach().numpy()
        else:
            self._candidate_Q_mean = None
        result = Q_vals.max(1)

        if kwargs.get("if_get_Q_vals", False) and kwargs.get('if_update', False):
            return result.values
        else:
            # Get the indices from the original index list
            ind = result.indices[:, :, None].repeat(1, 1, mu.shape[-1])
            a = mu.gather(dim=1, index=ind).cpu().detach().numpy().squeeze(1)
            refinement_index = result.indices.cpu().detach().numpy().astype(np.float32)

            if self._args["WOLP_if_dual_exploration"]:
                # Eps-decay for Refine-Q
                _mask = self._rng.uniform(low=0.0, high=1.0, size=mu.shape[0]) < kwargs["epsilon"]["critic"]
                a[_mask] = _query[_mask, self._rng.randint(_query.shape[1])]
                # These indices should not be used to provide the selection bonus
                refinement_index[_mask] = LARGE_NEG
            return a, _query, refinement_index

    def _prep_update_cont(self, actions, rewards):
        query = None
        selection_bonus = None
        listwise_rewards = None
        if self._args['env_name'].startswith("mujoco"):
            action_taken, query, refinement_index = actions[:, :self._critic_dim_action], \
                actions[:, self._critic_dim_action:-1], actions[:, -1]
            refinement_index = np.array(refinement_index, dtype=np.int32)

            if (self._args["WOLP_if_ar"]) or (
            self._args["WOLP_if_joint_actor"]):  # cddpg / auto-regressive actor / wolp-joint
                query = query.reshape([query.shape[0], self._args['WOLP_cascade_list_len'], -1])
                listwise_rewards = rewards.repeat(1, self._args[
                    'WOLP_cascade_list_len'])  # By default it is elementwise reward
                if self._args["WOLP_cascade_type_list_reward"].lower() == "last":
                    # Add the env reward only to the final list index
                    listwise_rewards[:, :-1] = 0.

                if self._args["WOLP_if_ar_selection_bonus"]:
                    # It is unclear what to do if refinement_index==LARGE_NEG.
                    # If we treat the exploration in refinement_Q as the env, then selection_bonus never makes sense
                    raise NotImplementedError
                    selection_bonus = torch.zeros(query.shape, device=self._device)
        else:
            raise NotImplementedError
        return action_taken, query, listwise_rewards, refinement_index, selection_bonus

    def _update(self,
                input_dict: dict,
                next_input_dict: dict,
                actions: np.ndarray,
                rewards: torch.tensor,
                reversed_dones: torch.tensor,
                act_embed_base: BaseEmbedding):

        if self._args["WOLP_if_pairwise_distance_bonus"]:
            dist_bonus = rewards[:, 1][:, None]
            rewards = rewards[:, 0][:, None]

        # === 1. REFINEMENT Q-net Update === #
        action_taken, query, listwise_rewards, refinement_index, selection_bonus = self._prep_update_cont(
            actions=actions.astype(np.float32), rewards=rewards)
        # === Get next Q-vals
        with torch.no_grad():
            next_Q = self._select_action(input_dict=next_input_dict, if_get_Q_vals=True,
                                         if_update=True, epsilon={"actor": 0.0, "critic": 0.0})
            next_refine_Q = deepcopy(next_Q)
        # ==== Get Taken Q-vals
        Q = self.main_ref_critic(torch.cat(
            [input_dict["state_embed"], torch.tensor(action_taken, device=self._device)], dim=-1))

        with torch.no_grad():
            # Note: q is Q-vals on candidate-set whereas Q is Q-val of the taken action!!
            select_Q_max = Q.max(dim=-1).values

        # === Bellman Error
        target = rewards + next_Q * reversed_dones * self._args["Qnet_gamma"]
        value_loss = torch.nn.functional.mse_loss(Q, target)

        if self.opt_ref_critic_obs_enc is not None: self.opt_ref_critic_obs_enc.zero_grad()
        self.opt_ref_critic.zero_grad()
        value_loss.backward()
        if self._args["if_grad_clip"]: clip_grad(self.main_ref_critic.parameters(), 1.0)  # gradient clipping
        if (self.opt_ref_critic_obs_enc is not None and self._args['if_grad_clip']):
            clip_grad(self.main_ref_critic_obs_enc.parameters(), 1.0)
        self.opt_ref_critic.step()
        if self.opt_ref_critic_obs_enc is not None: self.opt_ref_critic_obs_enc.step()

        # === Update Retrieval Critic
        if self._args["WOLP_if_pairwise_distance_bonus"]:
            rewards += dist_bonus

        if self._args["WOLP_if_dual_critic"]:
            assert not self._args['WOLP_if_ar'] and not self._args['WOLP_if_joint_actor']
            with torch.no_grad():
                if self._args['env_name'].startswith("mujoco"):
                    next_s = next_input_dict["state_embed"]
                else:
                    next_s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                next_mu = self.target_actor(next_s).squeeze(1)  # bat x dim_act

                if self._args['env_name'].startswith("mujoco"):
                    next_s = next_input_dict["state_embed"]
                else:
                    next_s = self.target_extra_critic_obs_enc(next_input_dict["state_embed"])
                next_Q = self.target_extra_critic(torch.cat([next_s, next_mu], dim=-1))  # b x 1
            mu = torch.tensor(action_taken, device=self._device)

            if self._args['env_name'].startswith("mujoco"):
                _s = input_dict["state_embed"]
            else:
                _s = self.main_extra_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
            Q = self.main_extra_critic(torch.cat([_s, mu], dim=-1))

            # === Bellman Error
            target = rewards + next_Q * reversed_dones * self._args["retrieve_Qnet_gamma"]
            extra_value_loss = torch.nn.functional.mse_loss(Q, target)

            self.opt_extra_critic.zero_grad()
            extra_value_loss.backward()
            if self._args["if_grad_clip"]: clip_grad(self.main_extra_critic.parameters(), 1.0)
            self.opt_extra_critic.step()

        if self._args["WOLP_if_joint_actor"]:
            with torch.no_grad():
                mu = self.target_actor(next_input_dict["state_embed"], if_joint_update=True)  # bat x list * dim_act
                next_Q = self.target_joint_critic(
                    torch.cat([next_input_dict["state_embed"], mu], dim=-1))  # b x 1

                # Avoid backprop to Actor
                mu = torch.tensor(query, device=self._device).reshape([query.shape[0], -1])
            Q = self.main_joint_critic(torch.cat([input_dict["state_embed"], mu], dim=-1))

            # === Bellman Error
            target = rewards + next_Q * reversed_dones * self._args["retrieve_Qnet_gamma"]
            joint_value_loss = torch.nn.functional.mse_loss(Q, target)

            self.opt_joint_critic.zero_grad()
            joint_value_loss.backward()
            clip_grad(self.main_joint_critic.parameters(), 1.0)
            self.opt_joint_critic.step()

        if self._args["WOLP_if_ar"]:
            if self._args["WOLP_ar_if_use_full_next_Q"]:  # for wolp-MultiAgent
                with torch.no_grad():
                    # === Next-Q on Current list: Proto-action
                    if self._args["obs_enc_apply"]:
                        next_s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                    else:
                        next_s = next_input_dict["state_embed"][:, None, :]

                    # Get the weight from Actor
                    next_mu = self.target_actor(next_s)  # batch x sequence-len x dim_act

                    # === Next-Q: Critic
                    if self._args["obs_enc_apply"]:
                        next_s = self.target_ar_critic_obs_enc(next_input_dict["state_embed"])
                    else:
                        next_s = next_input_dict["state_embed"][:, None, :]
                    next_Q = self.target_ar_critic(state=next_s, action_seq=next_mu)  # b x seq
            else:  # for all other ARDDPGs
                with torch.no_grad():
                    # === Next-Q on Next list: Proto-action
                    if self._args["obs_enc_apply"]:
                        next_s = self.target_actor_obs_enc(next_input_dict["state_embed"])
                    else:
                        next_s = next_input_dict["state_embed"][:, None, :]

                    # === Next-Q on Next list: Critic
                    if self._args["obs_enc_apply"]:
                        next_critic_s = self.target_ar_critic_obs_enc(next_input_dict["state_embed"])
                    else:
                        next_critic_s = next_input_dict["state_embed"][:, None, :]

                    # === Different ways to compute target Q-value === #
                    if self._args["WOLP_ar_type_listwise_update"].lower() == "next-ts-same-index":
                        next_mu = self.target_actor(next_s)  # batch x list_len x dim_act
                        next_Q = self.target_ar_critic(state=next_critic_s, action_seq=next_mu)  # b x 1
                    else:
                        # Manipulate the Q-val for the last list-index
                        if self._args["WOLP_if_new_list_obj"][0] == 'r':
                            next_mu = self.target_actor(next_s)[:, 0, :][:, None, :]  # batch x 1 x dim_act
                            _next_Q = self.target_ar_critic(state=next_critic_s, action_seq=next_mu,
                                                            if_next_list_Q=True)  # b x 1
                        elif self._args["WOLP_if_new_list_obj"] == 1:
                            raise NotImplementedError
                            # ===== From t: Bandit
                            # from selection Q-net
                            _next_Q = select_Q_max[:, None]
                        else: raise ValueError

                        if self._args["WOLP_ar_type_listwise_update"].lower() == "0-th-next-ts":
                            next_Q = _next_Q
                        elif self._args["WOLP_ar_type_listwise_update"].lower() == "next-list-index":
                            # For each list index, the target Q-value is computed based on the immediately next index
                            # === Next-Q on Current list: Proto-action
                            if self._args["obs_enc_apply"]:
                                curr_s = self.target_actor_obs_enc(input_dict["state_embed"])
                            else:
                                curr_s = input_dict["state_embed"][:, None, :]

                            # Get the weight from Actor
                            curr_mu = self.target_actor(curr_s)  # batch x sequence-len x dim_act

                            # === Next-Q on Current list: Critic
                            if self._args["obs_enc_apply"]:
                                curr_s = self.target_ar_critic_obs_enc(input_dict["state_embed"])

                            curr_Q = self.target_ar_critic(state=curr_s, action_seq=curr_mu)  # b x seq
                            next_Q = torch.cat([curr_Q[:, 1:], _next_Q], dim=-1)
                        else:
                            raise ValueError

            mu = torch.tensor(query, device=self._device)
            # === Q on Taken list-action: Critic
            if self._args["obs_enc_apply"]:
                _s = self.main_ar_critic_obs_enc(input_dict["state_embed"])
            else:
                _s = input_dict["state_embed"][:, None, :]
            Q = self.main_ar_critic(state=_s, action_seq=mu)

            if self._args["WOLP_if_ar_contextual_prop"]:  # Only use the last entry and rely on backprop thru time!
                print('This needs to be checked if working')
                import ipdb;
                ipdb.set_trace()
                next_Q, Q = next_Q[:, -1][:, None], Q[:, -1][:, None]
                if self._args["WOLP_cascade_list_len"] > 1 and \
                        self._args["WOLP_cascade_type_list_reward"] in ["elementwise", "last"]:
                    listwise_rewards = listwise_rewards[:, -1][:, None]

            # === Bellman Error
            if self._args["WOLP_discounted_cascading"] and not self._args["WOLP_ar_if_use_full_next_Q"]:
                list_len = self._args["WOLP_cascade_list_len"]
                discount_exponent = torch.tensor(list_len - np.arange(list_len),
                                                 dtype=torch.float32,
                                                 device=self._device)[None, :]
                target_Q = next_refine_Q if self._args["WOLP_refineQ_target"] else next_Q
                target = torch.pow(self._args["retrieve_Qnet_gamma"], discount_exponent - 1) * rewards + \
                         torch.pow(self._args["retrieve_Qnet_gamma"], discount_exponent) * target_Q * reversed_dones
            elif (self._args["WOLP_cascade_type_list_reward"] in ["elementwise", "last"]) and \
                    not self._args["WOLP_ar_if_use_full_next_Q"]:
                listwise_reversed_dones = torch.cat([
                    torch.ones([rewards.shape[0], self._args["WOLP_cascade_list_len"] - 1], device=self._device),
                    reversed_dones
                ], dim=1)
                target = listwise_rewards + next_Q * listwise_reversed_dones * self._args["retrieve_Qnet_gamma"]
            else:
                target = rewards + next_Q * reversed_dones * self._args["retrieve_Qnet_gamma"]
            ar_value_loss = torch.nn.functional.mse_loss(Q, target)

            if self.opt_ar_critic_obs_enc is not None: self.opt_ar_critic_obs_enc.zero_grad()
            self.opt_ar_critic.zero_grad()
            ar_value_loss.backward()
            if self._args["if_grad_clip"]: clip_grad(self.main_ar_critic.parameters(), 1.0)
            if self.opt_ar_critic_obs_enc is not None and self._args["if_grad_clip"]:
                clip_grad(self.main_ar_critic_obs_enc.parameters(), 1.0)
            self.opt_ar_critic.step()
            if self.opt_ar_critic_obs_enc is not None: self.opt_ar_critic_obs_enc.step()

        # === Update Actor
        train_actor = (self.main_actor is not None and self._args['global_ts'] > self._args['delayed_actor_training'])
        if train_actor:
            if self._args["WOLP_if_ar"]:
                if self._args["obs_enc_apply"]:
                    _s = self.main_actor_obs_enc(input_dict["state_embed"])
                else:
                    _s = input_dict["state_embed"][:, None, :]
                mu = self.main_actor(_s)
                Q = self.main_ar_critic(state=_s, action_seq=mu)

                if self._args["WOLP_if_ar_contextual_prop"]:  # Use the last entry to rely on backprop thru time
                    Q = Q[:, -1]  # batch x 1
            elif self._args["WOLP_if_joint_actor"]:
                if self._args["obs_enc_apply"]:
                    _s = self.main_actor_obs_enc(input_dict["state_embed"])
                else:
                    _s = input_dict["state_embed"]
                mu = self.main_actor(_s, if_joint_update=True)  # batch x list-len * dim-action
                Q = self.main_joint_critic(torch.cat([_s, mu], -1))
            else:
                if self._args["obs_enc_apply"]:
                    _s = self.main_actor_obs_enc(input_dict["state_embed"]).squeeze(1)
                else:
                    _s = input_dict["state_embed"]
                mu = self.main_actor(_s)

                if self._args["obs_enc_apply"]:
                    _s = self.main_ref_critic_obs_enc(input_dict["state_embed"]).squeeze(1)
                else:
                    _s = input_dict["state_embed"]

                if self._args["WOLP_if_dual_critic"]:
                    Q = self.main_extra_critic(torch.cat([_s[:, None, :].repeat(1, mu.shape[1], 1), mu], dim=-1))
                else:
                    Q = self.main_ref_critic(torch.cat([_s[:, None, :].repeat(1, mu.shape[1], 1), mu], dim=-1))

            policy_loss = -Q.sum(dim=-1).mean()  # Average over samples & preserve contribution of slate indices

            if self.opt_actor_obs_enc is not None: self.opt_actor_obs_enc.zero_grad()
            self.opt_actor.zero_grad()
            policy_loss.backward()
            if self._args["if_grad_clip"]: clip_grad(self.main_actor.parameters(), 1.0)  # gradient clipping
            clip_grad(self.main_actor.parameters(), 1.0)  # gradient clipping
            self.opt_actor.step()
            if self.opt_actor_obs_enc is not None and self._args['if_grad_clip']:
                clip_grad(self.main_actor_obs_enc.parameters(), 1.0)
            if self.opt_actor_obs_enc is not None: self.opt_actor_obs_enc.step()

        res = {
            "value_loss": value_loss.item(),  # "next_Q_var": _var, "next_Q_mean": _mean,
            "policy_loss": policy_loss.item() if train_actor else 0.0,
            "loss": value_loss.item() + policy_loss.item() if train_actor else value_loss.item(),
        }

        if self._args["WOLP_if_ar"]:
            res["ar_value_loss"] = ar_value_loss.item()

        if self._args["WOLP_if_joint_actor"]:
            res["joint_value_loss"] = joint_value_loss.item()

        if self._args["WOLP_if_dual_critic"]:
            res["extra_value_loss"] = extra_value_loss

        return res

    def _sync(self, tau: float = 0.0):
        """ when _type is None, we update both actor and critic """
        if tau > 0.0:  # Soft update of params
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

            for param, target_param in zip(self.main_ref_critic.parameters(), self.target_ref_critic.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self.main_ref_critic_obs_enc is not None:
                for param, target_param in zip(self.main_ref_critic_obs_enc.parameters(),
                                               self.target_ref_critic_obs_enc.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["WOLP_if_ar"]:
                for param, target_param in zip(self.main_ar_critic.parameters(), self.target_ar_critic.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                if self.main_ar_critic_obs_enc is not None:
                    for param, target_param in zip(self.main_ar_critic_obs_enc.parameters(),
                                                   self.target_ar_critic_obs_enc.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["WOLP_if_joint_actor"]:
                for param, target_param in zip(self.main_joint_critic.parameters(),
                                               self.target_joint_critic.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                    if self.main_joint_critic_obs_enc is not None:
                        for param, target_param in zip(self.main_joint_critic_obs_enc.parameters(),
                                                       self.target_joint_critic_obs_enc.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["WOLP_if_dual_critic"]:
                for param, target_param in zip(self.main_extra_critic.parameters(),
                                               self.target_extra_critic.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                    if self.main_extra_critic_obs_enc is not None:
                        for param, target_param in zip(self.main_extra_critic_obs_enc.parameters(),
                                                       self.target_extra_critic_obs_enc.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
        else:
            if self.main_actor is not None:
                self.target_ref_critic.load_state_dict(self.main_ref_critic.state_dict())
                self.target_actor.load_state_dict(self.main_actor.state_dict())

            if self._args["obs_enc_apply"]:
                self.target_actor_obs_enc.load_state_dict(self.main_actor_obs_enc.state_dict())
                self.target_ref_critic_obs_enc.load_state_dict(self.main_ref_critic_obs_enc.state_dict())

            if self._args["WOLP_if_ar"]:
                self.target_ar_critic.load_state_dict(self.main_ar_critic.state_dict())

    def _train(self):
        if self.main_actor is not None:
            self.main_actor.train()
            self.target_actor.train()

        self.main_ref_critic.train()
        self.target_ref_critic.train()

        if self._args["obs_enc_apply"] and self.opt_actor_obs_enc is not None:
            self.main_actor_obs_enc.train()
            self.target_actor_obs_enc.train()
            self.main_ref_critic_obs_enc.train()
            self.target_ref_critic_obs_enc.train()
            if self._args["WOLP_if_ar"]:
                if self.main_ar_critic_obs_enc is not None:
                    self.main_ar_critic_obs_enc.train()
                    self.target_ar_critic_obs_enc.train()
            if self._args["WOLP_if_joint_actor"]:
                if self.main_extra_critic_obs_enc is not None:
                    self.main_extra_critic_obs_enc.train()
                    self.target_joint_critic_obs_enc.train()
            if self._args["WOLP_if_dual_critic"]:
                if self.main_extra_critic_obs_enc is not None:
                    self.main_extra_critic_obs_enc.train()
                    self.target_extra_critic_obs_enc.train()

        if self._args["WOLP_if_ar"]:
            self.main_ar_critic.train()
            self.target_ar_critic.train()

        if self._args["WOLP_if_dual_critic"]:
            self.main_extra_critic.train()
            self.target_extra_critic.train()

    def _eval(self):
        if self.main_actor is not None:
            self.main_actor.eval()
            self.target_actor.eval()

        self.main_ref_critic.eval()
        self.target_ref_critic.eval()

        if self._args["obs_enc_apply"] and (self.opt_actor_obs_enc is not None):
            self.main_actor_obs_enc.eval()
            self.target_actor_obs_enc.eval()
            self.main_ref_critic_obs_enc.eval()
            self.target_ref_critic_obs_enc.eval()
            if self._args["WOLP_if_ar"]:
                if self.main_ar_critic_obs_enc is not None:
                    self.main_ar_critic_obs_enc.eval()
                    self.target_ar_critic_obs_enc.eval()
            if self._args["WOLP_if_joint_actor"]:
                if self.main_extra_critic_obs_enc is not None:
                    self.main_extra_critic_obs_enc.eval()
                    self.target_joint_critic_obs_enc.eval()
            if self._args["WOLP_if_dual_critic"]:
                if self.main_extra_critic_obs_enc is not None:
                    self.main_extra_critic_obs_enc.eval()
                    self.target_extra_critic_obs_enc.eval()

        if self._args["WOLP_if_ar"]:
            self.main_ar_critic.eval()
            self.target_ar_critic.eval()

        if self._args["WOLP_if_dual_critic"]:
            self.main_extra_critic.eval()
            self.target_extra_critic.eval()

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
        if self._args["obs_enc_apply"]:
            torch.save(self.main_actor_obs_enc.state_dict(), os.path.join(save_dir, f"main_actor_obs_enc.pkl"))
            torch.save(self.target_actor_obs_enc.state_dict(), os.path.join(save_dir, f"target_actor_obs_enc.pkl"))
            torch.save(self.opt_actor_obs_enc.state_dict(), os.path.join(save_dir, f"opt_actor_obs_enc.pkl"))
        torch.save(self.main_actor.state_dict(), os.path.join(save_dir, f"main_actor.pkl"))
        torch.save(self.target_actor.state_dict(), os.path.join(save_dir, f"target_actor.pkl"))
        torch.save(self.opt_actor.state_dict(), os.path.join(save_dir, f"opt_actor.pkl"))

        ## Critic
        if self._args["obs_enc_apply"]:
            torch.save(self.main_ref_critic_obs_enc.state_dict(),
                       os.path.join(save_dir, f"main_ref_critic_obs_enc.pkl"))
            torch.save(self.target_ref_critic_obs_enc.state_dict(),
                       os.path.join(save_dir, f"target_ref_critic_obs_enc.pkl"))
            torch.save(self.opt_ref_critic_obs_enc.state_dict(), os.path.join(save_dir, f"opt_ref_critic_obs_enc.pkl"))
        torch.save(self.main_ref_critic.state_dict(), os.path.join(save_dir, f"main_ref_critic.pkl"))
        torch.save(self.target_ref_critic.state_dict(), os.path.join(save_dir, f"target_ref_critic.pkl"))
        torch.save(self.opt_ref_critic.state_dict(), os.path.join(save_dir, f"opt_ref_critic.pkl"))

        if self._args["WOLP_if_joint_actor"]:
            if self._args["obs_enc_apply"]:
                torch.save(self.main_joint_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"main_joint_critic_obs_enc.pkl"))
                torch.save(self.target_joint_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"target_joint_critic_obs_enc.pkl"))
                torch.save(self.opt_joint_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"opt_joint_critic_obs_enc.pkl"))
            torch.save(self.main_joint_critic.state_dict(), os.path.join(save_dir, f"main_joint_critic.pkl"))
            torch.save(self.target_joint_critic.state_dict(), os.path.join(save_dir, f"target_joint_critic.pkl"))
            torch.save(self.opt_joint_critic.state_dict(), os.path.join(save_dir, f"opt_joint_critic.pkl"))

        if self._args["WOLP_if_ar"]:
            if self._args["obs_enc_apply"]:
                torch.save(self.main_ar_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"main_ar_critic_obs_enc.pkl"))
                torch.save(self.target_ar_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"target_ar_critic_obs_enc.pkl"))
                torch.save(self.opt_ar_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"opt_ar_critic_obs_enc.pkl"))
            torch.save(self.main_ar_critic.state_dict(), os.path.join(save_dir, f"main_ar_critic.pkl"))
            torch.save(self.target_ar_critic.state_dict(), os.path.join(save_dir, f"target_ar_critic.pkl"))
            torch.save(self.opt_ar_critic.state_dict(), os.path.join(save_dir, f"opt_ar_critic.pkl"))

        if self._args["WOLP_if_dual_critic"]:
            if self._args["obs_enc_apply"]:
                torch.save(self.main_extra_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"main_extra_critic_obs_enc.pkl"))
                torch.save(self.target_extra_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"target_extra_critic_obs_enc.pkl"))
                torch.save(self.opt_extra_critic_obs_enc.state_dict(),
                           os.path.join(save_dir, f"opt_extra_critic_obs_enc.pkl"))
            torch.save(self.main_extra_critic.state_dict(), os.path.join(save_dir, f"main_extra_critic.pkl"))
            torch.save(self.target_extra_critic.state_dict(), os.path.join(save_dir, f"target_extra_critic.pkl"))
            torch.save(self.opt_extra_critic.state_dict(), os.path.join(save_dir, f"opt_extra_critic.pkl"))

    def _load(self, load_dir):
        logging("Load the agent: {}".format(load_dir))
        ## Actor
        if self._args["obs_enc_apply"]:
            self.main_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"main_actor_obs_enc.pkl")))
            self.target_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"target_actor_obs_enc.pkl")))
            self.opt_actor_obs_enc.load_state_dict(torch.load(os.path.join(load_dir, f"opt_actor_obs_enc.pkl")))
        self.main_actor.load_state_dict(torch.load(os.path.join(load_dir, f"main_actor.pkl")))
        self.target_actor.load_state_dict(torch.load(os.path.join(load_dir, f"target_actor.pkl")))
        self.opt_actor.load_state_dict(torch.load(os.path.join(load_dir, f"opt_actor.pkl")))

        ## Critic
        if self._args["obs_enc_apply"]:
            self.main_ref_critic_obs_enc.load_state_dict(
                torch.load(os.path.join(load_dir, f"main_ref_critic_obs_enc.pkl")))
            self.target_ref_critic_obs_enc.load_state_dict(
                torch.load(os.path.join(load_dir, f"target_ref_critic_obs_enc.pkl")))
            self.opt_ref_critic_obs_enc.load_state_dict(
                torch.load(os.path.join(load_dir, f"opt_ref_critic_obs_enc.pkl")))

        self.main_ref_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_ref_critic.pkl")))
        self.target_ref_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_ref_critic.pkl")))
        self.opt_ref_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_ref_critic.pkl")))

        if self._args["WOLP_if_joint_actor"]:
            if self._args["obs_enc_apply"]:
                self.main_joint_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_joint_critic_obs_enc.pkl")))
                self.target_joint_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_joint_critic_obs_enc.pkl")))
                self.opt_joint_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_joint_critic_obs_enc.pkl")))
            self.main_joint_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_joint_critic.pkl")))
            self.target_joint_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_joint_critic.pkl")))
            self.opt_joint_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_joint_critic.pkl")))

        if self._args["WOLP_if_ar"]:
            if self._args["obs_enc_apply"]:
                self.main_ar_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_ar_critic_obs_enc.pkl")))
                self.target_ar_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_ar_critic_obs_enc.pkl")))
                self.opt_ar_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_ar_critic_obs_enc.pkl")))
            self.main_ar_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_ar_critic.pkl")))
            self.target_ar_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_ar_critic.pkl")))
            self.opt_ar_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_ar_critic.pkl")))

        if self._args["WOLP_if_dual_critic"]:
            if self._args["obs_enc_apply"]:
                self.main_extra_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"main_extra_critic_obs_enc.pkl")))
                self.target_extra_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"target_extra_critic_obs_enc.pkl")))
                self.opt_extra_critic_obs_enc.load_state_dict(
                    torch.load(os.path.join(load_dir, f"opt_extra_critic_obs_enc.pkl")))
            self.main_extra_critic.load_state_dict(torch.load(os.path.join(load_dir, f"main_extra_critic.pkl")))
            self.target_extra_critic.load_state_dict(torch.load(os.path.join(load_dir, f"target_extra_critic.pkl")))
            self.opt_extra_critic.load_state_dict(torch.load(os.path.join(load_dir, f"opt_extra_critic.pkl")))
