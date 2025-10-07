import os
import torch
import numpy as np

from torch import optim
from copy import deepcopy

from large_rl.policy.arch.cnn import CNN, OldCNN, FlatMLP, Flat
from large_rl.embedding.base import BaseEmbedding
from large_rl.policy import RandomPolicy
from large_rl.commons.utils import logging

LARGE_NEG = -1000000.


class Agent(object):
    """ Boilerplate of agent """

    def __init__(self, args: dict, **kwargs):
        self.epoch = 0
        self._args = args
        self._if_train = True
        self._max_index = None  # common for all agents
        self._max_Q_improvement = None  # common for all agents
        self._max_Q_improvement_percent = None  # common for all agents
        self._rng = np.random.RandomState(self._args.get("seed", 2021))
        self._device = self._args.get("device", "cpu")
        if self._args["env_name"].startswith("mujoco"):
            self._candidate_list = None
        else:
            self._candidate_list = list(range(self._args["num_all_actions"]))
        self.model_save_path = self._args['agent_save_path']
        if args["save_dir"] != "":
            self.model_save_path = os.path.join(self._args['agent_save_path'], args["save_dir"])
        else:
            self.model_save_path = os.path.join(self._args['agent_save_path'],
                                                'seed{:05d}debug2022'.format(args["seed"]))
        self.model_load_path = self._args['agent_load_path']
        self.update_count = 0

    def continuous_select_action(self, obs: torch.tensor, act_embed_base: BaseEmbedding, epsilon: dict, **kwargs):
        """
            This function is largely different from wolp.py because of the continuous action v/s discrete action difference
        """
        policy_obs = torch.tensor(obs, dtype=torch.float32, device=self._device)

        # Prep input for agent
        input_dict = {"state_embed": policy_obs}

        # Main inference
        action = self._select_action(input_dict=input_dict, epsilon=epsilon, **kwargs)

        # Construct the Agent response
        res = {"action": np.asarray(action)}
        res["max_index"] = self._max_index if self._max_index is not None else np.zeros((obs.shape[0],), dtype=np.int32)
        res["max_Q_improvement"] = self._max_Q_improvement if self._max_Q_improvement is not None else np.zeros((obs.shape[0],), dtype=np.float32)
        res["max_Q_improvement_percent"] = self._max_Q_improvement_percent if self._max_Q_improvement_percent is not None else np.zeros((obs.shape[0],), dtype=np.float32)

        '''
        if self._args["agent_type"].startswith("wolp"):
            res["query"] = self._query  # Query from Actor in Wolp / LIRD
        else:
            res["query"] = None
        '''

        if self._args["agent_type"].startswith("wolp"):
            if self._args["WOLP_dual_exp_if_ignore"]:
                _mask = self._selectionQ_mask
            else:
                _mask = np.asarray([False] * obs.shape[0])

            if self._query is not None:
                _mat = np.ones((obs.shape[0], self._query.shape[1], self._query.shape[-1]),
                               dtype=np.float32) * LARGE_NEG
                _mat[~_mask] = self._query[~_mask]
                res["query"] = _mat
            else:
                res["query"] = np.ones((obs.shape[0], 1, 1)) * LARGE_NEG

            if self._query_max is not None:
                _mat = np.ones((obs.shape[0], self._query_max.shape[1]),
                               dtype=np.float32) * LARGE_NEG
                _mat[~_mask] = self._query_max[~_mask]
                res["query_max"] = _mat
            else:
                res["query_max"] = np.ones((obs.shape[0], 1)) * LARGE_NEG
        else:
            res["query"] = None

        return res

    def select_action(self, obs: torch.tensor, act_embed_base: BaseEmbedding, epsilon: dict, naive_eval=False, **kwargs):
        if self._args["env_name"].lower().startswith("mujoco"):
            return self.continuous_select_action(obs, act_embed_base, epsilon, naive_eval=naive_eval)

        # Instantiate the main action base and we will fill this using random policy and agent
        action = np.zeros((obs.shape[0], self._args["recsim_slate_size"]))

        # === Epsilon decay policy and retrieve the input for policy
        if self._args["agent_type"].startswith("wolp") and (epsilon["critic"] < 1.0):  # When in training! Not Roll-out
            # If Wolp, we explore inside the agent
            mask = np.asarray([False] * obs.shape[0])
        else:
            # If DQN, we explore here
            _rand = self._rng.uniform(low=0.0, high=1.0, size=obs.shape[0])  # With prob epsilon select a random action
            mask = _rand < epsilon["critic"]  # epsilon decay; eps-critic for dqn
        random_policy_obs, policy_obs = obs[mask, :], obs[~mask, :]  # total would be: batch_step_size x dim_state

        # === Select actions using the random policy
        # Slate size is always 1. This is about listwise RL tasks, which we do not consider.
        action_random_ind = self.select_random_action(batch_size=sum(mask), slate_size=self._args["recsim_slate_size"])
        if not np.alltrue(mask):
            if self._args["env_name"].startswith("recsim") or (
                    self._args["env_name"].lower() == "mine" and self._args['mw_obs_flatten']):
                if not self._args["agent_type"].startswith("wolp"):  # if DQN!
                    if self._args["env_name"].lower() == "recsim-data":
                        # (batch * candidates) x history x dim-state
                        policy_obs = policy_obs.repeat(self._args["num_all_actions"], 1, 1)
                    else:
                        # batch x candidates x dim-state
                        policy_obs = policy_obs[:, None, :].repeat(1, self._args["num_all_actions"], 1)
            elif self._args["env_name"].lower() == "mine":  # image based observation
                if self._args["agent_type"].startswith("wolp"):  # if wolp and others
                    # batch x candidates x H x W x C
                    policy_obs = policy_obs[:, None, ...]
                else:  # if DQN!
                    # batch x candidates x H x W x C
                    policy_obs = policy_obs[:, None, ...].repeat(1, self._args["num_all_actions"], 1, 1, 1)
            else:
                raise ValueError

            # Prep input for agent
            input_dict = {"state_embed": policy_obs,
                          # "act_embed": act_embed_base.embedding_torch[None, ...].repeat(policy_obs.shape[0], 1, 1)}
                          "act_embed": act_embed_base.embedding_torch[None, ...].repeat(sum(~mask), 1, 1)}

            # Main inference
            action_policy_id = self._select_action(input_dict=input_dict, epsilon=epsilon, naive_eval=naive_eval, **kwargs)

            # Concatenate with the random action
            if action_random_ind.shape[0] == 0:
                action = action_policy_id
            else:
                action[mask, :], action[~mask, :] = action_random_ind, action_policy_id
        else:
            # Especially because it applies only for initial replay buffer filling
            action = action_random_ind  # batch_step_size x slate_size

            if self._args["agent_type"].startswith("wolp"):
                if self._args["WOLP_if_new_exploration"]:
                    # Prep input for agent
                    input_dict = {"state_embed": torch.tensor(random_policy_obs, device=self._device),
                                  "act_embed": act_embed_base.embedding_torch[None, ...].repeat(
                                      random_policy_obs.shape[0], 1, 1)
                                  }

                    # Main inference
                    action = self._select_action(input_dict=input_dict, epsilon=epsilon, naive_eval=naive_eval, **kwargs)
                else:
                    # === Sample query
                    if self._args["WOLP_cascade_list_len"] > 1:
                        # Remove the already selected items from the candidates
                        candidate_lists = list()
                        for i in range(sum(mask)):
                            _list = self._candidate_list.copy()
                            _list.pop(action_random_ind[i][0])
                            candidate_lists.append(_list)

                        _id = self.select_random_action(batch_size=sum(mask),
                                                        slate_size=self._args["WOLP_cascade_list_len"] - 1,
                                                        candidate_lists=candidate_lists)
                        _id = np.hstack([action_random_ind, _id])  # Add the main random action => batch x list-len
                        np.random.shuffle(_id.T)  # shuffling the ids along the row
                    else:
                        _id = action_random_ind  # cddpg with list-len = 1 or Wolp
                    self._query = act_embed_base.get(index=_id, if_np=True)  # proto-actions sampled from actor

                    # === Sample topK candidates
                    # Add the topK candidates for each cascaded actor
                    # Remove the already selected items from the candidates
                    candidate_lists = list()
                    # For all batches, identify the remaining candidate items
                    for i in range(sum(mask)):
                        _list = list(np.delete(self._candidate_list.copy(), _id[i]))
                        candidate_lists.append(_list)

                    _num = (self._args["WOLP_topK"] - 1) * self._args["WOLP_cascade_list_len"]
                    __id = self.select_random_action(batch_size=sum(mask),
                                                     slate_size=_num, candidate_lists=candidate_lists)
                    _id = np.hstack([_id, __id])  # batch x (list-len * topK)
                    self._topk_act = _id.reshape(
                        (sum(mask), self._args["WOLP_cascade_list_len"], self._args["WOLP_topK"]))

                    topk_embed = act_embed_base.get(index=_id, if_np=False)
                    _o = obs[:, None, :].repeat(1, topk_embed.shape[1], 1)
                    Q_vals = self.main_ref_critic(torch.cat([_o, topk_embed], dim=-1)).squeeze(-1)
                    self._candidate_Q_mean = Q_vals.mean(dim=-1).cpu().detach().numpy()

        # Construct the Agent response
        res = {"action": np.asarray(action).astype(np.int)}
        res["max_index"] = self._max_index if self._max_index is not None else np.asarray([0])
        res["max_Q_improvement"] = self._max_Q_improvement if self._max_Q_improvement is not None else np.asarray([0])
        res["max_Q_improvement_percent"] = self._max_Q_improvement_percent if self._max_Q_improvement_percent is not None else np.asarray([0])

        if self._args["agent_type"].startswith("wolp"):
            if self._args["WOLP_dual_exp_if_ignore"]:
                _mask = self._selectionQ_mask
            else:
                _mask = np.asarray([False] * obs.shape[0])

            if self._query is not None:
                _mat = np.ones((obs.shape[0], self._query.shape[1], self._query.shape[-1])) * (-1)
                _mat[~_mask] = self._query[~_mask]
                res["query"] = _mat
            else:
                res["query"] = np.ones((obs.shape[0], 1, 1)) * (-1)

            if self._query_max is not None:
                _mat = np.ones((obs.shape[0], self._query_max.shape[1])) * (-1)
                _mat[~_mask] = self._query_max[~_mask]
                res["query_max"] = _mat
            else:
                res["query_max"] = np.ones((obs.shape[0], 1)) * (-1)

            if self._topk_act is not None:
                _mat = np.ones((obs.shape[0], self._topk_act.shape[1], self._topk_act.shape[-1])) * (-1)
                _mat[~_mask] = self._topk_act[~_mask]
                res["topk_act"] = _mat
            else:
                res["topk_act"] = np.ones((obs.shape[0], 1, 1)) * (-1)
        else:
            res["query"] = None
            res["topk_act"] = None

        # if self._args["agent_type"].startswith("wolp"):
        #     res["query"] = self._query  # Query from Actor in Wolp / LIRD
        #     res["topk_act"] = self._topk_act
        # else:
        #     res["query"] = None
        #     res["topk_act"] = None

        return res

    def select_random_action(self, batch_size: int, slate_size: int = None, candidate_lists=None):
        if self._args["env_name"].lower().startswith("mujoco"):
            return np.array([self._args["reacher_action_space"].sample() for _ in range(batch_size)])
        else:
            if slate_size is None: slate_size = self._args["recsim_slate_size"]
            if candidate_lists is None:
                candidate_lists = np.tile(A=np.asarray(self._candidate_list)[None, :], reps=(batch_size, 1))
            return np.asarray(
                [RandomPolicy.select_action(batch_size=1, candidate_list=candidate_lists[i], slate_size=slate_size)[0]
                 for i in range(batch_size)]
            )

    def _select_action(self, input_dict: dict, **kwargs) -> torch.tensor:
        raise NotImplementedError

    def update(self, obses: torch.tensor, actions: np.ndarray, rewards: np.ndarray, next_obses: torch.tensor,
               dones: np.ndarray, act_embed_base: BaseEmbedding, **kwargs):
        self.update_count += 1
        if self._args["env_name"].startswith("recsim") or (
                self._args["env_name"].lower() == "mine" and self._args['mw_obs_flatten']):
            if self._args["agent_type"] not in ["wolp", "wolp-sac", "random"]:  # if DQN!
                if self._args["env_name"].lower() == "recsim-data":
                    # (batch * candidates) x history x dim-state
                    obses = obses.repeat(1, 1, 1)
                    next_obses = next_obses.repeat(self._args["num_all_actions"], 1, 1)
                else:
                    # batch x candidates x dim-state
                    obses = obses[:, None, :].repeat(1, 1, 1)
                    next_obses = next_obses[:, None, :].repeat(1, self._args["num_all_actions"], 1)
        elif self._args["env_name"].lower() == "mine":  # image based observation
            if self._args["agent_type"] not in ["wolp", "wolp-sac", "random"]:  # if DQN!
                # batch x candidates x H x W x C
                obses = obses[:, None, ...].repeat(1, 1, 1, 1, 1)
                next_obses = next_obses[:, None, ...].repeat(1, self._args["num_all_actions"], 1, 1, 1)
            else:
                # batch x candidates x H x W x C
                obses = obses[:, None, ...].repeat(1, 1, 1, 1, 1)
                next_obses = next_obses[:, None, ...].repeat(1, obses.shape[1], 1, 1, 1)
        elif self._args["env_name"].startswith("mujoco"):
            pass
        else:
            raise ValueError

        # convert them into tensors
        rewards = torch.tensor(rewards.astype(np.float32), device=self._device)
        reversed_dones = torch.tensor((1 - dones).astype(np.float32), device=self._device)
        if len(reversed_dones.shape) == 1:
            reversed_dones = reversed_dones[:, None]

        # Prep input for agent
        if self._args["env_name"].startswith("mujoco"):
            act_embed = None
        else:
            act_embed = act_embed_base.embedding_torch[None, ...].repeat(self._args["batch_size"], 1, 1).detach()
        input_dict = {"state_embed": obses, "act_embed": act_embed}
        next_input_dict = {"state_embed": next_obses, "act_embed": act_embed}  # Intended!! since stationary act emb
        self._update(
            input_dict=input_dict, next_input_dict=next_input_dict, actions=actions, rewards=rewards,
            reversed_dones=reversed_dones, act_embed_base=act_embed_base, **kwargs
        )

    def _update(self,
                input_dict: dict,
                next_input_dict: dict,
                actions: np.ndarray,
                rewards: torch.tensor,
                reversed_dones: torch.tensor,
                act_embed_base: BaseEmbedding, **kwargs):
        raise NotImplementedError

    def save_model(self, epoch: int = 0):
        save_dir = self.model_save_path
        save_dir = os.path.join(save_dir, f"ep{epoch}")
        self._save(save_dir)

    def load_model(self, epoch: int = 0, no_epoch=False):
        load_dir = self.model_load_path
        if not no_epoch:
            load_dir = os.path.join(load_dir, f"ep{epoch}")
        self._load(load_dir)

    def _save(self, save_dir):
        raise NotImplementedError

    def _load(self, load_dir):
        raise NotImplementedError

    def _instantiate_obs_encoder(self):
        if self._args["env_name"].lower() == "recsim-data":
            # Taken from /recsim_data/user_model/reward_model.py
            from large_rl.encoder.set_summariser import DeepSet, BiLSTM, Transformer as Trans
            if self._args["recsim_rm_obs_enc_type"] == "deepset":
                _obs_enc = DeepSet(
                    dim_in=54, dim_out=self._args["recsim_dim_embed"], p_drop=self._args["recsim_rm_dropout"]
                ).to(device=self._args["device"])
            elif self._args["recsim_rm_obs_enc_type"] == "lstm":
                _obs_enc = BiLSTM(
                    dim_in=54, dim_out=self._args["recsim_dim_embed"], p_drop=self._args["recsim_rm_dropout"]
                ).to(device=self._args["device"])
            elif self._args["recsim_rm_obs_enc_type"] == "transformer":
                _obs_enc = Trans(
                    dim_in=54, dim_out=self._args["recsim_dim_embed"], p_drop=self._args["recsim_rm_dropout"]
                ).to(device=self._args["device"])
            else:
                raise ValueError
        else:
            img_size = self._args['mw_observation_size']
            dim_in_channels = self._args['mw_obs_channel']
            if self._args["mw_type_obs_enc"].lower() == "cnn":
                _obs_enc = CNN(img_size=img_size, dim_in_channels=dim_in_channels,
                               channels=self._args['mw_enc_channels']).to(
                    self._device)
            elif self._args["mw_type_obs_enc"].lower() == "old-cnn":
                _obs_enc = OldCNN(img_size=img_size, dim_in_channels=dim_in_channels,
                                  dim_out=self._args["mw_dim_state"]).to(self._device)
            elif self._args["mw_type_obs_enc"].lower() == "flat-mlp":
                _obs_enc = FlatMLP(img_size=img_size, dim_in_channels=dim_in_channels,
                                   dim_out=self._args["mw_dim_state"]).to(self._device)
            elif self._args["mw_type_obs_enc"].lower() == "flat":
                _obs_enc = Flat(img_size=img_size, dim_in_channels=dim_in_channels).to(self._device)
            else:
                raise ValueError
        return _obs_enc

    def _define_obs_encoders(self, actor=True, critic=True, twin_q=False):
        if actor:
            self.main_actor_obs_enc = self._instantiate_obs_encoder()
            self.target_actor_obs_enc = deepcopy(self.main_actor_obs_enc)
            self.opt_actor_obs_enc = optim.Adam(self.main_actor_obs_enc.parameters(), lr=self._args["mw_obs_enc_lr"])
            logging(self.main_actor_obs_enc)

        if critic:
            self.main_ref_critic_obs_enc = self._instantiate_obs_encoder()
            self.target_ref_critic_obs_enc = deepcopy(self.main_ref_critic_obs_enc)
            self.opt_ref_critic_obs_enc = optim.Adam(self.main_ref_critic_obs_enc.parameters(),
                                                     lr=self._args["mw_obs_enc_lr"])
            if twin_q:
                self.main_ref_critic_obs_enc_twin = self._instantiate_obs_encoder()
                self.target_ref_critic_obs_enc_twin = deepcopy(self.main_ref_critic_obs_enc_twin)
                self.opt_ref_critic_obs_enc_twin = optim.Adam(self.main_ref_critic_obs_enc_twin.parameters(),
                                                                lr=self._args["mw_obs_enc_lr"])
            logging(self.main_ref_critic_obs_enc)

    def _define_ar_critic_obs_encoders(self, twin_q=False):
        self.main_ar_critic_obs_enc = self._instantiate_obs_encoder()
        self.target_ar_critic_obs_enc = deepcopy(self.main_ar_critic_obs_enc)
        self.opt_ar_critic_obs_enc = optim.Adam(self.main_ar_critic_obs_enc.parameters(),
                                                lr=self._args["mw_obs_enc_lr"])
        if twin_q:
            self.main_ar_critic_obs_enc_twin = self._instantiate_obs_encoder()
            self.target_ar_critic_obs_enc_twin = deepcopy(self.main_ar_critic_obs_enc_twin)
            self.opt_ar_critic_obs_enc_twin = optim.Adam(self.main_ar_critic_obs_enc_twin.parameters(),
                                                           lr=self._args["mw_obs_enc_lr"])
        logging(self.main_ar_critic_obs_enc)

    def _define_joint_critic_obs_encoders(self, twin_q=False):
        self.main_joint_critic_obs_enc = self._instantiate_obs_encoder()
        self.target_joint_critic_obs_enc = deepcopy(self.main_joint_critic_obs_enc)
        self.opt_joint_critic_obs_enc = optim.Adam(self.main_joint_critic_obs_enc.parameters(),
                                                   lr=self._args["mw_obs_enc_lr"])
        if twin_q:
            self.main_joint_critic_obs_enc_twin = self._instantiate_obs_encoder()
            self.target_joint_critic_obs_enc_twin = deepcopy(self.main_joint_critic_obs_enc_twin)
            self.opt_joint_critic_obs_enc_twin = optim.Adam(self.main_joint_critic_obs_enc_twin.parameters(),
                                                              lr=self._args["mw_obs_enc_lr"])
        logging(self.main_joint_critic_obs_enc)

    def _define_extra_critic_obs_encoders(self, twin_q=False):
        self.main_extra_critic_obs_enc = self._instantiate_obs_encoder()
        self.target_extra_critic_obs_enc = deepcopy(self.main_extra_critic_obs_enc)
        self.opt_extra_critic_obs_enc = optim.Adam(self.main_extra_critic_obs_enc.parameters(),
                                                   lr=self._args["mw_obs_enc_lr"])
        if twin_q:
            self.main_extra_critic_obs_enc_twin = self._instantiate_obs_encoder()
            self.target_extra_critic_obs_enc_twin = deepcopy(self.main_extra_critic_obs_enc_twin)
            self.opt_extra_critic_obs_enc_twin = optim.Adam(self.main_extra_critic_obs_enc_twin.parameters(),
                                                              lr=self._args["mw_obs_enc_lr"])
        logging(self.main_extra_critic_obs_enc)

    def increment_epoch(self, _v=1):
        self.epoch += _v

    def reset(self, **kwargs):
        self._reset(**kwargs)

    def _reset(self, **kwargs):
        pass

    @property
    def info(self):
        return {"None": None}

    def sync(self, tau: float = 0.0):
        self._sync(tau=tau)

    def _sync(self, tau: float = 0.0):
        raise NotImplementedError

    def train(self):
        self._if_train = True
        self.reset()
        self._train()

    def _train(self):
        raise NotImplementedError

    def eval(self):
        self._if_train = False
        self.reset()
        self._eval()

    def _eval(self):
        raise NotImplementedError


class Random(Agent):
    def select_action(self, obs, candidate_list, **kwargs):
        action = np.asarray([
            RandomPolicy.select_action(batch_size=1, candidate_list=candidate_list[i],
                                       slate_size=self._args["recsim_slate_size"])[0]
            for i in range(candidate_list.shape[0])
        ])
        return {"action": action}
