"""
## Data
- User attributes: user.npy
- Item attributes: item_attr.npy
- Click log-data: train/val/test_log.csv
    - cols: click,hist_seq,item_id,user_id
"""
import os
import gym
import numpy as np
import pandas as pd

from typing import Dict

import torch
from gym import spaces

from large_rl.envs.recsim_data.observation import ObservationFactory
from large_rl.embedding.base import BaseEmbedding
from large_rl.envs.recsim_data.user_model.reward_model import UserModel


class DatasetEnv(gym.Env):
    def __init__(self, dict_embedding: Dict[str, BaseEmbedding], dict_dfLog: dict = None, args: dict = None):
        self._args = args
        self._rng = np.random.RandomState(self._args["env_seed"])
        self._ts = 0
        self._if_eval = False
        self.dict_dfLog = dict_dfLog
        self.dict_embedding = dict_embedding
        self.user_ids = list(range(self.dict_embedding["user"].shape[0]))
        self._reset_cursors()  # used to sample from the log data
        self.action_space = spaces.Box(low=0, high=args["num_all_actions"], shape=(args["num_envs"], 1), dtype=np.int)

        # Prepare the user model
        self.dict_UMs = dict()
        for name in ["offline", "online"]:
            self._args["rm_weight_path"] = os.path.join(self._args["recsim_data_dir"], f"trained_weight/{name}.pkl")
            self.dict_UMs[name] = UserModel(item_embedding=self.dict_embedding["item"], args=self._args)

    def _reset_cursors(self):
        self._ts = 0
        self._prev, self._cur = {"offline": 0, "online": 0}, \
            {"offline": self._args["num_envs"], "online": self._args["num_envs"]}

    def encode_obs(self):
        with torch.no_grad():
            obs = self.obs.make_obs(dict_embedding=self.dict_embedding, if_separate=False, device=self._args["device"])
            # obs = self.dict_UMs["online" if self._if_eval else "offline"].encode(obs=obs)
        return obs

    def step(self, action: np.ndarray):
        obs = self.obs.make_obs(dict_embedding=self.dict_embedding, if_separate=False, device=self._args["device"])
        reward, a_user = self.dict_UMs["online" if self._if_eval else "offline"].predict(obs=obs, action=action)
        reward = reward.astype(np.float32) * self._args.get("data_click_bonus", 1.0)
        done = np.asarray([self._ts >= self._args["max_episode_steps"]] * self._args["num_envs"])
        self.obs.update_history_seq(a_user=a_user)  # Update the user history sequence
        obs = self.encode_obs()
        self._ts += 1
        return obs, reward, done, {"gt_items": a_user}

    def reset(self):
        # used to sample from the log data
        self._reset_cursors()
        if ("offline" in self.dict_dfLog) and ("online" in self.dict_dfLog):
            flg = "online" if self._if_eval else "offline"
            if self._cur[flg] >= self.dict_dfLog[flg].shape[0]:
                sample = self.dict_dfLog[flg].sample(min(self.dict_dfLog[flg].shape[0], self._args["num_envs"]))
            else:
                # when it reaches the bottom of the log data.
                sample = self.dict_dfLog[flg].iloc[self._prev[flg]: min(self.dict_dfLog[flg].shape[0], self._cur[flg])]
            user_id = sample["userId"].values
        else:
            # In training without log data, we start with the zero-filled state representing a new user
            # We randomly sample the user attributes from the log data
            user_id = self._rng.choice(a=self.user_ids, size=self._args["num_envs"])
        self.obs = ObservationFactory(batch_size=self._args["num_envs"], user_id=user_id)
        obs = self.encode_obs()
        return obs

    @property
    def if_eval(self):
        return self._if_eval

    def train(self):
        self._if_eval = False

    def eval(self):
        self._if_eval = True

    @property
    def act_embedding(self):
        _emb = self.dict_embedding["item"].embedding_np

        from sklearn.preprocessing import MinMaxScaler  # this is to clearly constraint the action space after noise
        if self._args["DEBUG_size_action_space"] == "small":
            _emb = MinMaxScaler(feature_range=(0, 1)).fit_transform(_emb)
        elif self._args["DEBUG_size_action_space"] == "large":
            _emb = MinMaxScaler(feature_range=(-1, 1)).fit_transform(_emb)
        elif self._args["DEBUG_size_action_space"] == "unbounded":
            pass  # do nothing!
        return _emb

    @property
    def task_embedding(self):
        return self.dict_embedding["item"].embedding_np


def prep_emb(args):
    item_emb = BaseEmbedding(num_embeddings=args["num_all_actions"],
                             dim_embed=args["recsim_num_categories"],
                             embed_type=args["recsim_emb_type"],
                             embed_path=args["item_embedding_path"],
                             device=args["device"])
    user_emb = BaseEmbedding(num_embeddings=args["num_all_actions"],
                             dim_embed=args["recsim_num_categories"],
                             embed_type=args["recsim_emb_type"],
                             embed_path=args["user_embedding_path"],
                             device=args["device"])
    dict_embedding = {"item": item_emb, "user": user_emb}
    return dict_embedding


def launch_env(args: dict):
    """ Launch an env based on args """
    # import pudb; pudb.start()
    dict_embedding = prep_emb(args=args)
    dict_dfLog = {
        "offline": pd.read_csv(os.path.join(args["recsim_data_dir"], "offline_log.csv")),
        "online": pd.read_csv(os.path.join(args["recsim_data_dir"], "online_log.csv"))
    }
    print("Log data; offline: {} online: {}".format(dict_dfLog["offline"].shape, dict_dfLog["online"].shape))
    env = DatasetEnv(dict_embedding=dict_embedding, dict_dfLog=dict_dfLog, args=args)
    return env
