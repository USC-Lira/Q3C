import torch
import numpy as np
from typing import Dict

from large_rl.embedding.base import BaseEmbedding
from large_rl.commons.args import HISTORY_SIZE, SKIP_TOKEN


class ObservationFactory(object):
    def __init__(self, batch_size: int, user_id: np.ndarray = None, if_mdp: bool = False):
        self._batch_size = batch_size
        self._user_id = user_id if user_id is not None else np.ones(self._batch_size) * SKIP_TOKEN
        # FIFO queue; batch_step_size x history_size
        self._history_seq = np.ones((self._batch_size, HISTORY_SIZE)) * SKIP_TOKEN

        # for MDP setting in RecSim
        self._if_mdp = if_mdp
        self._state = None

    @property
    def shape(self):
        if self._if_mdp:
            return self._state.shape
        else:
            return self._history_seq.shape

    def update_history_seq(self, a_user: np.ndarray, active_user_mask: np.ndarray = None):
        # Update happens only on the clicked items
        _mask = (a_user != SKIP_TOKEN)  # (batch_size)-size array

        # FIFO update; batch_step_size x history_size
        self._history_seq[_mask] = np.concatenate([self._history_seq[_mask, 1:], a_user[_mask, None]], axis=1)
        if active_user_mask is not None:
            self._history_seq = self._history_seq[active_user_mask]  # Remove the obs of inactive users

    def load_history_seq(self, history_seq: np.ndarray):
        self._history_seq = history_seq  # batch_size x history_size

    def make_obs(self, dict_embedding: Dict[str, BaseEmbedding], if_separate: bool = False, device: str = "cpu"):
        """ Make an observation by concatenating the user attributes and its sequence of clicked items' embedding

        Args:
            dict_embedding (dict):
            if_separate (bool): binary flag changing the format of output either np.ndarray or dict
            device (str): cpu or cuda

        Returns:
            self._if_mdp:
                state (np.ndarray): batch_size x dim_user
            else:
                if_separate:
                    _dict (dict):
                        user_feat (np.ndarray): batch_size x history_size x dim_user
                        history_seq (np.ndarray): batch_size x history_size x dim_item
                else:
                    obs (np.ndarray): batch_step_size x history_size x (dim_user + dim_item)
        """
        if not self._if_mdp:
            history_seq = torch.tensor(self._history_seq, dtype=torch.int64, device=device)
            hist_feat = dict_embedding["item"].get(index=history_seq, if_np=False)

            if all(self._user_id != SKIP_TOKEN):
                user_id = torch.tensor(self._user_id, dtype=torch.int64, device=device)
                user_feat = dict_embedding["user"].get(index=user_id, if_np=True)[:, None, :]
                user_feat = np.tile(A=user_feat, reps=(1, self._history_seq.shape[1], 1))
                user_feat = torch.tensor(user_feat, device=device)
                if if_separate:
                    return {"user_feat": user_feat, "history_seq": hist_feat}
                else:
                    return torch.cat([user_feat, hist_feat], dim=-1)
            return hist_feat
        else:
            return self._state

    def __getitem__(self, item):
        """ This is for ReplayBuffer """
        if self._if_mdp:
            return self._state[item, :]
        else:
            return {"user_id": self._user_id[item], "history_seq": self._history_seq[item, :]}
