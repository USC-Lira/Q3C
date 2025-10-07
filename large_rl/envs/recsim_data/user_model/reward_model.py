import torch
import numpy as np

from large_rl.policy.arch.mlp import MLP, FILM_MLP
from large_rl.encoder.set_summariser import DeepSet, BiLSTM, Transformer as Trans
from large_rl.commons.args import SKIP_TOKEN
from large_rl.embedding.base import BaseEmbedding
from large_rl.commons.utils import min_max_scale_vec as min_max_scale

SCORE_MIN = 0.7


class UserModel(object):
    def __init__(self, item_embedding: BaseEmbedding, args: dict):
        self._args = args
        self._rng = np.random.RandomState(self._args["env_seed"])
        self.item_embedding = item_embedding
        self.reward_model = launch_RewardModel(args=self._args)

    def encode(self, obs):
        obs = self.reward_model.obs_encoder(obs)
        return obs

    def predict(self, obs: torch.tensor, action: np.ndarray):
        with torch.no_grad():
            emb = self.item_embedding.get(index=action, if_np=False)[:, 0, :]
            score = self.reward_model.compute_score(obs=obs, emb=emb)
            score = score.cpu().numpy()

        # a_user, reward = list(), list()
        # for i in range(self._args["num_envs"]):
        #     # get the metrics for a user
        #     _score = score[i]
        #     if _score > SCORE_MIN:  # If it's too high then rescale the predicted score
        #         _score = min_max_scale(_score, _min=SCORE_MIN, _max=1.0)
        #     if self._rng.random() < _score:
        #         _a_user_ind = action[i][0]
        #         reward.append(1)
        #     else:
        #         _a_user_ind = SKIP_TOKEN
        #         reward.append(0)
        #     a_user.append(_a_user_ind)
        # reward, a_user = np.asarray(reward), np.asarray(a_user)
        # print(reward)
        # print(a_user)

        # Optimised code
        score = score.flatten()
        mask = score > SCORE_MIN
        if any(mask):
            score[mask] = min_max_scale(_vec=score[mask], _min=SCORE_MIN, _max=1.0)
        mask = np.asarray(self._rng.random(size=self._args["num_envs"]) < score)
        reward = mask.astype(np.int)
        a_user = np.ones_like(reward) * SKIP_TOKEN
        a_user[mask] = action.flatten()[mask]
        return reward, a_user


class RewardModel(object):
    def __init__(self, args: dict):
        self._args = args
        self._device = self._args["device"]
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
        self.obs_encoder = _obs_enc
        self.sigmoid = torch.nn.Sigmoid()

        if self._args["recsim_rm_if_film"]:
            self.model = FILM_MLP(
                dim_in=self._args["recsim_dim_embed"], in_film=self._args["recsim_dim_embed"], dim_out=1,
                if_norm_each=True, if_norm_final=True, p_drop=self._args["recsim_rm_dropout"]
            ).to(device=self._args["device"])
        else:
            self.model = MLP(
                dim_in=self._args["recsim_dim_embed"] + self._args["recsim_dim_embed"], dim_out=1,
                if_norm_each=True, if_norm_final=True, p_drop=self._args["recsim_rm_dropout"]
            ).to(device=self._args["device"])

        # Load the pretrained state
        if self._args.get("rm_weight_path", None):
            print("RewardModel>> Loading from {}".format(self._args["rm_weight_path"]))
            _state = torch.load(self._args["rm_weight_path"], map_location=self._device)
            self.model.load_state_dict(_state["model"])
            self.obs_encoder.load_state_dict(_state["obs_encoder"])

    def compute_score(self, obs: torch.tensor, emb: torch.tensor) -> torch.tensor:
        state_embed = self.obs_encoder(obs)  # batch_size x dim_obs
        if self._args["recsim_rm_if_film"]:
            scores = self.model(state_embed, emb)  # batch_size x 1
        else:
            scores = self.model(torch.cat([state_embed, emb], dim=-1))  # batch_size x 1
        scores = self.sigmoid(scores)
        return scores

    def train(self):
        self.obs_encoder.train()
        self.model.train()

    def eval(self):
        self.obs_encoder.eval()
        self.model.eval()

    def set_if_check_grad(self, flg: bool):
        self._check_grad = flg

    def state_dict(self):
        return {"model": self.model.state_dict(), "obs_encoder": self.obs_encoder.state_dict(), }


def launch_RewardModel(args: dict):
    if args["recsim_reward_model_type"] == "normal":
        return RewardModel(args=args)
    else:
        raise ValueError
