from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces
from copy import deepcopy
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

from large_rl.envs.recsim import choice_model
from large_rl.envs.recsim import document
from large_rl.envs.recsim import user
from large_rl.envs.recsim import utils
from large_rl.envs.recsim.simulator import environment
from large_rl.envs.recsim.simulator import recsim_gym
from large_rl.envs.wrapper import wrap_env
from large_rl.envs.recsim.distribution import Distribution
from large_rl.envs.recsim.wrapper_multiuser import _worker_shared_memory, launch_env
from large_rl.commons.args import MIN_QUALITY_SCORE, MAX_QUALITY_SCORE, MAX_VIDEO_LENGTH
from large_rl.commons.utils import min_max_scale_vec, scale_number, softplus_np, softplus_math


class IEvUserModel(user.AbstractUserModel):
    def __init__(self,
                 slate_size,
                 choice_model_ctor=None,
                 response_model_ctor=None,
                 user_state_ctor=None,
                 no_click_mass=1.0,
                 args=None):
        super(IEvUserModel, self).__init__(
            response_model_ctor=response_model_ctor,
            user_sampler=IEvUserSampler(user_ctor=user_state_ctor, no_click_mass=no_click_mass, args=args),
            slate_size=slate_size, args=args)
        if choice_model_ctor is None: raise Exception('A choice model needs to be specified!')
        self.choice_model_ctor = choice_model_ctor
        self.choice_model = choice_model_ctor(self._user_state.choice_features)
        self._prev_rec = deque(maxlen=2)
        self._rng = np.random.RandomState(self._args["env_seed"])

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.budget <= 0

    def update_state(self, slate_documents, responses):
        user_state = self._user_state

        for doc, response in zip(slate_documents, responses):
            if not response.clicked:
                # Step penalty if no selection
                user_state.budget -= user_state.step_penalty

            if self._args["recsim_if_user_update_even_no_click"] or response.clicked:
                self.choice_model.score_documents(user_state, [doc])

                # scores is a list of length 1 since only one doc observation is set.
                expected_utility = self.choice_model.scores[0]

                if self._args["recsim_item_dist"].lower() != "sklearn-gmm":
                    if self._args["recsim_if_user_global_transition"]:
                        # mask = doc.task_embed
                        mask = 1.0
                    else:
                        mask = np.eye(self._args["recsim_dim_embed"])[doc.cluster_id]

                    # Step size should vary based on interest.
                    alpha = min_max_scale_vec(_vec=np.abs(user_state.user_interests), _min=0.0, _max=1.0)
                    update = alpha * mask * (doc.task_embed - user_state.user_interests)  # Update user vec towards item
                else:
                    # Step size should vary based on interest.
                    alpha = min_max_scale_vec(_vec=np.abs(user_state.user_interests), _min=0.0, _max=1.0)
                    update = alpha * (doc.task_embed - user_state.user_interests)  # Update user vec towards item

                # Relativeness score of item to user
                _score = np.dot(user_state.user_interests, doc.task_embed)
                positive_update_prob = (1 / (1 + np.exp(-_score)))  # sigmoid
                _eps = self._user_sampler._rng.rand(1)
                if _eps < positive_update_prob:  # Update user vec towards item
                    user_state.user_interests += update
                else:  # Move user vec away from item
                    user_state.user_interests -= update
                user_state.user_interests = np.clip(a=user_state.user_interests, a_min=-1.0, a_max=1.0)

                if self._args["recsim_budget_logic_type"] == "original":
                    # User consumes an item using own budget: content length is fixed
                    user_state.budget -= response.content_length

                    # User gets positive/negative experience so budget is further updated
                    """
                    - user_state.user_quality_factor: always 0.0
                    - expected_utility: item score computed by user model
                    - user_state.document_quality_factor: always one
                    - doc.quality: sampled from gaussian
                    """
                    received_utility = (user_state.user_quality_factor * expected_utility) + \
                                       (user_state.document_quality_factor * doc.quality)
                    user_state.budget += (user_state.user_update_alpha * response.content_length *
                                          received_utility)
                elif self._args["recsim_budget_logic_type"] == "new":
                    # consume item
                    user_state.budget -= response.content_length

                    # Compute the base utility perception of user on the item
                    # if _eps < positive_update_prob:
                    #     expected_utility = np.abs(expected_utility)
                    # else:
                    #     expected_utility = - np.abs(expected_utility)
                    score = (user_state.user_quality_factor * expected_utility)
                    score += (user_state.document_quality_factor * doc.quality)

                    # Update the budget with the score scaled into some predefined range
                    user_state.budget += (1 * np.tanh(score))
                elif self._args["recsim_budget_logic_type"] == "simple-original":
                    user_state.budget -= response.content_length
                    user_state.budget += (response.content_length * user_state.user_quality_factor * _score)

    def simulate_response(self, documents):
        # List of empty responses
        responses = [self._response_model_ctor() for _ in documents]

        # Sample some clicked responses using user's choice model and populate responses.
        doc_obs = [doc for doc in documents]

        if self._args["recsim_if_novelty_bonus"]:  # not used anymore
            if len(self._prev_rec) > 1:
                _history_vec = np.mean(self._prev_rec, axis=0)
                # negated dot product
                _d = - np.dot(doc_obs[0].create_observation(
                    not self._args["recsim_if_switch_act_task_emb"]), _history_vec)
                _d = (1 / (1 + np.exp(- _d)))

                # # inv of euc-dist
                # _d = np.linalg.norm(_history_vec - doc_obs[0].create_observation())
                # _d = 1 / _d if _d > 1.0 else 0.9999

                self._novelty_bonus = 1 + _d
            else:
                self._novelty_bonus = None
            self._prev_rec.append(doc_obs[0].create_observation(not self._args["recsim_if_switch_act_task_emb"]))
        else:
            self._novelty_bonus = None
        self.choice_model.score_documents(self._user_state, doc_obs)
        selected_index = self.choice_model.choose_item(novelty_bonus=self._novelty_bonus)

        for i, response in enumerate(responses):
            response.quality = documents[i].quality
            response.cluster_id = documents[i].cluster_id

        if selected_index is None:
            return responses
        self._generate_click_response(documents[selected_index], responses[selected_index])
        return responses

    def _generate_click_response(self, doc, response):
        user_state = self._user_state
        response.clicked = True
        response.doc_id = doc.doc_id()
        response.content_length = min(user_state.budget, doc.content_length)

    def reset(self):
        self._user_state = self._user_sampler.sample_user()
        self._prev_rec = deque(maxlen=2)


class IEvUserState(user.AbstractUserState):
    """Class to represent interest evolution users."""

    def __init__(self,
                 user_interests,
                 cluster_id,
                 budget=None,
                 score_scaling=None,
                 attention_prob=None,
                 no_click_mass=None,
                 min_doc_utility=None,
                 user_update_alpha=None,
                 step_penalty=None,
                 user_quality_factor=None,
                 document_quality_factor=None,
                 args=None):
        """Initializes a new user."""

        super(IEvUserState, self).__init__(args=args)
        self.user_interests = user_interests
        self.budget = budget
        self.min_doc_utility = min_doc_utility

        # Convenience wrapper
        self.choice_features = {
            'score_scaling': score_scaling,
            'attention_prob': attention_prob,
            'no_click_mass': no_click_mass,
            'min_normalizer': 0.0,  # min_normalizer,
            'args': self._args,
        }
        self.user_update_alpha = user_update_alpha
        self.step_penalty = step_penalty
        self.user_quality_factor = user_quality_factor
        self.document_quality_factor = document_quality_factor
        self.cluster_id = cluster_id

    def score_document(self, doc_obs):
        if not doc_obs.if_valid:
            return 0.0
        doc_obs = doc_obs.create_observation(if_task_embed=not self._args["recsim_if_switch_act_task_emb"])
        if self.user_interests.shape != doc_obs.shape:
            raise ValueError('User and document feature dimension mismatch!')
        if self._args["recsim_type_user_utility_computation"] == "dot_prod":
            _d = np.dot(self.user_interests, doc_obs)
            # Heavy-tailed softplus activation fn
            if _d < 0.0:
                _d -= 1.5
            _d = softplus_np(_d)
            return _d
        elif self._args["recsim_type_user_utility_computation"] == "dp-ec":
            _d = np.dot(self.user_interests, doc_obs)
            # Heavy-tailed softplus activation fn
            if _d < 0.0:
                _d -= 1.5
            _d = softplus_np(_d)

            __d = np.linalg.norm(self.user_interests - doc_obs)
            if __d == 0.0:
                __d = 1 / 0.0001
            else:
                __d = 1 / __d
            return _d + __d
        elif self._args["recsim_type_user_utility_computation"] == "cosine":
            _d = cosine_similarity(self.user_interests[None, :], doc_obs[None, :]).flatten()[0]
            _d = scale_number(x=_d, to_min=0.0, to_max=2.0, from_min=-1.0, from_max=1.0)  # Convert the range
            return _d
        elif self._args["recsim_type_user_utility_computation"] == "euc_dist":
            _d = np.linalg.norm(self.user_interests - doc_obs)
            if _d == 0.0:
                return 1 / 0.0001
            else:
                return 1 / _d
        elif self._args["recsim_type_user_utility_computation"] == "scaled_euc_dist":
            _d = np.linalg.norm(self.user_interests - doc_obs)
            _d = scale_number(x=_d, to_min=0.0, to_max=1.0, from_min=0.0, from_max=self._args["_max_l2_dist"])
            return 1 - _d  # Inverse
        else:
            raise ValueError

    def create_observation(self, **kwargs):
        """Return an observation of this user's observable state."""

        if self._args["recsim_if_noisy_obs"]:
            noise = kwargs["rng"].uniform(low=-0.3, high=0.3, size=self._args["recsim_dim_embed"])
            # noise = kwargs["rng"].uniform(low=-0.6, high=0.6, size=self._args["recsim_dim_embed"])
            return self.user_interests + noise
        else:
            return self.user_interests

    def observation_space(self):
        return spaces.Box(shape=(self._args["recsim_dim_embed"],), dtype=np.float32, low=-1.0, high=1.0)


class IEvUserSampler(user.AbstractUserSampler):
    """Class that samples users for utility model experiment."""

    def __init__(self,
                 user_ctor=IEvUserState,
                 document_quality_factor=1.0,
                 no_click_mass=1.0,
                 min_normalizer=-1.0,
                 **kwargs):
        """Creates a new user state sampler."""
        self._no_click_mass = no_click_mass
        self._min_normalizer = min_normalizer
        self._document_quality_factor = document_quality_factor
        self._seed = kwargs["args"]["env_seed"]  # checked!
        super(IEvUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        features = {}
        features['user_interests'], features['cluster_id'] = Distribution.sample_user_interest(rng=self._rng,
                                                                                               args=self._args)
        features['budget'] = self._args["recsim_user_budget"]
        features['no_click_mass'] = self._no_click_mass
        features['step_penalty'] = self._args["recsim_step_penalty"]
        features['score_scaling'] = 0.05  # for cascading choice models
        features['attention_prob'] = 0.65  # for cascading choice models
        features['user_quality_factor'] = 0.0
        # features['user_quality_factor'] = self._rng.uniform(low=0.0, high=1.0)
        features['document_quality_factor'] = 1.0
        # features['document_quality_factor'] = self._rng.uniform(low=0.0, high=1.0)
        features["args"] = self._args
        # features['user_update_alpha'] = 0.9 * (1.0 / 3.4)  # RecSim's hardcoded default value
        features['user_update_alpha'] = self._rng.uniform(low=0.0, high=1.0)
        return self._user_ctor(**features)


class IEvResponse(user.AbstractResponse):

    def __init__(self,
                 clicked=False,
                 content_length=0.0,
                 quality=0.0,
                 cluster_id=-1,
                 doc_id=-1,  # -1 means Skip!
                 args=None):
        super(IEvResponse, self).__init__(args=args)
        self.clicked = clicked
        self.content_length = content_length
        self.quality = quality
        self.cluster_id = cluster_id
        self.doc_id = doc_id

    def create_observation(self):
        return {
            'click': int(self.clicked),
            'content_length': np.array(self.content_length),
            'quality': np.array(self.quality),
            'cluster_id': int(self.cluster_id),
            'clicked_doc_id': int(self.doc_id),
        }

    @classmethod
    def response_space(cls):
        return spaces.Dict({
            'click': spaces.Discrete(2),
            'content_length': spaces.Box(low=0.0, high=MAX_VIDEO_LENGTH, shape=tuple(), dtype=np.float32),
            'quality': spaces.Box(low=MIN_QUALITY_SCORE, high=MAX_QUALITY_SCORE, shape=tuple(), dtype=np.float32),
            'cluster_id': spaces.Discrete(cls._args["num_categories"])
        })


class IEvItem(document.AbstractDocument):

    def __init__(self,
                 doc_id,
                 task_embed,
                 item_embed,
                 cluster_id=None,
                 content_length=None,
                 quality=None):
        """Generates a random set of features for this interest evolution Video."""

        # Document features (i.e. distribution over topics)
        self.task_embed = task_embed  # Not visible to agent and only used internally in RecSim
        self.item_embed = item_embed  # Visible to Agent

        # Cluster ID
        self.cluster_id = cluster_id

        # Length of video
        self.content_length = content_length

        # Document quality (i.e. trashiness/nutritiousness)
        self.quality = quality

        self._if_valid = True

        # doc_id is an integer representing the unique ID of this document
        super(IEvItem, self).__init__(doc_id)

    def create_observation(self, if_task_embed: bool = False):
        """Returns observable properties of this document as a float array."""
        if if_task_embed:
            return self.task_embed
        else:
            return self.item_embed

    @classmethod
    def observation_space(cls):
        return spaces.Box(shape=(cls._args["recsim_dim_embed"],), dtype=np.float32, low=-1.0, high=1.0)

    def set_valid_flg(self, if_valid):
        self._if_valid = if_valid

    @property
    def if_valid(self):
        return self._if_valid


class IEvItemSampler(document.AbstractDocumentSampler):
    """Class that samples videos for utility model experiment."""

    def __init__(self,
                 doc_ctor=IEvItem,
                 min_utility=-3.0,
                 max_utility=3.0,
                 **kwargs):
        super(IEvItemSampler, self).__init__(doc_ctor, **kwargs)
        self._num_clusters = self._args["recsim_num_categories"]
        self._min_utility = min_utility
        self._max_utility = max_utility
        num_trashy, num_nutritious = int(self._num_clusters * 0.7), int(self._num_clusters * 0.3)
        num_trashy = num_trashy + int((num_trashy + num_nutritious) != self._num_clusters)  # Minor adjust

        trashy = np.linspace(self._min_utility, 0, num_trashy)
        nutritious = np.linspace(0, self._max_utility, num_nutritious)
        self.cluster_means = np.concatenate((trashy, nutritious))

    def sample_document(self, doc_id: int, cluster_id: int, item_embed: np.ndarray, task_embed: np.ndarray,
                        content_length: float = None) -> IEvItem:
        doc_features = {}
        doc_features['doc_id'] = doc_id
        doc_features['cluster_id'] = cluster_id
        doc_features['task_embed'] = task_embed
        doc_features['item_embed'] = item_embed
        # doc_features['content_length'] = self._rng.uniform(low=1.0, high=4.0)
        doc_features['content_length'] = 4.0
        # Original implementation
        doc_features['quality'] = self._rng.normal(loc=self.cluster_means[cluster_id], scale=0.1, size=1)[0]
        # doc_features['quality'] = self._rng.uniform(low=-1.0, high=1.0)
        # doc_features['quality'] = self._rng.uniform(low=-3.0, high=3.0)
        return self._doc_ctor(**doc_features)


def total_clicks_reward(responses):
    """Calculates the total number of clicks from a list of responses.

    Args:
       responses: A list of IEvResponse objects

    Returns:
      reward: A float representing the total clicks from the responses
    """
    reward = 0.0
    for r in responses:
        reward += r.clicked
    return reward


def launch_choice_model(_name: str):
    _name = _name.lower()
    if _name == "deterministic":
        return choice_model.DeterministicChoiceModel
    elif _name == "multinomial":
        return choice_model.MultinomialProportionalChoiceModel
        # return choice_model.MultinomialLogitChoiceModel
    elif _name in ["linear", "tree"]:
        return choice_model.ComplexChoiceModel
    else:
        raise ValueError


def create_environment(args: dict, env_id):
    """Creates an interest evolution environment."""
    _args = deepcopy(args)
    _args["env_seed"] += env_id

    user_model = IEvUserModel(slate_size=args['recsim_slate_size'],
                              choice_model_ctor=launch_choice_model(_name=args["recsim_choice_model_type"]),
                              no_click_mass=args["recsim_no_click_mass"],
                              response_model_ctor=IEvResponse,
                              user_state_ctor=IEvUserState,
                              args=_args)

    ievenv = environment.SingleUserEnvironment(user_model=user_model,
                                               slate_size=args['recsim_slate_size'],
                                               resample_documents=False,
                                               # Note: this changes the item embedding at each ts
                                               args=_args)

    env = recsim_gym.RecSimGymEnv(raw_environment=ievenv,
                                  reward_aggregator=total_clicks_reward,
                                  metrics_aggregator=utils.aggregate_video_cluster_metrics,
                                  metrics_writer=utils.write_video_cluster_metrics)
    env = wrap_env(env=env, args=_args)
    return env


def create_vector_environment(args: dict):
    if args["recsim_sklearnGMM_if_sparse_centroids"]:
        args["sklearn_gmm_centroids"] = np.eye(args["recsim_num_categories"])
        args["sklearn_gmm_centroids"][args["recsim_num_categories"] // 2:] *= -1
    else:
        rng = np.random.RandomState(args["env_seed"])
        args["sklearn_gmm_centroids"] = rng.uniform(low=-1.0, high=1.0,
                                                    size=(args["recsim_num_categories"], args["recsim_dim_embed"]))

    document_sampler = IEvItemSampler(doc_ctor=IEvItem, seed=args['env_seed'], args=args)
    env_cls = launch_env(if_async=args["if_async"])

    def _fn(_env_id):
        return create_environment(args=args, env_id=_env_id)

    env = env_cls(
        env_fns=[partial(_fn, _env_id=env_id) for env_id in range(args["num_envs"])],
        worker=_worker_shared_memory, document_sampler=document_sampler, args=args, syncEnv_if_respawn=True
    )
    env.reset()
    return env
