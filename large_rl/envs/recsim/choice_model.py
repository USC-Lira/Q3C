from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import torch
import numpy as np

# === Ad hoc solution
_MAT = np.asarray([
    [1.0, 1.0],  # top right corner
    [-1.0, -1.0],  # bottom left corner
    [1.0, -1.0],  # bottom right corner
    [-1.0, 1.0],  # top left corner
])

ITEM2CATEGORY = {0: (0, _MAT[0]), 1: (0, _MAT[1]), 2: (1, _MAT[2]), 3: (1, _MAT[3])}


def user2category(_embed):
    assert _embed.shape[0] == 2
    # _dist = [np.sqrt(np.square(_embed - _vec).sum()) for _vec in _MAT]  # this is okay too
    _embed = np.tile(A=_embed[None, :], reps=(4, 1))
    _dist = np.linalg.norm(x=_embed - _MAT, axis=-1)
    return ITEM2CATEGORY[int(np.argmin(_dist))][0]


def softmax(vector):
    """Computes the softmax of a vector."""
    normalized_vector = np.array(vector) - np.max(vector)  # For numerical stability
    return np.exp(normalized_vector) / np.sum(np.exp(normalized_vector))


@six.add_metaclass(abc.ABCMeta)
class AbstractChoiceModel(object):
    """Abstract class to represent the user choice model.

    Each user has a choice model.
    """
    _scores = None
    _score_no_click = None

    def __init__(self, choice_features):
        self._args = choice_features["args"]
        self._seed = self._args["env_seed"]
        self._rng = np.random.RandomState(self._args["env_seed"])

    @abc.abstractmethod
    def score_documents(self, user_state, doc_obs):
        """Computes unnormalized scores of documents in the slate given user state.

        Args:
          user_state: An instance of AbstractUserState.
          doc_obs: A numpy array that represents the observation of all documents in
            the slate.
        Attributes:
          scores: A numpy array that stores the scores of all documents.
          score_no_click: A float that represents the score for the action of
            picking no document.
        """

    @property
    def scores(self):
        return self._scores

    @property
    def score_no_click(self):
        return self._score_no_click

    @abc.abstractmethod
    def choose_item(self, **kwargs):
        """Returns selected index of document in the slate.

        Returns:
          selected_index: a integer indicating which item was chosen, or None if
            none were selected.
        """


class NormalizableChoiceModel(AbstractChoiceModel):
    """A normalizable choice model."""

    @staticmethod
    def _score_documents_helper(user_state, doc_obs):
        scores = np.array([])
        for doc in doc_obs:
            scores = np.append(scores, user_state.score_document(doc))
        return scores

    def choose_item(self, **kwargs):
        if kwargs["novelty_bonus"] is not None:
            self._scores *= kwargs["novelty_bonus"]
        all_scores = np.append(self._scores, self._score_no_click)
        all_probs = all_scores / np.sum(all_scores)
        selected_index = self._rng.choice(len(all_probs), p=all_probs)
        if selected_index == len(all_probs) - 1:
            selected_index = None
        return selected_index


class MultinomialLogitChoiceModel(NormalizableChoiceModel):

    def __init__(self, choice_features):
        super(MultinomialLogitChoiceModel, self).__init__(choice_features=choice_features)
        self._no_click_mass = choice_features.get('no_click_mass', -float('Inf'))

    def score_documents(self, user_state, doc_obs):
        logits = self._score_documents_helper(user_state, doc_obs)
        logits = np.append(logits, self._no_click_mass)
        # Use softmax scores instead of exponential scores to avoid overflow.
        all_scores = softmax(logits)
        self._scores = all_scores[:-1]
        self._score_no_click = all_scores[-1]


class MultinomialProportionalChoiceModel(NormalizableChoiceModel):

    def __init__(self, choice_features):
        super(MultinomialProportionalChoiceModel, self).__init__(choice_features=choice_features)
        self._min_normalizer = choice_features.get('min_normalizer')
        self._no_click_mass = choice_features.get('no_click_mass', 0)

    def score_documents(self, user_state, doc_obs):
        scores = self._score_documents_helper(user_state, doc_obs)
        all_scores = np.append(scores, self._no_click_mass)
        all_scores = all_scores - self._min_normalizer
        assert all_scores[all_scores < 0.0].size == 0, 'Normalized scores have non-positive elements.'
        self._scores = all_scores[:-1]
        self._score_no_click = all_scores[-1]


class CascadeChoiceModel(NormalizableChoiceModel):

    def __init__(self, choice_features):
        super(CascadeChoiceModel, self).__init__(choice_features=choice_features)
        self._attention_prob = choice_features.get('attention_prob', 1.0)
        self._score_scaling = choice_features.get('score_scaling')
        if self._attention_prob < 0.0 or self._attention_prob > 1.0:
            raise ValueError('attention_prob must be in [0,1].')
        if self._score_scaling < 0.0:
            raise ValueError('score_scaling must be positive.')

    def _positional_normalization(self, scores):
        self._score_no_click = 1.0
        for i in range(len(scores)):
            s = self._score_scaling * scores[i]
            assert s <= 1.0, ('score_scaling cannot convert score %f into a '
                              'probability') % scores[i]
            scores[i] = self._score_no_click * self._attention_prob * s
            self._score_no_click *= (1.0 - self._attention_prob * s)
        self._scores = scores


class ExponentialCascadeChoiceModel(CascadeChoiceModel):

    def score_documents(self, user_state, doc_obs):
        scores = self._score_documents_helper(user_state, doc_obs)
        scores = np.exp(scores)
        self._positional_normalization(scores)


class ProportionalCascadeChoiceModel(CascadeChoiceModel):

    def __init__(self, choice_features):
        super(ProportionalCascadeChoiceModel, self).__init__(choice_features=choice_features)
        self._min_normalizer = choice_features.get('min_normalizer')
        super(ProportionalCascadeChoiceModel, self).__init__(choice_features)

    def score_documents(self, user_state, doc_obs):
        scores = self._score_documents_helper(user_state, doc_obs)
        scores = scores - self._min_normalizer
        assert not scores[scores < 0.0], 'Normalized scores have non-positive elements.'
        self._positional_normalization(scores)


class DeterministicChoiceModel(AbstractChoiceModel):
    """ DeterministicChoiceModel """

    def score_documents(self, user_state, doc_obs):
        self._user_state = user_state
        self._doc_obs = doc_obs
        self._scores = [1.0]  # for external modules

    def choose_item(self, **kwargs):
        if self._args["recsim_item_dist"].lower() == "sklearn-gmm":
            raise ValueError("user's cluster_id is static but user vector is moving around!")
            _topk = 1
            gt_category, action_category = [self._user_state.cluster_id], [self._doc_obs[0].cluster_id]
        else:
            if self._args["recsim_item_dist"].lower() == "two-modal-gmm":
                gt_category = [user2category(_embed=self._user_state.user_interests)]
            else:
                _topk = 1  # self._args["recsim_user_num_modality"]
                gt_category = np.argpartition(self._user_state.user_interests, -_topk)[-_topk:].tolist()
            action_category = [np.argmax(np.asarray(self._doc_obs[0].task_embed), axis=-1)]

        # This is to support the listwise actions
        for _id, c in enumerate(action_category):
            if c in gt_category:  # check if the recommended category matches the user category
                return _id
        return None


# class ComplexChoiceModel(NormalizableChoiceModel):
#     def __init__(self, choice_features):
#         super(ComplexChoiceModel, self).__init__(choice_features=choice_features)
#         self._min_normalizer = choice_features.get('min_normalizer')
#         self._no_click_mass = choice_features.get('no_click_mass', 0)
#
#         _dim_in = self._args["recsim_dim_embed"] * 2
#         if self._args["recsim_choice_model_type"] == "linear":
#             # Note: this must not be on gpu since item and uer vecs are small
#             self._model = torch.nn.Sequential(
#                 torch.nn.Linear(_dim_in, 16),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(16, 1),
#                 torch.nn.Sigmoid(),
#             ).to(device="cpu")
#             # self._model.state_dict(torch.load(f"{ROOT_DIR}/large_rl/envs/recsim/weights/mlp.pt"))
#         elif self._args["recsim_choice_model_type"] == "tree":
#             from sklearn.tree import DecisionTreeClassifier as clf
#             from sklearn.datasets import make_classification
#             X, y = make_classification(n_samples=1000,
#                                        weights=[0.8, 0.2],  # [skip-prob, click-prob] in the label set
#                                        n_features=self._args["recsim_dim_embed"],
#                                        random_state=self._seed)
#             self._model = clf(max_depth=5, random_state=self._seed)
#             self._model.fit(X, y)
#
#     def score_documents(self, user_state, doc_obs):
#         _item = np.asarray(doc_obs[0].create_observation(if_task_embed=True)).astype(np.float32)
#         _user = user_state.user_interests.astype(np.float32)
#         _in = np.concatenate([_item, _user], axis=-1)
#         if self._args["recsim_choice_model_type"] == "linear":
#             with torch.no_grad():
#                 scores = self._model(torch.tensor(_in)).numpy()
#                 item_score = scores[0]
#                 skip_score = 1 - item_score
#         elif self._args["recsim_choice_model_type"] == "tree":
#             # scores = self._model.predict(_in.reshape(1, -1))[0]
#             scores = self._model.predict_proba(_item.reshape(1, -1))[0]
#             skip_score = scores[0]
#             item_score = scores[1]  # index 1 for click prob
#         else:
#             raise ValueError
#         self._scores = [item_score]
#         # self._score_no_click = self._no_click_mass
#         self._score_no_click = skip_score
