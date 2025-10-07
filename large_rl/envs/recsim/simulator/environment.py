from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class AbstractEnvironment(object):
    """Abstract class representing the recommender system environment.

    Attributes:
      user_model: An list or single instantiation of AbstractUserModel
        representing the user/users.
      slate_size: An integer representing the slate size.
      candidate_set: An instantiation of CandidateSet.
      num_clusters: An integer representing the number of document clusters.
    """

    def __init__(self,
                 user_model,
                 slate_size,
                 resample_documents=True,
                 args: dict = None):
        self._user_model = user_model
        self._slate_size = slate_size
        self._resample_documents = resample_documents
        self._args = args

    @abc.abstractmethod
    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """

    @abc.abstractmethod
    def reset_sampler(self):
        """Resets the relevant samplers of documents and user/users."""

    @property
    def slate_size(self):
        return self._slate_size

    @property
    def user_model(self):
        return self._user_model

    @abc.abstractmethod
    def step(self, slate):
        """Executes the action, returns next state observation and reward.

        Args:
          slate: An integer array of size slate_size (or list of such arrays), where
          each element is an index into the set of current_documents presented.

        Returns:
          user_obs: A gym observation representing the user's next state
          doc_obs: A list of observations of the documents
          responses: A list of AbstractResponse objects for each item in the slate
          done: A boolean indicating whether the episode has terminated
        """


class SingleUserEnvironment(AbstractEnvironment):

    def reset(self):
        self._user_model.reset()
        user_obs = self._user_model.create_observation()
        return (user_obs, None)

    def reset_sampler(self):
        self._user_model.reset_sampler()

    def step(self, documents):
        # If the user has been dead already, then do nothing
        if not self._user_model.is_terminal():
            # Simulate the user's response
            responses = self._user_model.simulate_response(documents)

            # Update the user's state.
            self._user_model.update_state(documents, responses)
        else:
            # List of empty responses
            responses = self._user_model.create_empty_response(self._args["recsim_slate_size"])

        # Obtain next user state observation.
        user_obs = self._user_model.create_observation()

        # Check if reaches a terminal state and return after executing action
        done = self._user_model.is_terminal()

        return (user_obs, None, responses, done)
