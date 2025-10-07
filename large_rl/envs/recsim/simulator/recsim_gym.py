"""A wrapper for using Gym environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import collections
import numpy as np
from gym import spaces
from large_rl.envs.recsim.simulator import environment


def _dummy_metrics_aggregator(responses, metrics, info):
    del responses  # Unused.
    del metrics  # Unused.
    del info  # Unused.
    return


def _dummy_metrics_writer(metrics, add_summary_fn):
    del metrics  # Unused.
    del add_summary_fn  # Unused.
    return


class RecSimGymEnv(gym.Env):
    """Class to wrap recommender system environment to gym.Env.

    Attributes:
      game_over: A boolean indicating whether the current game has finished
      action_space: A gym.spaces object that specifies the space for possible
        actions.
      observation_space: A gym.spaces object that specifies the space for possible
        observations.
    """

    def __init__(self,
                 raw_environment,
                 reward_aggregator,
                 metrics_aggregator=_dummy_metrics_aggregator,
                 metrics_writer=_dummy_metrics_writer):
        """Initializes a RecSim environment conforming to gym.Env.

        Args:
          raw_environment: A recsim recommender system environment.
          reward_aggregator: A function mapping a list of responses to a number.
          metrics_aggregator: A function aggregating metrics over all steps given
            responses and response_names.
          metrics_writer:  A function writing final metrics to TensorBoard.
        """
        self._environment = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer

    @property
    def environment(self):
        """Returns the recsim recommender system environment."""
        return self._environment

    @property
    def game_over(self):
        return False

    @property
    def action_space(self):
        """Returns the action space of the environment.

        Each action is a vector that specified document slate. Each element in the
        vector corresponds to the index of the document in the candidate set.
        """
        action_space = spaces.MultiDiscrete(
            self._environment._args["num_all_actions"] * np.ones(
                (self._environment.slate_size,)
            ))
        return action_space

    @property
    def observation_space(self):
        """Returns the observation space of the environment.

        Each observation is a dictionary with three keys `user`, `doc` and
        `response` that includes observation about user state, document and user
        response, respectively.
        """
        return self._environment.user_model.observation_space()

    def step(self, action):
        """Runs one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, done, info).

        Args:
          action (object): An action provided by the environment

        Returns:
          A four-tuple of (observation, reward, done, info) where:
            observation (object): agent's observation that include
              1. User's state features
              2. Document's observation
              3. Observation about user's slate responses.
            reward (float) : The amount of reward returned after previous action
            done (boolean): Whether the episode has ended, in which case further
              step() calls will return undefined results
            info (dict): Contains responses for the full slate for
              debugging/learning.
        """
        user_obs, doc_obs, responses, done = self._environment.step(action)
        all_responses = tuple(response.create_observation() for response in responses)
        reward = self._reward_aggregator(responses)
        info_dict = self.extract_env_info()
        info_dict.update(all_responses[0])
        return user_obs, reward, done, info_dict

    def reset(self):
        user_obs, doc_obs = self._environment.reset()
        return user_obs

    def reset_sampler(self):
        self._environment.reset_sampler()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    def extract_env_info(self):
        info = {
            # 'env': self._environment
        }
        return info
