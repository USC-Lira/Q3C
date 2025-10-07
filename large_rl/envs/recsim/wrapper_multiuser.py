import torch
import sys
import numpy as np

from tqdm import tqdm
from typing import List
from copy import deepcopy
from gym.vector.utils import concatenate
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.async_vector_env import write_to_shared_memory, AsyncVectorEnv

from large_rl.commons.utils import logging
from large_rl.envs.recsim.document import AbstractDocument, CandidateSet, AbstractDocumentSampler
from large_rl.envs.recsim.distribution import Distribution
from large_rl.commons.pt_activation_fns import ACTIVATION_FN, ff

RESAMPLING_FREQ = ["ts", "ep"]
NUM_THREADS = 100000


def non_lin_transform(_embed: np.ndarray, args: dict):
    if args["recsim_act_emb_if_nonLin_transform"]:
        _embed = ACTIVATION_FN[args["recsim_act_emb_nonLin_transform"]]()(torch.tensor(_embed)).numpy()
        # _embed = _embed ** 2
        # _embed = ff(xx=_embed)
    return _embed


class MultiUserBaseWrapper(object):
    def _prep(self, document_sampler: AbstractDocumentSampler, args: dict):
        self._args = args
        self._if_eval = False
        self._candidate_set = CandidateSet(args=args)
        self._dist = Distribution(args=args)
        self._document_sampler = document_sampler
        self._rng = np.random.RandomState(args["env_seed"])
        self._prep_items()

        if self._args["recsim_if_valid_box"]:
            self._set_valid_boxes()

    @property
    def args(self):
        return self._args

    @property
    def candidate_set(self):
        return self._candidate_set

    def train(self):
        self._if_eval = False

    def eval(self):
        self._if_eval = True

    def _prep_items(self):
        """
        - Item-vec(e_i): item embedding visible to agent
          - Sim: Noisy Category-based GMM
          - Data: Pretrained on Amazon Product
        - Task-vec(e_t): item embedding invisible to agent since this is user's perception of items
          - Sim: Category-based GMM w/o Noise
          - Data: Category-based GMM w/o Noise
        """
        logging("=== Populate items: start ===")

        _shape = (self._args["num_all_actions"], self._args["recsim_dim_embed"])
        if self._args["recsim_act_emb_lin_scale"] == 1.0:
            _scales = np.ones(_shape)
        else:
            _scales = self._rng.randn(*_shape) * self._args["recsim_act_emb_lin_scale"]
        _shifts = self._rng.randn(*_shape) * self._args["recsim_act_emb_lin_shift"]

        # Make sure that all categories are included in Item-set
        _category_list = list(range(self._args["recsim_num_categories"]))

        for _doc_id in tqdm(self._candidate_set.items_list, total=len(self._candidate_set.items_list)):
            if len(_category_list) > 0:
                _cluster_id = _category_list.pop(0)
            else:
                _cluster_id = None
            _embed, _cluster_id = self._dist.sample_item_embed_and_category(item_id=_doc_id,
                                                                            category_id=_cluster_id)
            _task_embed = _embed.copy()
            _item_embed = np.multiply(_embed, _scales[_doc_id]) + _shifts[_doc_id]  # linear scaling
            _item_embed = non_lin_transform(_embed=_item_embed, args=self._args)
            _doc = self._document_sampler.sample_document(
                doc_id=_doc_id, item_embed=_item_embed, task_embed=_task_embed, cluster_id=_cluster_id
            )
            self._candidate_set.add_document(document=_doc)

        logging("=== Populate items: end ===")

    def execute(self, fn_name: str, args: list = None):
        raise NotImplementedError

    @property
    def act_embedding(self):
        _embed = self._candidate_set.create_observation_np(
            if_task_embed=self._args["recsim_if_switch_act_task_emb"])

        from sklearn.preprocessing import MinMaxScaler  # this is to clearly constraint the action space after noise
        if self._args["DEBUG_size_action_space"] == "small":
            _embed = MinMaxScaler(feature_range=(0, 1)).fit_transform(_embed)
        elif self._args["DEBUG_size_action_space"] == "large":
            _embed = MinMaxScaler(feature_range=(-1, 1)).fit_transform(_embed)
        elif self._args["DEBUG_size_action_space"] == "unbounded":
            pass  # do nothing!
        return _embed

    @property
    def task_embedding(self):
        return self._candidate_set.create_observation_np(if_task_embed=not self._args["recsim_if_switch_act_task_emb"])

    def slate_to_doc_cls(self, slates) -> List[AbstractDocument]:
        return self._slate_to_doc_cls(slates=slates)

    def _slate_to_doc_cls(self, slates) -> List[AbstractDocument]:
        return [self._candidate_set.get_documents(document_ids=slate) for slate in slates]

    def get_img_paths(self, slates: np.ndarray):
        docs = self._slate_to_doc_cls(slates=slates)
        res = [[docs[j][i].img_path for i in range(slates.shape[1])] for j in range(slates.shape[0])]
        return res

    def check_candidate_set(self, candidate_set: np.ndarray) -> np.ndarray:
        res = list()
        for _id in range(self.observations.shape[0]):
            # Same logic as in interest_evolution_generic.py
            _d = np.linalg.norm(self.observations[_id] - candidate_set[_id], axis=-1)
            _d[_d != 0.0] = 1 / _d[_d != 0.0]
            _d[_d == 0.0] = 1 / 0.0001
            res.append(_d.mean())
        return np.asarray(res)

    def _set_valid_boxes(self, box_size=4.0, num_boxes=100):  # same as mujoco_env
        _rng = np.random.RandomState(self._args["env_seed"])
        self.box_centers = _rng.uniform(low=-1.0, high=1.0, size=[num_boxes, self.act_embedding.shape[-1]])
        self.box_size = box_size

    def _check_valid_action(self, action):
        if_valid = True
        if self._args["recsim_if_valid_box"]:
            emb = action[0].create_observation(if_task_embed=not self._args["recsim_if_switch_act_task_emb"])
            _d = np.sqrt(((emb - self.box_centers) ** 2).sum(-1))
            if_valid = any(_d < self.box_size)
        action[0].set_valid_flg(if_valid=if_valid)
        return action


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue, if_respawn=True):
    """ To attach additional fn to spawned process """
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory, observation_space)
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if if_respawn:
                    if done:
                        observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory, observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))

            # ===== START: Extension ======
            # Note: We can attach our fn to each process

            # TEMPLATE; DON'T DELETE
            elif command == "a":
                action_space = env.action_space
                pipe.send(((None, "hello world", action_space), True))

            elif command == "get_item_embedding":
                pipe.send((env.act_embedding, True))

            elif command == "get_items_dict":
                pipe.send((env.items_dict, True))

            # ===== END: Extension ======

            else:
                raise ValueError
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class MultiUserAsyncWrapper(AsyncVectorEnv, MultiUserBaseWrapper):
    def __init__(self, **kwargs):
        AsyncVectorEnv.__init__(self,
                                env_fns=kwargs["env_fns"],
                                observation_space=kwargs["env_fns"][0]().observation_space,
                                action_space=kwargs["env_fns"][0]().action_space,
                                shared_memory=kwargs.get("shared_memory", True),
                                copy=kwargs.get("copy", True),
                                context=kwargs.get("context", None),
                                daemon=kwargs.get("daemon", True),
                                worker=kwargs.get("worker", None))
        self._prep(args=kwargs["args"], document_sampler=kwargs["document_sampler"])

    def execute(self, fn_name: str, args: list = None):
        """ Asynchronouslly Execute the fn on the raw env """
        self._assert_is_running()
        if args is None:
            args = [None for _ in range(self.num_envs)]
        for pipe, arg in zip(self.parent_pipes, args):
            pipe.send((fn_name, arg))
        res, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        return res, successes

    def step(self, actions):
        documents = self._slate_to_doc_cls(slates=actions)
        self.step_async(documents)
        res = self.step_wait()
        return res

    def reset(self, **kwargs):
        self.reset_async()
        return self.reset_wait()


class MultiUserSyncWrapper(SyncVectorEnv, MultiUserBaseWrapper):
    """ Debugging purpose """

    def __init__(self, **kwargs):
        SyncVectorEnv.__init__(self,
                               env_fns=kwargs["env_fns"],
                               observation_space=kwargs["env_fns"][0]().observation_space,
                               action_space=kwargs["env_fns"][0]().action_space,
                               copy=kwargs.get("copy", False))
        self._prep(args=kwargs["args"], document_sampler=kwargs["document_sampler"])
        self._if_respawn = kwargs["syncEnv_if_respawn"]

    def execute(self, fn_name: str, args: list = None):
        """ Alternative of execute API  in AsynchronousEnv """
        res = list()
        for i in range(len(self.envs)):
            if fn_name == "get_item_embedding":
                _res = self.envs[i].act_embedding
            elif fn_name == "get_items_dict":
                _res = self.envs[i].items_dict
            else:
                raise ValueError
            res.append(_res)
        return res, True

    def step(self, actions):
        documents = self._slate_to_doc_cls(slates=actions)
        self.step_async(documents)
        res = self.step_wait()
        return res

    def step_wait(self):
        observations, infos = [], []
        # import pudb; pudb.start()
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            action = self._check_valid_action(action=action)
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if self._if_respawn:
                if self._dones[i]:
                    observation = env.reset()
            observations.append(observation)
            infos.append(info)
        if "TimeLimit.truncated" in infos:
            del infos["TimeLimit.truncated"]
        infos = {k: np.asarray([dic[k] for dic in infos]) for k in infos[0]}  # Aggregate contents of info over envs
        self.observations = concatenate(observations, self.observations, self.single_observation_space)

        return (
            deepcopy(self.observations) if self.copy else self.observations, np.copy(self._rewards),
            np.copy(self._dones),
            infos)

    def reset(self, **kwargs):
        self.reset_async()
        res = self.reset_wait()  # just obs
        return res

    def reset_wait(self):
        self._dones[:] = False
        observations = []
        for env in self.envs:
            observation = env.reset()
            observations.append(observation)
        self.observations = concatenate(observations, self.observations, self.single_observation_space)

        return deepcopy(self.observations) if self.copy else self.observations


def launch_env(if_async: bool = True):
    if if_async:
        return MultiUserAsyncWrapper
    else:
        return MultiUserSyncWrapper
