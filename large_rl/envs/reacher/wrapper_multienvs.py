import os.path
import numpy as np
from copy import deepcopy
from gym.vector.utils import concatenate

from large_rl.envs.mining.wrapper_multienvs import AsyncVectorEnv as Async, SyncVectorEnv as Sync, BaseWrapper as base
from large_rl.envs.mining.wrapper_multienvs import _worker_shared_memory
from large_rl.commons.utils import tile_images


class BaseWrapper(base):
    def _prep(self, args: dict):
        self._args = args
        self._rng = np.random.RandomState(self._args["env_seed"])
        self._if_eval = False

    @property
    def act_embedding(self):
        return np.zeros((1, 1))


class VecAsyncWrapper(Async, BaseWrapper):
    def __init__(self, **kwargs):
        Async.__init__(self,
                       env_fns=kwargs["env_fns"],
                       observation_space=kwargs["env_fns"][0]().observation_space,
                       action_space=kwargs["env_fns"][0]().action_space,
                       shared_memory=kwargs.get("shared_memory", True),
                       copy=kwargs.get("copy", True),
                       context=kwargs.get("context", None),
                       daemon=kwargs.get("daemon", True),
                       worker=kwargs.get("worker", None))
        self._prep(args=kwargs["args"])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def set_path(self, path):
        # print([os.path.join(path, "{:03d}".format(i)) for i in range(self.num_envs)])
        self.execute([os.path.join(path, "{:03d}".format(i)) for i in range(self.num_envs)])

    def set_seed(self, seed_list: list):
        for _seed in seed_list:
            self.execute(fn_name="set_seed", args=seed_list)

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

    # def render(self, mode='rgb_array'):
    # return self.execute("render", [mode for _ in range(self.num_envs)])[0][0]

    def render(self, mode='rgb_array'):
        img_list = self.execute("render", [mode for _ in range(self.num_envs)])[0]
        bigimg = tile_images(img_list)
        if mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def video_saving(self, vid_dir):
        self.execute('video_saving', [(vid_dir, i) for i in range(self.num_envs)])


class VecSyncWrapper(Sync, BaseWrapper):
    """ Debugging purpose """

    def __init__(self, **kwargs):
        Sync.__init__(self,
                      env_fns=kwargs["env_fns"],
                      observation_space=kwargs["env_fns"][0]().observation_space,
                      action_space=kwargs["env_fns"][0]().action_space,
                      copy=kwargs.get("copy", True))
        self._if_respawn = kwargs["syncEnv_if_respawn"]
        self._prep(args=kwargs["args"])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_wait(self):  # for debugging purpose
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if self._if_respawn:
                if self._dones[i]:
                    observation = env.reset()
            observations.append(observation)
            infos.append(info)
        if "TimeLimit.truncated" in infos:
            del infos["TimeLimit.truncated"]
        self.observations = concatenate(observations, self.observations, self.single_observation_space)
        return (
            deepcopy(self.observations) if self.copy else self.observations, np.copy(self._rewards),
            np.copy(self._dones),
            infos)

    def render(self, mode='rgb_array'):
        img_list = [env.render(mode=mode) for env in self.envs]
        bigimg = tile_images(img_list)
        if mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError
        # return self.envs[0].render(mode=mode)

    def set_path(self, path):
        [self.envs[i].set_path(os.path.join(path, "{:03d}".format(i))) for i in range(self.num_envs)]

    def video_saving(self, vid_dir):
        self.envs[0].video_saving(vid_dir)

    def set_seed(self, seed_list: list):
        for _seed in seed_list:
            self.execute(fn_name="set_seed", args=seed_list)

    def execute(self, fn_name: str, args: list = None):
        """ Alternative of execute API in AsynchronousEnv """
        res = list()
        for i in range(len(self.envs)):
            if fn_name == "set_mine_score":
                _res = self.envs[i].set_mine_score(args)
            elif fn_name == "set_seed":
                _res = self.envs[i].set_seed(seed=args[i])
            else:
                raise ValueError
            res.append(_res)
        return res, True


def launch_env(if_async: bool = True):
    if if_async:
        return VecAsyncWrapper
    else:
        return VecSyncWrapper


def create_vector_environment(args: dict):
    if args["env_name"] == "mujoco-reacher":
        from large_rl.envs.reacher.reacher import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-ant":
        from large_rl.envs.reacher.ant_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-half_cheetah":
        from large_rl.envs.reacher.half_cheetah_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-hopper":
        from large_rl.envs.reacher.hopper_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-humanoid":
        from large_rl.envs.reacher.humanoid_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-humanoidstandup":
        from large_rl.envs.reacher.humanoidstandup_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-inverted_double_pendulum":
        from large_rl.envs.reacher.inverted_double_pendulum_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-inverted_pendulum":
        from large_rl.envs.reacher.inverted_pendulum_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-pusher":
        from large_rl.envs.reacher.pusher_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-swimmer":
        from large_rl.envs.reacher.swimmer_v4 import create_environment as _fn
        create_environment = _fn
    elif args["env_name"] == "mujoco-walker2d":
        from large_rl.envs.reacher.walker2d_v4 import create_environment as _fn
        create_environment = _fn
    else: raise ValueError
    seed_list = [env_id + args['env_seed'] for env_id in range(args["num_envs"])]  # To set the seed externally

    env_cls = launch_env(if_async=args["if_async"])  # do NOT use `True`, process within process is super slow!!!
    train_env = env_cls(
        env_fns=[lambda: create_environment(args=args, seed=seed_list[i]) for i in range(args["num_envs"])],
        worker=_worker_shared_memory, args=args, syncEnv_if_respawn=True
    )
    train_env.set_seed(seed_list=seed_list)  # Seed set for RecSim needs to be done externally
    train_env.reset()
    return train_env
