import numpy as np
import random


class ReplayBufferRecSim(object):
    """ Experience Replay Memory Buffer which can accommodate the candidate sets """

    def __init__(self, args: dict):
        self._args = args
        self._maxsize = args["buffer_size"]
        self._storage = []
        self._next_idx = 0
        self._storage_mine_agent_pos = list()
        self._storage_ret = list() if args["agent_type"].startswith("wolp") else None
        self._next_idx_ret = 0 if args["agent_type"].startswith("wolp") else None

    def __len__(self):
        if self._storage_ret is not None:
            return len(self._storage) + len(self._storage_ret)
        else:
            return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, **kwargs):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._args["WOLP_dual_exp_if_ignore"]:
            if kwargs["if_retriever"]:
                if self._next_idx_ret >= len(self._storage_ret):
                    self._storage_ret.append(data)
                else:
                    self._storage_ret[self._next_idx_ret] = data
                self._next_idx_ret = (self._next_idx_ret + 1) % self._maxsize
            else:
                if self._next_idx >= len(self._storage):
                    self._storage.append(data)
                else:
                    self._storage[self._next_idx] = data
                self._next_idx = (self._next_idx + 1) % self._maxsize
        else:
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: list, if_retriever=False):
        actions, rewards, dones = list(), list(), list()
        obses_t, obses_tp1 = list(), list()
        _size = len(self._storage)

        for i in idxes:
            if self._args["WOLP_dual_exp_if_ignore"] and if_retriever:  # for retriever
                data = self._storage_ret[i]
            else:  # for selectionQ or DQN
                if self._args["WOLP_dual_exp_if_ignore"] and (i + 1) > _size:
                    data = self._storage_ret[i - _size]
                else:
                    data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data

            if self._args["WOLP_dual_exp_if_ignore"] and not if_retriever and (i + 1) > _size:
                # when update selectionQ if you sample from _storage_ret then remove all other info
                if not self._args["env_name"].lower().startswith('mujoco'):
                    action = [int(action[0][0])]
                # if self._args["agent_type"] == "wolp":
                #     action = [int(action[0])]
                # else:

            obses_t.append(obs_t)
            obses_tp1.append(obs_tp1)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            dones.append(done)

        obses_t, obses_tp1 = np.asarray(obses_t).astype(np.float32), np.asarray(obses_tp1).astype(np.float32)
        actions = np.array(actions)
        return obses_t, actions, np.array(rewards), obses_tp1, np.array(dones)

    def sample(self, batch_size, if_retriever=False):
        # Note: random module is under control of the random seed that has been set from the upper level(main.py)!
        if if_retriever:
            assert self._args["WOLP_dual_exp_if_ignore"]
            _size = len(self._storage_ret)
            idxes = [random.randint(0, _size - 1) for _ in range(batch_size)]
        else:
            _size = len(self._storage)
            if self._args["WOLP_dual_exp_if_ignore"]:
                _size += len(self._storage_ret)
            idxes = [random.randint(0, _size - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, if_retriever=if_retriever)

    def refresh(self):
        self._storage = []
        self._next_idx = 0


def _test():
    print("=== test ===")
    args = {
        "num_envs": 100000,
        "buffer_size": 100000,
        "agent_type": "wolp-sadf",
        "WOLP_dual_exp_if_ignore": True
    }

    # instantiate the replay memory
    replay_buffer = ReplayBufferRecSim(args=args)
    states = actions = rewards = np.random.randn(args["num_envs"])
    for i in range(args["num_envs"]):
        _flg = np.random.uniform() > 0.5
        replay_buffer.add(obs_t=states[i],
                          action=actions[i],
                          reward=rewards[i],
                          obs_tp1=states[i],
                          done=False, if_selectQ=True, if_retriever=_flg)

    obses, actions, rewards, next_obses, dones = replay_buffer.sample(batch_size=args["num_envs"])
    assert obses.shape[0] == args["num_envs"]


if __name__ == '__main__':
    _test()
