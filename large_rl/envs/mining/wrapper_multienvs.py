import sys
import numpy as np
from copy import deepcopy
from sklearn.manifold import TSNE
from gym.vector.utils import concatenate
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.async_vector_env import write_to_shared_memory, AsyncVectorEnv

from large_rl.envs.wrapper import wrap_env
from large_rl.commons.utils import tile_images
from large_rl.envs.mining.tool_set import Tool_set
from large_rl.envs.mining.miningroom import get_action_str, MiningRoomEnv


class BaseWrapper(object):
    def _prep(self, args: dict):
        self._args = args
        self._rng = np.random.RandomState(self._args["env_seed"])
        self._if_eval = False

        # Init mine tree and tools
        self._init_mines()
        self._tool_set = Tool_set(args=self._args, prop_to_idx=self.mine_property_to_idx,
                                  idx_to_prop=self.idx_to_mine_property)
        self._tool_set.build_tool_set(tree=self._tree)
        self._init_action_embed()
        # self.show_actions()
        if self._args["mw_test_save_action_embedding_tsne"]:
            self.show_action_embeddings()

        # set MW_ACTION_OFFSET
        if args['mw_four_dir_actions']:
            self.MW_ACTION_OFFSET = 4
        else:
            self.MW_ACTION_OFFSET = 3

    def show_action_embeddings(self):
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        _embed = self.act_embedding
        model = TSNE(n_components=2,
                     perplexity=10,
                     init="pca",
                     random_state=0,
                     method="exact",
                     n_iter=10000,  # 1000
                     n_jobs=-1)
        _embed = model.fit_transform(_embed)
        x = _embed[:, 0]
        y = _embed[:, 1]
        if self._args['mw_four_dir_actions']:
            # import seaborn as sns
            # col_list = sns.color_palette('pastel', n_colors=8)
            # col_list += sns.color_palette('Set2', n_colors=self._args["mw_mine_size"] - 7)
            #
            # # col_list = [(125, 175, 200), (150,100,150), (100,100,50), (100,50,100), (000,255,255), (50,100,150), (000,255,000), (150,50,100),
            # #             (50,50,150), (100,150,100), (000,000,255), (50,100,50), (150,100,50), (50,150,150), (100,50,50),
            # #             (50,150,50)]
            # # col_list = np.array(col_list) / 255.0
            #
            n = ['right', '', '', '']
            # n += [f"{i + 4}({self._tool_set.tools[i].mns},{self._tool_set.tools[i].mne})" for i in
            for tool in self._tool_set.tools:
                if tool.mns == 12 and tool.mne == 15:
                    n += [f"({tool.mns}, {tool.mne})"]
                elif tool.mns == 2 and tool.mne == 11:
                    n += [f"({tool.mns}, {tool.mne})"]
                elif tool.mns == 7 and tool.mne == 15:
                    n += [f"({tool.mns}, {tool.mne})"]
                else:
                    n += [""]


            # n += [f"({self._tool_set.tools[i].mns},{self._tool_set.tools[i].mne})" for i in
            #       range(self._args['mw_tool_size'])]
            # # colors = ["red", "red", "red", "red"]
            # # colors += ["green" for _ in range(self._args['mw_tool_size'])]
            # colors = [col_list[0] for _ in range(4)]
            # for tool in self._tool_set.tools:
            #     colors.append(col_list[tool.mns + 1])
            # colors = np.array(colors)

            col_list = [(211,211,211), (50,150,150), (100,100,50), (125, 175, 200), (50,50,150)]
            col_list = np.array(col_list) / 255.0
            colors = [col_list[0] for _ in range(4)]
            colors[0] = col_list[3]
            for tool in self._tool_set.tools:
                if tool.mns == 12 and tool.mne == 15:
                    colors.append(col_list[1])
                elif tool.mns == 2 and tool.mne == 11:
                    colors.append(col_list[2])
                elif tool.mns == 7 and tool.mne == 15:
                    colors.append(col_list[4])
                else:
                    colors.append(col_list[0])
            colors = np.array(colors)
        else:
            n = ['0left', '1right', '2forward']
            n += [f"{i + 3}({self._tool_set.tools[i].mns},{self._tool_set.tools[i].mne})" for i in
                  range(self._args['mw_tool_size'])]
            colors = ["red", "red", "red"]
            colors += ["green" for _ in range(self._args['mw_tool_size'])]
            colors = np.array(colors)
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=colors)
        ax.set_xticks([])
        ax.set_yticks([])
        font = {'size': 14}

        matplotlib.rc('font', **font)

        # patches = []
        # patch = mpatches.Patch(color=col_list[0], label='Navigation')
        # patches.append(patch)
        # for i in range(self._args["mw_mine_size"]):
        #     patch = mpatches.Patch(color=col_list[i+1], label=f'Mine {i}')
        #     patches.append(patch)


        # for i, txt in enumerate(n):
        #     ax.annotate(txt, (x[i], y[i]))
        # ax.legend(handles=patches, loc='best', ncol=1, bbox_to_anchor=(0.8, 0.2, 0.2, 0.6))
        import os
        directory = './src'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, 'env_action_embedding.pdf'), bbox_inches='tight')
        print("action embedding tsne saved")
        # print(x, y)

    def _init_action_embed(self):
        # load all the embedding from tool set
        if self._args['mw_action_id']:
            act_right = tuple([1, 0] + [0] + [0])
            act_down = tuple([1, 1.0 / 3] + [0] + [0])
            act_left = tuple([1, 2.0 / 3] + [0] + [0])
            act_up = tuple([1, 3.0 / 3] + [0] + [0])
            self._action_embed = [act_right, act_down, act_left, act_up]
        elif self._args['mw_four_dir_actions']:
            if self._args['mw_dir_one_hot']:
                act_right = tuple([1, 1, 0, 0, 0] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                                  + [0 for _ in range(self._args['mw_mine_size'] + 1)])
                act_down = tuple([1, 0, 1, 0, 0] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                                 + [0 for _ in range(self._args['mw_mine_size'] + 1)])
                act_left = tuple([1, 0, 0, 1, 0] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                                 + [0 for _ in range(self._args['mw_mine_size'] + 1)])
                act_up = tuple([1, 0, 0, 0, 1] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                               + [0 for _ in range(self._args['mw_mine_size'] + 1)])
            else:
                act_right = tuple([1, 1, 0] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                                  + [0 for _ in range(self._args['mw_mine_size'] + 1)])
                act_down = tuple([1, 0, 1] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                                 + [0 for _ in range(self._args['mw_mine_size'] + 1)])
                act_left = tuple([1, -1, 0] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                                 + [0 for _ in range(self._args['mw_mine_size'] + 1)])
                act_up = tuple([1, 0, -1] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                               + [0 for _ in range(self._args['mw_mine_size'] + 1)])
            self._action_embed = [act_right, act_down, act_left, act_up]
        else:
            act_left = tuple([1, 1, 0, 0] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                             + [0 for _ in range(self._args['mw_mine_size'] + 1)])
            act_right = tuple([1, 0, 1, 0] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                              + [0 for _ in range(self._args['mw_mine_size'] + 1)])
            act_forward = tuple([1, 0, 0, 1] + [0 for _ in range(self._args['mw_mine_size'] + 1)]
                                + [0 for _ in range(self._args['mw_mine_size'] + 1)])
            self._action_embed = [act_left, act_right, act_forward]
        self._action_embed += [self._tool_set.tools[i].emb for i in range(self._args['mw_tool_size'])]
        self._action_embed = np.asarray(self._action_embed)
        print("[Env] Action embedding generated")
        # print(self._action_embed.shape)

    @property
    def args(self):
        return self._args

    def train(self):
        self._if_eval = False

    def eval(self):
        self._if_eval = True

    def execute(self, fn_name: str, args: list = None):
        raise NotImplementedError

    @property
    def act_embedding(self):
        from sklearn.preprocessing import MinMaxScaler  # this is to clearly constraint the action space

        _embed = self._action_embed
        if not self._args["mw_embedding_perfect"]:
            _shape = (self._args["num_all_actions"], self._action_embed.shape[-1])
            if self._args["mw_act_emb_lin_scale"] == 1.0:
                _scales = np.ones(_shape)
            else:
                _scales = self._rng.randn(*_shape) * self._args["mw_act_emb_lin_scale"]
            _shifts = self._rng.randn(*_shape) * self._args["mw_act_emb_lin_shift"]
            _embed = np.multiply(_embed, _scales) + _shifts  # linear scaling

        if self._args["mw_if_high_dim"]:
            temp_rng = np.random.RandomState(123)
            _kernel = temp_rng.rand(_embed.shape[1], self._args["mw_new_action_dim"])
            _embed = np.matmul(_embed, _kernel)

        if self._args['mw_tsne_embedding']:
            model = TSNE(n_components=self._args["mw_tsne_dim"],
                         perplexity=4,
                         init="pca",
                         random_state=0,
                         method="exact",
                         n_iter=250,  # 1000
                         n_jobs=-1)
            _embed = model.fit_transform(_embed)

        if self._args["DEBUG_size_action_space"] == "small":
            _embed = MinMaxScaler(feature_range=(0, 1)).fit_transform(_embed)
        elif self._args["DEBUG_size_action_space"] == "large":
            _embed = MinMaxScaler(feature_range=(-1, 1)).fit_transform(_embed)
        else:
            raise ValueError
        return _embed

    @property
    def task_embedding(self):
        return self.act_embedding

    def true_action_meaning_dict(self):
        action_meaning_dict = dict()
        for _id in range(self._args["num_all_actions"]):
            action = self._convert_id_to_action(actions=np.asarray(_id))[0]
            action_meaning_dict[_id] = get_action_str(action=action, args=self._args)
        return action_meaning_dict

    def _init_mines(self):
        self._init_mine_tree()
        # print('Tree: ', self._tree)
        self.mine_property_to_idx = dict()  # key: one-hot vec of mineId, value: mineId
        self.idx_to_mine_property = dict()  # key: mineId, value: one-hot vec of mineId
        for i in range(self._args['mw_mine_size'] + 1):
            prop = [0 for _ in range(self._args['mw_mine_size'] + 1)]
            prop[i] = 1
            prop = tuple(prop)
            self.mine_property_to_idx[prop] = i
            self.idx_to_mine_property[i] = prop

    def _init_mine_tree(self):
        assert self._args['mw_mine_tree_max_depth'] >= self._args['mw_mine_tree_min_depth'] >= 1
        if self._args['mw_mine_tree_max_depth'] == 1:
            assert self._args['mw_tool_size'] == self._args['mw_mine_size']
            self._tree = [[i for i in range(self._args['mw_mine_size'])]]
        else:
            assert self._args['mw_mine_size'] <= self._args['mw_tool_size'] <= ((self._args['mw_mine_size'] ** 2) // 4)
            self._tree = []
            pick_start_point = 0
            tree_layer = 0
            ## init all the available mine nodes
            available_mine_num = self._args["mw_mine_size"]
            while (tree_layer < self._args['mw_mine_tree_max_depth'] - 1 and available_mine_num > 0):
                if available_mine_num >= self._args["mw_mine_size"] // self._args['mw_mine_tree_max_depth']:
                    min_num = max(1, self._args["mw_mine_size"] // self._args['mw_mine_tree_max_depth'])
                    max_num = max(2, self._args["mw_mine_size"] // self._args['mw_mine_tree_min_depth'] + 1)
                    layer_node_num = self._rng.randint(
                        min_num,
                        max_num)
                else:
                    layer_node_num = available_mine_num
                layer_nodes = [i for i in range(pick_start_point, pick_start_point + layer_node_num)]
                pick_start_point += layer_node_num
                self._tree.append(layer_nodes)
                tree_layer += 1
                available_mine_num -= layer_node_num
            ## Use the leftover nodes to generate the last layer
            if available_mine_num > 0:
                layer_nodes = [
                    i for i in range(self._args["mw_mine_size"] - available_mine_num, self._args["mw_mine_size"])]
                self._tree.append(layer_nodes)
        print("Mine-Tree", self._tree)

    def _convert_id_to_action(self, actions: np.ndarray) -> list:
        """ Returns Tool obj if action is to use tool otherwise int representing movement """
        _res = list()
        actions = actions.flatten()
        for _action_id in actions:
            if _action_id >= self.MW_ACTION_OFFSET:
                _res.append(self._tool_set.get_tool(tool_id=_action_id - self.MW_ACTION_OFFSET))
            else:
                _res.append(_action_id)
        return _res

    def check_candidate_set(self, candidate_set: np.ndarray) -> np.ndarray:
        _candidate_list = list()
        for _set in candidate_set:
            _set = self._convert_id_to_action(actions=_set)
            _candidate_list.append(_set)

        return np.asarray(self.execute(fn_name="check_candidate_set", args=_candidate_list)[0])


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
            elif command == 'set_path':
                env.set_path(dir_path=data)
                pipe.send((None, True))
            elif command == 'show_true_states':
                env.show_true_states()
                pipe.send((None, True))
            elif command == 'enable_render_info':
                env.enable_render_info()
                pipe.send((None, True))
            elif command == 'disable_render_info':
                env.disable_render_info()
                pipe.send((None, True))


            # ===== START: Extension ======
            # Note: We can attach our fn to each process

            # TEMPLATE; DON'T DELETE
            elif command == "a":
                pipe.send(((None, "hello world", env.action_space), True))

            elif command == "get_item_embedding":
                pipe.send((env.act_embedding, True))

            elif command == "get_items_dict":
                pipe.send((env.items_dict, True))

            elif command == 'set_seed':
                env.set_seed(data)
                pipe.send((None, True))

            elif command == 'render':
                pipe.send((env.render(mode=data), True))

            # ===== END: Extension ======

            else:
                raise ValueError
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class VecAsyncWrapper(AsyncVectorEnv, BaseWrapper):
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
        self._prep(args=kwargs["args"])

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
        actions = self._convert_id_to_action(actions=actions)
        self.step_async(actions)
        return self.step_wait()

    def reset(self, **kwargs):
        self.reset_async()
        res = self.reset_wait()
        return res

    def set_seed(self, seed_list: list):
        self.execute(fn_name="set_seed", args=seed_list)

    def render(self, mode='rgb_array'):
        img_list = self.execute("render", [mode for _ in range(self.num_envs)])[0]
        bigimg = tile_images(img_list)
        if mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def render_frame(self, mode='rgb_array', info=None):
        if mode != 'rgb_array':
            raise NotImplementedError
        if info is not None:
            img_list = [info[i]['frame'] for i in range(self.num_envs)]
        else:
            img_list = self.execute("render", [mode for _ in range(self.num_envs)])[0]
        return img_list

    def show_true_states(self):
        self.execute('show_true_states')

    def enable_render_info(self):
        # import pudb; pudb.set_trace()
        self.execute(fn_name="enable_render_info")

    def disable_render_info(self):
        self.execute(fn_name="disable_render_info")


class VecSyncWrapper(SyncVectorEnv, BaseWrapper):
    """ Debugging purpose """

    def __init__(self, **kwargs):
        SyncVectorEnv.__init__(self,
                               env_fns=kwargs["env_fns"],
                               observation_space=kwargs["env_fns"][0]().observation_space,
                               action_space=kwargs["env_fns"][0]().action_space,
                               copy=kwargs.get("copy", True))
        self._if_respawn = kwargs["syncEnv_if_respawn"]
        self._prep(args=kwargs["args"])

    def execute(self, fn_name: str, args: list = None):
        """ Alternative of execute API in AsynchronousEnv """
        res = list()
        for i in range(len(self.envs)):
            if fn_name == "set_seed":
                _res = self.envs[i].set_seed(seed=args[i])
            elif fn_name == "show_true_states":
                _res = self.envs[i].show_true_states()
            elif fn_name == "enable_render_info":
                _res = self.envs[i].enable_render_info()
            elif fn_name == "disable_render_info":
                _res = self.envs[i].disable_render_info()
            elif fn_name == "render":
                _res = self.envs[i].render(mode=args[i])
            elif fn_name == "check_candidate_set":
                _res = self.envs[i].check_candidate_set(candidate_set=args[i])
            else:
                raise ValueError
            res.append(_res)
        return res, True

    def step(self, actions):
        actions = self._convert_id_to_action(actions=actions)
        self.step_async(actions)
        return self.step_wait()

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if self._if_respawn:
                # if self._if_respawn or not self._if_eval:
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

    def reset(self, **kwargs):
        self.reset_async()
        res = self.reset_wait()
        return res

    def set_seed(self, seed_list: list):
        for _seed in seed_list:
            self.execute(fn_name="set_seed", args=seed_list)

    def render(self, mode='rgb_array'):
        return self.envs[0].render(mode=mode)

    def render_frame(self, mode='rgb_array', info=None):
        if mode != 'rgb_array':
            raise NotImplementedError
        if info is not None:
            img_list = [info[i]['frame'] for i in range(self.num_envs)]
        else:
            img_list = self.execute("render", [mode for _ in range(self.num_envs)])[0]
        return img_list

    def show_true_states(self):
        self.execute('show_true_states')

    def enable_render_info(self):
        self.execute("enable_render_info")

    def disable_render_info(self):
        self.execute("disable_render_info")


def launch_env(if_async: bool = True):
    if if_async:
        return VecAsyncWrapper
    else:
        return VecSyncWrapper


def create_environment(args: dict, seed):
    """Creates an interest evolution environment."""
    env = MiningRoomEnv(mine_size=args['mw_mine_size'],
                        tool_size=args['mw_tool_size'],
                        max_score=args['mw_max_score'],
                        bonus=args['mw_bonus'],
                        step_penalty_coef=args['mw_step_penalty_coef'],
                        goal_reaching_reward=args['mw_goal_reaching_reward'],
                        fullness=args['mw_fullness'],
                        # minNumRooms=args['mw_minNumRooms'],
                        # maxNumRooms=args['mw_maxNumRooms'],
                        maxRoomSize=args['mw_maxRoomSize'],
                        minRoomSize=args['mw_minRoomSize'],
                        maxTimeStep=args['max_episode_steps'] + 1,  # To fully execute the episode-length
                        seed=seed,
                        args=args)
    env = wrap_env(env=env, args=args)
    return env


def create_vector_environment(args: dict):
    seed_list = [i + args['env_seed'] for i in range(args["num_envs"])]  # To set the seed externally

    env_cls = launch_env(if_async=args["if_async"])
    train_env = env_cls(
        env_fns=[lambda: create_environment(args=args, seed=seed_list[i]) for i in range(args["num_envs"])],
        worker=_worker_shared_memory, args=args, syncEnv_if_respawn=True
    )
    train_env.set_seed(seed_list=seed_list)  # Seed set for RecSim needs to be done externally
    train_env.reset()
    return train_env
