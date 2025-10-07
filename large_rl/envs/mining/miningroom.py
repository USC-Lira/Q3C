import cv2
import copy
import numpy as np

from enum import IntEnum
from gym import spaces
from copy import deepcopy

from large_rl.envs.mining.tool_set import Tool
from large_rl.envs.mining.multiroom import MultiRoomEnv
from large_rl.envs.mining.minigrid import Mine, Grid, Wall, Goal, CELL_PIXELS


def convert_to_hex(rgba_color):
    return '#%02x%02x%02x' % (rgba_color[0], rgba_color[1], rgba_color[2])


def append_step_reward_to_image(image, cum_r, step_r, mine_r, action_embed, ts, cell_name, if_goal_reached):
    new_frame = cv2.putText(img=image, text=f'ts: {ts + 1}',
                            org=(10, 40), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=CELL_PIXELS / 40,
                            color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    new_frame = cv2.putText(img=new_frame, text=f'StepR: {step_r:.2f} / CumR: {cum_r:.2f} / MineR: {mine_r:.2f}',
                            org=(10, 65), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=CELL_PIXELS / 50,
                            color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    if if_goal_reached:
        new_frame = cv2.putText(img=new_frame, text=f'Goal Reached',
                                org=(10, 90), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=CELL_PIXELS / 40,
                                color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    else:
        new_frame = cv2.putText(img=new_frame, text=f'a: {action_embed}',
                                org=(10, 115), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=CELL_PIXELS / 40,
                                color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        if cell_name is not None:
            new_frame = cv2.putText(img=new_frame, text=f'front cell: {cell_name}',
                                    org=(10, 90), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=CELL_PIXELS / 40,
                                    color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return new_frame


def action_onehot2number(action_embed, args: dict):
    mine_size = args["mw_mine_size"] + 1
    if args['mw_action_id']:
        mine1_num = int(action_embed[2] * args["mw_mine_size"])
        mine2_num = int(action_embed[3] * args["mw_mine_size"])
    else:
        if args['mw_four_dir_actions']:
            if args['mw_dir_one_hot']:
                OFFSET = 5
            else:
                OFFSET = 3
        else:
            OFFSET = 4
        assert (len(action_embed) == OFFSET + 2 * mine_size)
        action_embed = action_embed[OFFSET:]
        mine1_num = -1
        mine2_num = -1
        mine1 = action_embed[:mine_size]
        mine2 = action_embed[mine_size:]
        for i in range(mine_size):
            if mine1[i]:
                mine1_num = i
                break
        for i in range(mine_size):
            if mine2[i]:
                mine2_num = i
                break
    return mine1_num, mine2_num


def get_action_str(action: int, args: dict):
    if action is None:
        act_label = ""
    else:
        if args['mw_four_dir_actions']:
            if type(action) == Tool:  # if action is to use tool
                act_label = action_onehot2number(action_embed=action.emb, args=args)
            elif action == 2:
                act_label = 'left'
            elif action == 0:
                act_label = 'right'
            elif action == 3:
                act_label = 'up'
            elif action == 1:
                act_label = 'down'
            else:
                act_label = None
        else:
            if type(action) == Tool:  # if action is to use tool
                act_label = action_onehot2number(action_embed=action.emb, args=args)
            elif action == 2:
                act_label = 'left'
            elif action == 0:
                act_label = 'right'
            else:
                act_label = None
    return str(act_label)


class MiningRoomEnv(MultiRoomEnv):
    # Enumeration of possible actions
    class Actions2(IntEnum):
        # Turn left, turn right, move forward
        right = 0
        down = 1
        left = 2
        up = 3

    def __init__(self,
                 mine_size=2,
                 tool_size=2,
                 fullness=0.5,
                 minRoomSize=4,
                 maxRoomSize=10,
                 max_score=10,
                 bonus=0.1,
                 step_penalty_coef=0.01,
                 seed=2022,
                 goal_reaching_reward=1,
                 maxTimeStep=200,
                 plt_mine_tool_graph=False,
                 args: dict = None):
        self.set_seed(seed)
        self._fullness = fullness
        self._mine_size = mine_size
        self._tool_size = tool_size
        self._max_score = max_score
        self.max_steps = maxTimeStep
        self._bonus = bonus
        self._goal_reaching_reward = goal_reaching_reward
        self._step_penalty_coef = step_penalty_coef
        # self.rgb_frames = []

        self._prep_COLOR_CONSTANTS()
        self._mining_init(args['mw_rand_mine_score'])
        if plt_mine_tool_graph:
            self.show_mines()
        self.distance = []
        self.info_dict_for_render = dict()
        self.info_dict_for_render['action'] = None
        self.info_dict_for_render['reward'] = 0
        self.info_dict_for_render['ep_mining_reward'] = 0
        self.info_dict_for_render['ep_reward'] = 0
        self.previous_distance_to_goal = 0
        self.max_distance_to_goal = 0
        self.accu_mining_reward = 0
        self.time_penalty = args["mw_time_penalty"]
        super(MiningRoomEnv, self).__init__(1, 1, maxRoomSize, minRoomSize, args=args)
        if self._args['mw_four_dir_actions']:
            self.actions = MiningRoomEnv.Actions2
        self.action_space = spaces.Discrete(len(self.actions) + tool_size)  # Actions are discrete integer values
        if not self._args["mw_randomise_grid"]:
            self._grid_original = deepcopy(self.grid)
        else:
            self._grid_original = None

    def set_seed(self, seed):
        self.seed(seed)

    def _prep_COLOR_CONSTANTS(self):
        """ Prep the colour related constants that require huge computations """
        self.COLORS_SHORTCUT = {
            'red': '255,000,000',
            'green': '000,255,000',
            'blue': '000,000,255',
            'purple': '112,039,195',
            'yellow': '255,255,000',
            'grey': '100,100,100',
            'cyan': '000,255,255',
        }
        self._wall_color = '100,100,100'
        self._goal_color = '000,255,000'

        for i in range(1, 256):
            # if (i ^ 3) >= self._mine_size:
            if (i ** 3) >= self._mine_size:
                self.COLOR_RANGE = i
                break

        self.COLORS = []
        self.COLOR_KEYS = []
        self.COLOR_VALUES = []

        for r in range(self.COLOR_RANGE):
            for g in range(self.COLOR_RANGE):
                for b in range(self.COLOR_RANGE):
                    new_r = round((r / self.COLOR_RANGE) * 150 + 50)
                    new_g = round((g / self.COLOR_RANGE) * 150 + 50)
                    new_b = round((b / self.COLOR_RANGE) * 150 + 50)
                    self.COLOR_KEYS.append('{r:03d},{g:03d},{b:03d}'.format(r=new_r, g=new_g, b=new_b))
                    self.COLOR_VALUES.append((new_r, new_g, new_b))

        for color_key in self.COLORS_SHORTCUT.values():
            color_values = list(map(int, color_key.split(",")))
            if color_key not in self.COLOR_KEYS:
                self.COLOR_KEYS.append(color_key)
                self.COLOR_VALUES.append(tuple(color_values))

        self.COLORS = dict(zip(self.COLOR_KEYS, self.COLOR_VALUES))
        self.COLOR_NAMES = sorted(list(self.COLORS.keys()))

        # Used to map colors to integers
        self.COLOR_IDX_VALUE = list(range(len(self.COLOR_KEYS)))
        self.COLOR_TO_IDX = dict(zip(self.COLOR_KEYS, self.COLOR_VALUES))

    ### Init mine set and tool set
    def _mining_init(self, mw_rand_mine_score):
        mine_colors = self._rand_subset(self.COLOR_KEYS, self._mine_size)
        if mw_rand_mine_score:
            self.mines = [Mine(mine_colors[i], index=i, score=self._rand_float(0, self._max_score)) for i in
                          range(len(mine_colors))]
        else:
            self.mines = [Mine(mine_colors[i], index=i, score=self._max_score) for i in
                          range(len(mine_colors))]

    def _reward(self):
        return self._goal_reaching_reward * (1 - 0.9 * self.step_count / self.max_steps)

    def bonus(self):
        return self._bonus

    def _manhattan_distance(self, start_pos, goal_pos):
        start_pos = np.array(start_pos)
        goal_pos = np.array(goal_pos)
        return np.sum(np.abs(start_pos - goal_pos))

    def _min_distance_to_goal(self):
        present_x, present_y = self.agent_pos
        room_num = self.is_in_room(present_x, present_y, True)
        if room_num == len(self.rooms) - 1:
            dis = self._manhattan_distance(start_pos=self.agent_pos, goal_pos=self.goal_pos)
        else:
            dis = self.distance[room_num] + self._manhattan_distance(start_pos=self.agent_pos,
                                                                     goal_pos=self.rooms[room_num + 1].entryDoorPos)
        return dis

    def set_mine_score(self, max_score):
        for node in self.mines:
            node.set_score(self._rand_float(0, max_score))

    def change_mine(self, fwd_pos, mne):
        if mne >= len(self.mines):
            self.grid.set(*fwd_pos, None)
            return
        self.grid.set(*fwd_pos, copy.deepcopy(self.mines[mne]))

    def _step_reward(self):
        _r = self._step_penalty_coef * (self.previous_distance_to_goal - self._min_distance_to_goal())
        return _r

    def _time_penalty(self):
        return self.time_penalty

    def enable_render_info(self):
        self.render_info = True

    def disable_render_info(self):
        self.render_info = False

    def check_candidate_set(self, candidate_set: list):
        bonus_multiple_solution = 0
        # Get the cell that leads to the shortest path
        # Right: 0 / Down: 1 / Left: 2 / Up: 3
        if self.agent_dir == 0:  # Facing Right
            _cell_side = self.grid.get(i=self.agent_pos[0] + 1, j=self.agent_pos[1])  # Check Down
            _cell_front = self.grid.get(*self._pos_tm1)
            if _cell_side is None:
                if _cell_front is None:
                    if 0 in candidate_set and 1 in candidate_set:  # right and down
                        bonus_multiple_solution = 1
                elif _cell_front.type == 'mine':
                    if 1 in candidate_set:  # down and right digging tool
                        for action in candidate_set:
                            if type(action) == Tool:
                                if action.mns == _cell_front.mine_id:  # check if the right digging tool is included
                                    bonus_multiple_solution = 1
        elif self.agent_dir == 1:  # Facing Down
            _cell_side = self.grid.get(i=self.agent_pos[0], j=self.agent_pos[1] + 1)  # Check Right
            _cell_front = self.grid.get(*self._pos_tm1)
            if _cell_side is None:
                if _cell_front is None:
                    if 0 in candidate_set and 1 in candidate_set:  # right and down
                        bonus_multiple_solution = 1
                elif _cell_front.type == 'mine':
                    if 0 in candidate_set:  # down
                        for action in candidate_set:  # check if the right digging tool is included
                            if type(action) == Tool:
                                if action.mns == _cell_front.mine_id:
                                    bonus_multiple_solution = 1
        return bonus_multiple_solution

    def step(self, action):
        """
        Args:
            action: Tool obj if action was to use tool otherwise int representing movement
        """
        self.info_dict_for_render['action'] = action
        self._pos_tm1 = deepcopy(self.front_pos)

        info = {}
        if self.render_info:
            frame = self.render(mode='rgb_array')
            info['frame'] = frame

        self.step_count += 1
        reward = self._time_penalty()
        done = False

        if self._args['mw_four_dir_actions']:
            if type(action) == Tool:  # if action is to use tool
                fwd_pos = self.front_pos  # Get the position in front of the agent
                fwd_cell = self.grid.get(*fwd_pos)  # Get the contents of the cell in front of the agent
                if fwd_cell != None and fwd_cell.type == 'mine':
                    if action.mns == fwd_cell.mine_id:
                        reward += fwd_cell.score  # Mine digging reward
                        self.change_mine(fwd_pos, action.mne)
                        if action.mne == self._mine_size:  # when a mine reaches the terminal state
                            reward += self.bonus()
                        self.accu_mining_reward += reward
            elif action == self.actions.right or action == self.actions.down \
                    or action == self.actions.left or action == self.actions.up:
                if action == self.actions.right:
                    self.agent_dir = 0
                elif action == self.actions.down:
                    self.agent_dir = 1
                elif action == self.actions.left:
                    self.agent_dir = 2
                elif action == self.actions.up:
                    self.agent_dir = 3
                fwd_pos = self.front_pos  # Get the position in front of the agent
                fwd_cell = self.grid.get(*fwd_pos)  # Get the contents of the cell in front of the agent
                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agent_pos = fwd_pos
                    reward += self._step_reward()  # refresh step reward with new agent position
                if fwd_cell != None and fwd_cell.type == 'goal':
                    done = True
                    self.goal_reached = True
                    reward += self._reward()
            else:
                assert False, "unknown action"
        else:
            # print("env action: ", action)
            fwd_pos = self.front_pos  # Get the position in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)  # Get the contents of the cell in front of the agent
            if type(action) == Tool:  # if action is to use tool
                if fwd_cell != None and fwd_cell.type == 'mine':
                    reward = self._step_reward()
                    if action.mns == fwd_cell.mine_id:
                        reward += fwd_cell.score  # Mine digging reward
                        self.change_mine(fwd_pos, action.mne)
                        if action.mne == self._mine_size:  # when a mine reaches the terminal state
                            reward += self.bonus()
            elif action == self.actions.left:

                self.agent_dir = (self.agent_dir + 3) % 4
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4
            elif action == self.actions.forward:
                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agent_pos = fwd_pos
                    reward = self._step_reward()  # refresh step reward with new agent position

                if fwd_cell != None and fwd_cell.type == 'goal':
                    done = True
                    self.goal_reached = True
                    reward = self._reward()
            else:
                assert False, "unknown action"

        self.previous_distance_to_goal = self._min_distance_to_goal()

        self.info_dict_for_render['reward'] = reward
        self.info_dict_for_render['ep_mining_reward'] = self.accu_mining_reward
        self.info_dict_for_render['ep_reward'] = self.episode_reward
        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        self.episode_reward += reward

        info['ep_success'] = self.goal_reached
        info['ep_len'] = self.step_count
        info['ep_reward'] = self.episode_reward
        info['ep_subgoal_reached'] = self.subgoal_reached
        info['ep_mining_reward'] = self.accu_mining_reward
        info["agent_pos"] = self.agent_pos

        return obs, reward, done, info

    def show_mines(self):
        print('Mine proerties')
        print('{:<10} {:<10} {:10}'.format('INDEX', 'COLOR', 'SCORE'))
        for mine in self.mines:
            print('{:<10} {:<10} {:10}'.format(str(mine.mine_id), str(mine.color), str(mine.score)))

    def _gen_grid(self, width, height, if_on_launch: bool = False):

        if if_on_launch or self._args["mw_randomise_grid"]:
            roomList = []

            # Choose a random number of rooms to generate
            # numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms + 1)
            numRooms = 1
            assert self.minRoomSize == self.maxRoomSize

            while len(roomList) < numRooms:
                curRoomList = []

                entryDoorPos = (
                    self._rand_int(width - self.minRoomSize - 1, width - self.minRoomSize),
                    self._rand_int(width - self.minRoomSize - 1, width - self.minRoomSize)
                )

                # Recursively place the rooms
                self._placeRoom(
                    numRooms,
                    roomList=curRoomList,
                    minSz=self.minRoomSize,
                    maxSz=self.maxRoomSize,
                    entryDoorWall=2,
                    entryDoorPos=entryDoorPos
                )

                if len(curRoomList) > len(roomList):
                    roomList = curRoomList

            # Store the list of rooms in this environment
            assert len(roomList) > 0
            self.rooms = roomList

            # Create the grid
            self.grid = Grid(COLOR_TO_IDX=self.COLOR_TO_IDX, width=width, height=height, wall_color=self._wall_color,
                             args=self._args)
            wall = Wall(color=self._wall_color)

            # For each room
            for idx, room in enumerate(roomList):

                topX, topY = room.top
                sizeX, sizeY = room.size

                # Draw the top and bottom walls
                for i in range(0, sizeX):
                    self.grid.set(topX + i, topY, wall)
                    self.grid.set(topX + i, topY + sizeY - 1, wall)

                # Draw the left and right walls
                for j in range(0, sizeY):
                    self.grid.set(topX, topY + j, wall)
                    self.grid.set(topX + sizeX - 1, topY + j, wall)

                # If this isn't the first room, place the entry door
                if idx > 0:
                    self.grid.set(*room.entryDoorPos, None)

            if numRooms > 1:
                # Place the final goal in the last room
                self.goal_pos = self.place_obj(Goal(color=self._goal_color), roomList[-1].top, roomList[-1].size)
                self.place_agent(top=roomList[0].top, size=roomList[0].size)  # Randomize init position / direction

            else:  # when in single room setting
                _x, _y = roomList[0].top
                _x_offset, _y_offset = roomList[0].size
                # Place the final goal in the last room
                self.goal_pos = self.place_obj(obj=Goal(color=self._goal_color),
                                               top=roomList[-1].top,
                                               size=roomList[-1].size,
                                               pos=(_x + (_x_offset - 2), _y + (_y_offset - 2)))
                self.place_agent(top=roomList[0].top, size=roomList[0].size, pos=(_x + 1, _y + 1))

            for i in range(1, numRooms):
                entryPos = self.rooms[i].entryDoorPos
                if i == numRooms - 1:
                    endPos = self.goal_pos
                else:
                    endPos = self.rooms[i + 1].entryDoorPos
                self.distance.append(self._manhattan_distance(entryPos, endPos))
            for i in range(numRooms - 3, -1, -1):
                self.distance[i] += self.distance[i + 1]

            for x in range(self.width):
                for y in range(self.height):
                    # if x,y is in the room, generate mines
                    if self.is_in_room(x, y):
                        if self.grid.get(x, y) is None:
                            if x == self.start_pos[0] and y == self.start_pos[1]:
                                continue
                            rd = self._rand_float(0, 1.0)
                            if rd <= self._fullness:
                                mine_num = self._rand_int(0, len(self.mines))
                                mine = copy.deepcopy(self.mines[mine_num])
                                self.grid.set(x, y, mine)
                    # if x,y is not in the room, generate wall
                    else:
                        self.grid.set(x, y, wall)

            self.mission = 'traverse the rooms to get to the goal'
        else:
            # Refresh mine locations that have been overwritten by mine-change API
            # option1: reuse the exactly same mine locations
            self.grid = deepcopy(self._grid_original)

    def reset(self, if_on_launch: bool = False):
        self.accu_mining_reward = 0
        self.info_dict_for_render = dict()
        self.info_dict_for_render['action'] = None
        self.info_dict_for_render['reward'] = 0
        self.info_dict_for_render['ep_mining_reward'] = 0
        self.info_dict_for_render['ep_reward'] = 0
        super(MiningRoomEnv, self).reset(if_on_launch)
        if self._args['mw_rand_mine_category']:
            for x in range(self.width):
                for y in range(self.height):
                    grid = self.grid.get(x, y)
                    if grid is not None and grid.type == 'mine':
                        mine_num = self._rand_int(0, len(self.mines))
                        mine = copy.deepcopy(self.mines[mine_num])
                        self.grid.set(x, y, mine)
        if self._args['mw_rand_start_pos']:
            empty_grids_poses = []
            size = (1, 1)
            # if self._args['mw_start_from_first_room']:
            for x in range(self.rooms[0].top[0], self.rooms[0].top[0] + self.rooms[0].size[0]):
                for y in range(self.rooms[0].top[1] + self.rooms[0].size[1]):
                    if self.grid.get(x, y) is None:
                        empty_grids_poses.append((x, y))
            start_pos = self._rand_elem(empty_grids_poses)

            self.place_agent(start_pos, size)
            assert self.start_pos is not None
            assert self.start_dir is not None

            self.agent_pos = self.start_pos
            self.agent_dir = self.start_dir

        self.previous_distance_to_goal = self._min_distance_to_goal()
        self.max_distance_to_goal = self._min_distance_to_goal()

        # Return first observation
        obs = self.gen_obs()
        return obs

    def show_true_states(self):
        print(f"pos: {self.agent_pos}")
        print(f"dir: {self.agent_dir}")

    def gen_obs(self):
        if not self._args['mw_fully_observable']:
            return super(MiningRoomEnv, self).gen_obs()
        elif self._args['mw_obs_truth']:
            top = (self.rooms[0].top[0] + 1, self.rooms[0].top[1] + 1)
            size = (self.rooms[0].size[0] - 2, self.rooms[0].size[1] - 2)
            grid = self.grid.slice(top[0], top[1], size[0], size[1])
            pos = (self.agent_pos[0] - top[0], self.agent_pos[1] - top[1])
            return grid.encode_truth(agent_pos=pos, agent_dir=self.agent_dir)
        else:
            top = (self.rooms[0].top[0] + 1, self.rooms[0].top[1] + 1)
            size = (self.rooms[0].size[0] - 2, self.rooms[0].size[1] - 2)
            grid = self.grid.slice(top[0], top[1], size[0], size[1])
            pos = (self.agent_pos[0] - top[0], self.agent_pos[1] - top[1])
            return grid.encode_whole(agent_pos=pos, agent_dir=self.agent_dir)

    def render(self, mode='human', close=False):
        r = super().render(mode)
        fwd_pos = self.front_pos  # Get the position in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)  # Get the contents of the cell in front of the agent
        if fwd_cell is None:
            fwd_cell_name = 'empty'
        else:
            fwd_cell_name = fwd_cell.type
            if fwd_cell.type == 'mine':
                fwd_cell_name += str(fwd_cell.mine_id)

        if mode == 'rgb_array':
            return append_step_reward_to_image(
                image=r, step_r=self.info_dict_for_render['reward'], cum_r=self.info_dict_for_render['ep_reward'],
                mine_r=self.info_dict_for_render['ep_mining_reward'],
                action_embed=get_action_str(action=self.info_dict_for_render['action'], args=self._args),
                ts=self.step_count, cell_name=fwd_cell_name, if_goal_reached=self.goal_reached,
            )
        else:
            return r
