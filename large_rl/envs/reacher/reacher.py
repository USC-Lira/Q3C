import numpy as np
from gym import utils
from large_rl.envs.reacher import mujoco_env
from large_rl.envs.wrapper import wrap_env

WINDOW_SIZE = 400
'''
    Other ideas:
        1. just add noise to the agent's action
        2. Adversarial Environment: Penalize taking the same action for the same state.
        3. Multimodal action space, where one of them is randomly better?


        4. Introduce completely useless action dimensions or make bijective ignore completely some dimensions.

'''


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, action_type='original', max_episode_steps=250, bijective_dims=10, validity_type="none",
                 validity_box_size=0.05, **kwargs):
        utils.EzPickle.__init__(self)
        self.curr_step = 0
        self.action_type = action_type
        self.max_episode_steps = max_episode_steps
        mujoco_env.MujocoEnv.__init__(self, model_path='reacher.xml', frame_skip=2, action_type=action_type,
                                      bijective_dims=bijective_dims, validity_type=validity_type,
                                      validity_box_size=validity_box_size, **kwargs)

    def step(self, action):
        self.curr_step += 1
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        action = self._transform_action(a=action)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl
        success = -reward_dist < 0.02
        if_valid = 1
        if self.validity_type == "none" or self._check_valid_action(action):
            self.do_simulation(action, self.frame_skip)
        else:
            self.do_simulation(np.zeros_like(action), self.frame_skip)
            if_valid = 0
        self._update_variable_state()
        ob = self._get_obs()
        done = False
        if not if_valid:
            reward = -10
        return ob, reward, done, dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            success=success,
            if_valid=if_valid)

    def step_sim_before(self, action):
        self.curr_step += 1
        action = self._transform_action(a=action)
        if self.validity_type == "none" or self._check_valid_action(action):
            self.do_simulation(action, self.frame_skip)
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl
        success = -reward_dist < 0.02
        self._update_variable_state()
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, success=success)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.curr_step = 0
        self._update_variable_state()
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        ob = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
        return self._include_variable_state_in_obs(ob=ob)

    # def render(self,
    #            mode='human',
    #            width=WINDOW_SIZE,
    #            height=WINDOW_SIZE,
    #            camera_id=None,
    #            camera_name=None):
    #     if mode == 'rgb_array' or mode == 'depth_array':
    #         if camera_id is not None and camera_name is not None:
    #             raise ValueError("Both `camera_id` and `camera_name` cannot be"
    #                              " specified at the same time.")
    #
    #         no_camera_specified = camera_name is None and camera_id is None
    #         if no_camera_specified:
    #             camera_name = 'track'
    #
    #         if camera_id is None and camera_name in self.model._camera_name2id:
    #             camera_id = self.model.camera_name2id(camera_name)
    #
    #         self._get_viewer(mode).render(width, height, camera_id=camera_id)
    #
    #     if mode == 'rgb_array':
    #         # window size used for old mujoco-py:
    #         data = self._get_viewer(mode).read_pixels(width, height, depth=False)
    #         # original image is upside-down, so flip it
    #         return data[::-1, :, :]
    #     elif mode == 'depth_array':
    #         self._get_viewer(mode).render(width, height)
    #         # window size used for old mujoco-py:
    #         # Extract depth part of the read_pixels() tuple
    #         data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
    #         # original image is upside-down, so flip it
    #         return data[::-1, :]
    #     elif mode == 'human':
    #         self._get_viewer(mode).render()

    def render(self, mode='human', annotation_flg=False, width=WINDOW_SIZE, height=WINDOW_SIZE):
        # import pudb; pudb.start()
        if self.viewer and annotation_flg:
            del self.viewer._markers[:]

        if mode == 'rgb_array':
            camera_id = None
            camera_name = 'track'
            # if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
            if camera_name in self.model.camera_names:
                camera_id = self.model.camera_name2id(camera_name)
            if self.viewer and annotation_flg:
                self.add_joints_markers()
            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            if self.viewer and annotation_flg:
                self.add_joints_markers()
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            if self.viewer and annotation_flg:
                self.add_joints_markers()
            self._get_viewer(mode).render()

    def add_joints_markers(self):
        for actuator_name in self.sim.model.actuator_names:
            actuator_id = self.sim.model.actuator_name2id(actuator_name)
            actuator_pos = self.sim.data.get_joint_xanchor(actuator_name)
            self.viewer.add_marker(pos=actuator_pos,
                                   label="%s: %.5f" % (actuator_name, self.sim.data.ctrl[actuator_id - 1]),
                                   size=[0, 0, 0])  # this removes the box! so don't change this...

    def set_path(self, dir_path):
        self.save_path = dir_path

    def video_saving(self):
        now = datetime.now()
        if self.save_path is None:
            self.save_path = osp.join('./data/reacher_video/',
                                      self._args['env_name'])
        # vid_dir += '_' + dt_string1
        dt_string2 = now.strftime("%Y_%m_%d_%H_%M_%S")
        name = "{b}_{c}".format(b=dt_string2, c=self._rand_int(0, 1000))
        save_mp4(frames=self.rgb_frames, vid_dir=self.save_path, name=name, no_frame_drop=True)
        self.rgb_frames = None


def create_environment(args: dict, seed):
    """Creates an interest evolution environment."""
    env = ReacherEnv(
        # max_episode_steps=50,
        # reward_threshold=-3.75, args=args
        action_type=args['reacher_action_type'],
        max_episode_steps=args['max_episode_steps'],
        bijective_dims=args['reacher_bijective_dims'],
        validity_type=args['reacher_validity_type'],
        mujoco_env_box_seed=args['mujoco_env_box_seed'],
    )

    env = wrap_env(env=env, args=args)
    args['reacher_obs_space'] = env.observation_space.shape[0]
    args['reacher_action_shape'] = env.action_space.shape[0]
    args['reacher_action_space'] = env.action_space
    return env
