from collections import OrderedDict
import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))

DEFAULT_SIZE = 500
EXTRA_DIMS = 6
NOISY_DIMS = BIJECTIVE_DIMS = 50

def get_radius(volume, dim):
    # Write the entire function for if conditions for dim == 1 to dim == 11 and return calculated radius
    # For dim > 11, use the formula for radius of a sphere in n dimensions
    # https://en.wikipedia.org/wiki/Volume_of_an_n-ball#The_volume
    if dim == 1:
        return volume / 2
    elif dim == 2:
        return np.sqrt(volume / np.pi)
    elif dim == 3:
        return np.cbrt(volume * 3 / 4 / np.pi)
    elif dim == 4:
        return np.sqrt(np.sqrt(volume * 3 / 4 / np.pi))
    elif dim == 5:
        return np.power(volume * 15 / 8 / np.pi ** 2, 1 / 5)
    elif dim == 6:
        return np.power(volume * 6, 1/6) / np.sqrt(np.pi)
    elif dim == 7:
        return np.power(volume * 105 / 16 / np.pi ** 3, 1 / 7)
    elif dim == 8:
        return np.power(volume * 24, 1/8) / np.sqrt(np.pi)
    elif dim == 9:
        return np.power(volume * 945 / 32 / np.pi ** 4, 1 / 9)
    elif dim == 10:
        return np.power(volume * 120, 1/10) / np.sqrt(np.pi)
    elif dim == 11:
        return np.power(volume * 10395 / 64 / np.pi ** 5, 1 / 11)
    else:
        return (np.pi * dim) ** (1 / (2 * dim)) * np.sqrt(dim / (2 * np.e * np.pi)) * volume ** (1 / dim)

def volume_sphere(radius, dim):
    if dim==1:
        return 2*radius
    elif dim==2:
        return np.pi*radius**2
    elif dim==3:
        return 4/3*np.pi*radius**3
    elif dim == 4:
        return np.pi**2/2*radius**4
    elif dim == 5:
        return 8/15*np.pi**2*radius**5
    elif dim == 6:
        return np.pi**3/6*radius**6
    elif dim == 7:
        return 16/105*np.pi**3*radius**7
    elif dim == 8: 
        return np.pi**4/24*radius**8
    elif dim == 9:
        return 32/945*np.pi**4*radius**9
    elif dim == 10:
        return np.pi**5/120*radius**10
    elif dim == 11:
        return 64/10395*np.pi**5*radius**11
    else:
        return 1 / np.sqrt(np.pi * dim) * (2 * np.pi * np.exp(1) / dim) ** (dim / 2) * radius ** dim

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, action_type='original', bijective_dims=10, validity_type="none", validity_box_size=0.05,
                 mujoco_env_box_seed=123, **kwargs):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.action_type = action_type
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # =========== new
        self.np_random = None
        self._dim_a_original = self.model.actuator_ctrlrange.copy().astype(np.float32).shape[0]

        self.bijective_dims = bijective_dims
        self.validity_type = validity_type

        self._prep_noisy_dims(action_type=action_type)
        self.variable_state = 0
        # =========== new

        self._set_action_space(action_type)

        n = self._dim_a_original

        self.num_boxes = n * 33 # Setting 100 as the num_boxes when dim = 3
        # total_volume = (2 * self.action_space.high[0]) ** n
        total_volume = (1) ** n
        # volume_per_box = 0.5 * total_volume / self.num_boxes # Assuming half of the volume is valid
        volume_per_box = 0.6 * n * total_volume / self.num_boxes # Assuming half of the volume is valid
        if n == 17:
            volume_per_box *= 20 # For humanoid
        elif n == 3:
            volume_per_box *= 1.65 # For hopper
        elif n == 1:
            volume_per_box *= 3 # For pendulums
        self.validity_box_size = get_radius(volume_per_box, n)
        # self.validity_box_size = volume_per_box ** (1 / n) / 2

        # self.validity_box_size = validity_box_size
        # self.num_boxes = 100
        self._set_valid_boxes(box_size=self.validity_box_size, num_boxes=self.num_boxes,
                              seed=mujoco_env_box_seed)

        action = self.action_space.sample()
        count = 0
        for _ in range(10000):
            action = self.action_space.sample()
            count += self._check_valid_action(action)
        print('Percentage of volume of actions that are valid: ', count // 100, '%') # Match 79% for dim=3 (hopper)
        observation, _reward, done, _info = self.step(action)
        # assert not done

        self._set_observation_space(observation)

        self.seed()

    def _update_variable_state(self):
        if self.np_random is not None and self.action_type == 'variable':
            self.variable_state = self.np_random.randint(2)

    def _include_variable_state_in_obs(self, ob):
        if self.action_type.startswith('variable'):
            return np.concatenate([ob, np.array([self.variable_state])], -1)
        else:
            return ob

    def _prep_noisy_dims(self, action_type):
        if action_type == 'bijective' or action_type.startswith('variable'):
            temp_rng = np.random.RandomState(123)
            self.dims = list()
            for _d in range(self._dim_a_original):
                self.dims.append(self._dim_a_original * temp_rng.rand(self.bijective_dims) - 1)

    def _transform_action(self, a):
        # import ipdb;ipdb.set_trace()
        # TODO: Implement discontinuous action space
        if self.action_type == 'original':
            action = a
        elif self.action_type == 'extra_dims':
            action = a[:self._dim_a_original]
        elif self.action_type == 'bijective':
            action = list()
            for _d in range(self._dim_a_original):
                action.append(
                    np.tanh(np.dot(a[self.bijective_dims * _d: self.bijective_dims * (_d + 1)], self.dims[_d]))
                )
            action = np.asarray(action)
        elif self.action_type.startswith('variable'):
            # Applies to both 'variable' and 'variable_reset'
            action = list()
            for _d in range(self._dim_a_original):
                if self.variable_state == 0:
                    # todo: check if this is correct
                    action.append(
                        np.tanh(np.dot(a[self.bijective_dims * _d: self.bijective_dims * (_d + 1)], self.dims[_d]))
                    )
                    # action = np.array([
                    #     np.tanh(np.dot(a[:self.bijective_dims], self.dim1)),
                    #     np.tanh(np.dot(a[self.bijective_dims:], self.dim2))
                    # ])
                else:
                    action.append(
                        np.tanh(np.dot(a[self.bijective_dims * (- _d): self.bijective_dims * (-_d + 1)], self.dims[_d]))
                    )
                    # action = np.array([
                    #     np.tanh(np.dot(a[self.bijective_dims:], self.dim1)),
                    #     np.tanh(np.dot(a[:self.bijective_dims], self.dim2))
                    # ])
            action = np.asarray(action)
        else:
            raise ValueError
        return action

    def _set_action_space(self, action_type='original'):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        if action_type == 'original':
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        elif action_type == 'extra_dims':
            extra_dims = np.ones(EXTRA_DIMS, dtype=np.float32)
            self.action_space = spaces.Box(
                low=np.concatenate([low, -1 * extra_dims]),
                high=np.concatenate([high, 1 * extra_dims]),
                dtype=np.float32)
        elif action_type == 'bijective' or action_type.startswith('variable'):
            dims = np.ones(self._dim_a_original * self.bijective_dims, dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1 * dims,
                high=1 * dims,
                dtype=np.float32)
        else:
            raise ValueError
        return self.action_space

    def _set_valid_boxes(self, box_size=0.05, num_boxes=100, seed=123):
        temp_rng = np.random.RandomState(seed)
        # self.box_centers = self.action_space.high[0] * (temp_rng.random(size=[num_boxes, *self.action_space.shape]) - 1) / 2
        self.box_centers = temp_rng.random(size=[num_boxes, *self.action_space.shape])
        self.box_size = box_size

    def _check_valid_action(self, action):
        if self.validity_type == "none":
            return True
        elif self.validity_type == "box":
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            low, high = bounds.T
            scaled_action = (action - low) / (high - low)
            # scaled_action = action
            if any(np.sqrt(((scaled_action - self.box_centers)**2).sum(-1)) < self.box_size):
            # if any(np.all(((action >= self.box_centers - self.box_size) & (action <= self.box_centers + self.box_size)), -1)):
                return True
            else:
                return False
        else:
            return True
        

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def set_seed(self, seed):
        self.seed(seed)
