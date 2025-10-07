import numpy as np

from gym import utils
from large_rl.envs.reacher import mujoco_env
from large_rl.envs.wrapper import wrap_env

from gym.spaces import Box


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment is the cartpole environment based on the work done by
    Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can
    solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    just like in the classic environments but now powered by the Mujoco physics simulator -
    allowing for more complex experiments (such as varying the effects of gravity).
    This environment involves a cart that can moved linearly, with a pole fixed on it
    at one end and having another end free. The cart can be pushed left or right, and the
    goal is to balance the pole on the top of the cart by applying forces on the cart.

    ### Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |

    ### Observation Space

    The state space consists of positional values of different body parts of
    the pendulum system, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | anglular velocity (rad/s) |


    ### Rewards

    The goal is to make the inverted pendulum stand upright (within a certain angle limit)
    as long as possible - as such a reward of +1 is awarded for each timestep that
    the pole is upright.

    ### Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
    of [-0.01, 0.01] added to the values for stochasticity.

    ### Episode End
    The episode ends when any of the following happens:

    1. Truncation: The episode duration reaches 1000 timesteps.
    2. Termination: Any of the state space values is no longer finite.
    3. Termination: The absolutely value of the vertical angle between the pole and the cart is greater than 0.2 radian.

    ### Arguments

    No additional arguments are currently supported.

    ```
    env = gym.make('InvertedPendulum-v4')
    ```
    There is no v3 for InvertedPendulum, unlike the robot environments where a
    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.


    ### Version History

    * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum)
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        # MujocoEnv.__init__(
        #     self,
        #     "inverted_pendulum.xml",
        #     2,
        #     observation_space=observation_space,
        #     **kwargs
        # )
        self.first_step = True
        mujoco_env.MujocoEnv.__init__(self, model_path="inverted_pendulum.xml", frame_skip=2, **kwargs)

    def step(self, action):
        reward = 1.0
        action = self._transform_action(a=action)

        if_valid = 1
        if self.validity_type == "none" or self._check_valid_action(action) or self.first_step:
            self.do_simulation(action, self.frame_skip)
        else:
            action[:] = -1.
            self.do_simulation(action, self.frame_skip)
            if_valid = 0
        self.first_step = False
        self._update_variable_state()

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2))
        # ========= ORIGINAL
        # if self.render_mode == "human":
        #     self.render()
        # return observation, reward, terminated, False, info
        # ========= ORIGINAL
        if not if_valid:
            reward = -10
        return ob, reward, terminated, {"if_valid": if_valid}

    def reset_model(self):
        self._update_variable_state()
        self.first_step = True

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        ob = np.concatenate([self.data.qpos, self.data.qvel]).ravel()
        return self._include_variable_state_in_obs(ob=ob)

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


def create_environment(args: dict, seed):
    """Creates an interest evolution environment."""
    env = InvertedPendulumEnv(
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
