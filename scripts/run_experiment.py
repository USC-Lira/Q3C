from functools import reduce
import operator
import os
import gymnasium as gym
import numpy as np
import datetime
import imageio
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
import gymnasium_robotics

from q3c.model import Q3C
from q3c.utils.learning_rate_schedulers import delayed_exponential_schedule, exponential_schedule, linear_schedule, one_cycle_lr_schedule, cosine_annealing_schedule


class SuccessWrapper(gym.Wrapper):
    """Wrapper that adds is_success to info dict for Adroit environments."""
    
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add is_success based on success from info
        info['is_success'] = info['success']
        return obs, reward, terminated, truncated, info


@hydra.main(version_base=None, config_path="../configs", config_name="hydra_sb3_q3c_hyperparams.yaml")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gym.register_envs(gymnasium_robotics)
    # Convert Hydra config to a standard Python dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_params = cfg_dict["wandb"]
    # Select the environment and extract parameters
    env_name = cfg_dict["train"]["environment"]
    env_hyperparameters = cfg_dict["environment"][env_name]

    num_envs = cfg_dict["train"].pop("num_envs")

    if("Box" in env_name):
        boxed_env_kwargs = env_hyperparameters.pop("env_kwargs", {})
        
        if("InvertedDoublePendulumBox" in env_name):
            env_id = 'InvertedDoublePendulumRestricted-v0'
            register(
                id=env_id,
                entry_point='large_rl.envs.reacher.inverted_double_pendulum_v4_gymnasium:InvertedDoublePendulumEnv',
            )
        elif("InvertedPendulumBox" in env_name):
            env_id = 'InvertedPendulumRestricted-v0'
            register(
                id=env_id,
                entry_point='large_rl.envs.reacher.inverted_pendulum_v4_gymnasium:InvertedPendulumEnv',
            )
        elif("HopperBox" in env_name):
            env_id = 'HopperRestricted-v0'
            register(
                id=env_id,
                entry_point='large_rl.envs.reacher.hopper_v4_gymnasium:HopperEnv',
            )
        elif("HalfCheetahBox" in env_name):
            env_id = 'HalfCheetahRestricted-v0'
            register(
                id=env_id,
                entry_point='large_rl.envs.reacher.half_cheetah_v4_gymnasium:HalfCheetahEnv',
            )
        elif("SwimmerBox" in env_name):
            env_id = 'SwimmerRestricted-v0'
            register(
                id=env_id,
                entry_point='large_rl.envs.reacher.swimmer_v4_gymnasium:SwimmerEnv',
            )
        elif("Walker2dBox" in env_name):
            env_id = 'Walker2dRestricted-v0'
            register(
                id=env_id,
                entry_point='large_rl.envs.reacher.walker2d_v4_gymnasium:Walker2dEnv',
            )
        elif("AntBox" in env_name):
            env_id = 'AntRestricted-v0'
            register(
                id=env_id,
                entry_point='large_rl.envs.reacher.ant_v4_gymnasium:AntEnv',
            )

        env = make_vec_env(env_id, n_envs=num_envs, env_kwargs={
                'action_type': boxed_env_kwargs.get("reacher_action_type", 'original'),
                'max_episode_steps': boxed_env_kwargs.get("max_episode_steps", 1000),
                'bijective_dims': boxed_env_kwargs.get("reacher_bijective_dims", 5),
                'validity_type': boxed_env_kwargs.get("validity_type", 'none'),
                'mujoco_env_box_seed': boxed_env_kwargs.get("mujoco_env_box_seed", 123),
                'valid_box_volume_multiplier': env_hyperparameters.get('valid_box_volume_multiplier', None),
                 },
                wrapper_class=TimeLimit,
                wrapper_kwargs={'max_episode_steps': boxed_env_kwargs.get("max_episode_steps", 1000)}
                )

        eval_env = gym.make(env_id, **{
                    'action_type': boxed_env_kwargs.get("reacher_action_type", 'original'),
                    'max_episode_steps': boxed_env_kwargs.get("max_episode_steps", 1000),
                    'bijective_dims': boxed_env_kwargs.get("reacher_bijective_dims", 5),
                    'validity_type': boxed_env_kwargs.get("validity_type", 'none'),
                    'mujoco_env_box_seed': boxed_env_kwargs.get("mujoco_env_box_seed", 123),
                    'valid_box_volume_multiplier': env_hyperparameters.get('valid_box_volume_multiplier', None), 
                })

        n_eval_episodes = 10
    else:
        env = make_vec_env(env_name, n_envs=num_envs)
        eval_env = gym.make(env_name, render_mode='rgb_array')
        n_eval_episodes = 10
        if "Adroit" in env_name:
            eval_env = SuccessWrapper(eval_env)
            n_eval_episodes = 50

    eval_env = Monitor(eval_env)
    if("Box" in env_name): 
        eval_env = TimeLimit(eval_env, max_episode_steps=boxed_env_kwargs.get("max_episode_steps", 1000))
        
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{env_name}", exist_ok=True)
    task_folder = f"results/{env_name}/q3c"

    experiment_flags = {}
    for key in cfg_dict["train"]:
        if key not in ['save_gif', 'disable_wandb', 'environment']: 
            experiment_flags[key] = cfg_dict["train"][key]

    callback_freq = (env_hyperparameters['n_timesteps'] // num_envs) // 100
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPERIMENT_NAME = f"q3c_{env_name}_{timestamp}"
    random_id = np.random.randint(0, 100000)
    EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_{random_id}"
    wandb_hyperparameters = {
        'env_name': env_name,
        'algorithm': 'Q3C',
        'num_envs': num_envs,
        'valid_box_volume_multiplier': env_hyperparameters.pop('valid_box_volume_multiplier', None),
        **env_hyperparameters,
        **experiment_flags
    }
    if "Box" in env_name:
        for key, value in boxed_env_kwargs.items():
            wandb_hyperparameters[key] = value

    tags = [env_name, "q3c"]
    initial_smoothing_value = env_hyperparameters.pop('smoothing_value', 0.1)

    run = wandb.init(entity=wandb_params["entity"],
                     project=wandb_params["project"],
                     name=EXPERIMENT_NAME,
                     config=wandb_hyperparameters,
                     tags=tags,
                     sync_tensorboard=True,
                     monitor_gym=False,
                     save_code=False,
                     mode="disabled" if cfg_dict["train"]["disable_wandb"] else "online")
    # The noise objects for Q3C
    noise_type = env_hyperparameters.pop('noise_type')
    noise_std = env_hyperparameters.pop('noise_std')

    if noise_type == 'normal':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    elif noise_type == 'ornstein_uhlenbeck':
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=noise_std * np.ones(1))
    else:
        raise ValueError("Invalid noise type")

    total_timesteps = env_hyperparameters.pop('n_timesteps')
    policy_kwargs = eval(env_hyperparameters['policy_kwargs'])
    env_hyperparameters.pop('policy_kwargs')
    policy_kwargs['num_control_points'] = env_hyperparameters.pop('num_control_points')
    policy_kwargs['k'] = env_hyperparameters.pop('k')
    policy_kwargs['separation_loss_function'] = env_hyperparameters.pop('separation_loss_function', 'repulsion_loss')
    policy_kwargs['separation_loss_coefficient'] = env_hyperparameters.pop('separation_loss_coefficient', 0.01)
    lr_scheduler_function = experiment_flags.pop('lr_scheduler_function')
    if experiment_flags.pop('use_lr_schedule'):
        if lr_scheduler_function == 'exponential': 
            env_hyperparameters['learning_rate'] = exponential_schedule(env_hyperparameters['learning_rate'], total_timesteps, min(total_timesteps, 1000000))
        elif lr_scheduler_function == 'linear': 
            env_hyperparameters['learning_rate'] = linear_schedule(env_hyperparameters['learning_rate'])
        elif lr_scheduler_function == 'one_cycle_lr': 
            env_hyperparameters['learning_rate'] = one_cycle_lr_schedule(env_hyperparameters['learning_rate'])
        elif lr_scheduler_function == 'cosine': 
            env_hyperparameters['learning_rate'] = cosine_annealing_schedule(env_hyperparameters['learning_rate'])
        elif lr_scheduler_function == 'delayed_exponential':
            env_hyperparameters['learning_rate'] = delayed_exponential_schedule(env_hyperparameters['learning_rate'])
        else:
            raise ValueError(f"Invalid learning rate scheduler function: {lr_scheduler_function}")
    for key, value in experiment_flags.items():
        policy_kwargs[key] = value
    policy_kwargs['smoothing_value'] = initial_smoothing_value

    callback_list = []
    if not cfg_dict["train"]["disable_wandb"]:
        wandb_callback = WandbCallback()
        callback_list.append(wandb_callback)
    log_folder = task_folder + "/" + EXPERIMENT_NAME
    checkpoint_callback = CheckpointCallback(save_freq=total_timesteps, save_path=task_folder, name_prefix=f'q3c_{env_name}')
    eval_callback = EvalCallback(eval_env,
                                 log_path=log_folder, eval_freq=callback_freq, n_eval_episodes=n_eval_episodes,
                                 deterministic=True, render=False)
    callback_list.append(checkpoint_callback)
    callback_list.append(eval_callback)

    model = Q3C(env=env,
                action_noise=action_noise,
                device=device,
                policy_kwargs=policy_kwargs,
                **env_hyperparameters)
    new_logger = configure(log_folder, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    print("created model and logger")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward before training: {mean_reward} +/- {std_reward}")

    model.learn(total_timesteps=total_timesteps, log_interval=500, callback=callback_list)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward after training: {mean_reward} +/- {std_reward}")

    if cfg_dict["train"]["save_gif"]:
        os.makedirs("videos", exist_ok=True)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        for eps in range(5):
            obs, _ = eval_env.reset()
            frames = []
            score = 0
            for i in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = eval_env.step(action)
                img = eval_env.render()
                frames.append(img)
                score += rewards
                if terminated or truncated:
                    break
            imageio.mimsave(f'videos/{env_name}/q3c_episode_{eps}.gif', frames, duration=5)

    run.finish()
    print("Experiment finished")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
