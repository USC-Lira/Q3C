from copy import deepcopy

from large_rl.embedding.base import BaseEmbedding
from large_rl.policy.agent import Random
from large_rl.policy.dqn import DQN
from large_rl.policy.ddpg import DDPG
from large_rl.policy.arddpg_cont import ARDDPG_CONT
from large_rl.policy.wolp import WOLP
from large_rl.policy.wolp_sac import WOLP as WOLP_SAC
from large_rl.envs.recsim.environments.interest_evolution_generic import create_vector_environment as recsim_env
from large_rl.envs.mining.wrapper_multienvs import create_vector_environment as mine_env
from large_rl.envs.recsim_data.env import launch_env as recsim_data_env


def launch_embedding(args: dict):
    emb = BaseEmbedding(num_embeddings=args["num_all_actions"],
                        dim_embed=args["recsim_num_categories"],
                        device=args["device"])
    dict_embedding = {"item": emb, "task": deepcopy(emb)}
    return dict_embedding


def launch_env(args: dict):
    if args["env_name"] == "recsim":
        env = recsim_env(args=args)
    elif args["env_name"] == "recsim-data":
        env = recsim_data_env(args=args)
    elif args["env_name"].lower() == "mine":
        env = mine_env(args=args)
    elif args["env_name"].startswith("mujoco"):
        from large_rl.envs.reacher.wrapper_multienvs import create_vector_environment as mujoco_env
        env = mujoco_env(args=args)
    else:
        raise ValueError
    return env


def launch_agent(args: dict, **kwargs):
    if args["agent_type"].lower() == "dqn":
        args['cql_alpha'] = 0.
        agent = DQN(args=args, **kwargs)
    elif args["agent_type"].lower() == "random":
        agent = Random(args=args, **kwargs)
    elif args["agent_type"].lower() == "wolp":
        agent = WOLP(args=args, **kwargs)
    elif args["agent_type"].lower() == "wolp-sac":
        agent = WOLP_SAC(args=args, **kwargs)
    elif args["agent_type"].lower() == "ddpg":
        agent = DDPG(args=args, **kwargs)
    elif args["agent_type"].lower() == "arddpg_cont":
        agent = ARDDPG_CONT(args=args, **kwargs)
    elif args["agent_type"].lower() == "greedy_ac":
        from large_rl.policy.wolp_greedyAC import GreedyWOLP
        agent = GreedyWOLP(args=args, **kwargs)
    elif args["agent_type"].lower() == 'sac':
        from sac_dir.stable_baselines3 import SAC
        agent = SAC("MlpPolicy", device=args["device"], gradient_steps=-1, **kwargs)
    else:
        raise ValueError
    return agent
