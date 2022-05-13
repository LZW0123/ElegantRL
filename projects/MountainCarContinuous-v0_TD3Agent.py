from matplotlib.pyplot import get
import gym
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.train.config import get_gym_env_args, Arguments
from elegantrl.train.run import *

gym.logger.set_level(40)  # 阻止警告

get_gym_env_args(gym.make("MountainCarContinuous-v0"), if_print=True)

env_func = gym.make

env_args = {
    'env_num': 1,
    'env_name': 'MountainCarContinuous-v0',
    'max_step': 999,
    'state_dim': 2,
    'action_dim': 1,
    'if_discrete': False,
    'target_return': 90.0,
}


args = Arguments(AgentTD3, env_func=env_func, env_args=env_args)

args.target_step=args.max_step
args.gamma=0.99
args.eval_times=2**4 # repeatedly update network to keep critic's loss small
args.random_seed=0

train_and_evaluate(args)