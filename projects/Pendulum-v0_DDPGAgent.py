from matplotlib.pyplot import get
import gym
from elegantrl.agents.agent import AgentDDPG
from elegantrl.train.config import get_gym_env_args, Arguments
from elegantrl.train.run import *

gym.logger.set_level(40)  # 阻止警告

get_gym_env_args(gym.make("Pendulum-v0"), if_print=True)

env_func = gym.make

env_args = {  # the information is given by the function get_gym_env_args
    'env_num': 1,
    'env_name': 'Pendulum-v0',
    'max_step': 200,
    'state_dim': 3,
    'action_dim': 1,
    'if_discrete': False,
    'target_return': 65536,
}

args = Arguments(AgentDDPG, env_func=env_func, env_args=env_args)

args.target_step=args.max_step
args.gamma=0.99
args.eval_times=2**5 # repeatedly update network to keep critic's loss small
args.random_seed=0

train_and_evaluate(args)