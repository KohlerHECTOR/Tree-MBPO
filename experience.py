from mbpo import *
from stable_baselines3 import SAC, TD3
import gymnasium as gym
import sys


args = sys.argv[1:]

assert len(args) == 5, "python3 experience.py env_name estimator_cls pol_optim_cls nb_iter exp_name"
env_name = args[0]
iters = int(args[3])
exp_name = args[4]

if args[1] == "tree":
    transi = TransitionTreeModel()
    reward = RewardTreeModel()
    done = DoneTreeModel()

elif args[1] == "mlp":
    transi = TransitionMLPModel()
    reward = RewardMLPModel()
    done = DoneMLPModel()

else:
    AssertionError, "Only Model estimators are Decision Tree and MLP"


if args[2] == "sac":
    agent_cls = SAC

elif args[2] == "td3":
    agent_cls = TD3

else:
    AssertionError, "Only Pol Ooptim algos are SAC and TD3"

mbpo = MBPOAgent(gym.wrappers.time_limit.TimeLimit(gym.make(env_name), 1000), transi, reward, done, agent_cls)
mbpo.learn(iters)
mbpo.save(exp_name)
