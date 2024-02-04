from mbpo import *
from stable_baselines3 import SAC, TD3
import gymnasium as gym
import sys


args = sys.argv[1:]
env_name = args[0]
iters = int(args[2])
exp_name = args[3]

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



mbpo = MBPOAgent(gym.make(env_name), transi, reward, done, SAC)
mbpo.learn(iters)
mbpo.save(exp_name)
