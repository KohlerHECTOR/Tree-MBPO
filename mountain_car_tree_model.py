from mbpo.mbpo_agent import MBPOAgent as MBPO
from stable_baselines3 import SAC
from mbpo.model_estimators import TransitionTreeModel, RewardTreeModel, DoneTreeModel
import gymnasium as gym
from mbpo.model_env import make_env

real_env = gym.make("MountainCarContinuous-v0")
transi, reward, done = TransitionTreeModel(), RewardTreeModel(), DoneTreeModel()
mbpo = MBPO(real_env, transi, reward, done, make_env, SAC)
mbpo.learn(20)
folder_exp_name = "mountain_tree"
mbpo.save(folder_exp_name)

