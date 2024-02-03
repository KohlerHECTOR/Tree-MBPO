from mbpo_agent import MBPOAgent as MBPO
from stable_baselines3 import SAC
from tree_model import TransitionTreeModel, RewardTreeModel, DoneTreeModel
import gymnasium as gym
from model_env import make_tree_env

real_env = gym.make("MountainCarContinuous-v0")
transi, reward, done = TransitionTreeModel(), RewardTreeModel(), DoneTreeModel()
mbpo = MBPO(real_env, transi, reward, done, make_tree_env, SAC)
mbpo.learn(5)
