from mbpo import MBPOAgent as MBPO
from stable_baselines3 import SAC
from mbpo import TransitionMLPModel, RewardMLPModel, DoneMLPModel
import gymnasium as gym
from mbpo import make_env

real_env = gym.make("MountainCarContinuous-v0")
transi, reward, done = TransitionMLPModel(), RewardMLPModel(), DoneMLPModel()
mbpo = MBPO(real_env, transi, reward, done, make_env, SAC)
mbpo.learn(20)
folder_exp_name = "mountain_tree"
mbpo.save(folder_exp_name)
