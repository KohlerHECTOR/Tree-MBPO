from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3 import SAC, TD3
import numpy as np
import gymnasium as gym
from mbpo.real_data_collection import collect_real_data, init_rng_data
import time
import os
from tqdm.rich import trange
from joblib import dump
from mbpo.model_estimators import TransitionModel, RewardModel, DoneModel
from mbpo.model_env import make_env


class MBPOAgent:
    def __init__(
        self,
        real_env: gym.Env,
        transi_mod: TransitionModel,
        reward_mod: RewardModel,
        done_mod: DoneModel,
        policy_optim: OffPolicyAlgorithm,
        length_model_rollouts: int = 5,
        no_params: bool = True
    ):
        self.env = real_env
        self.transi = transi_mod
        self.reward = reward_mod
        self.done = done_mod
        self.agent = policy_optim
        self.k = length_model_rollouts
        self.no_params = no_params

    def init_real_data(self):
        self.S, self.A, self.R, self.Snext, self.Term = init_rng_data(self.env)

    def add_new_real_data(self):
        new_S, new_A, new_R, new_Snext, new_Term, eval = collect_real_data(
            self.agent, self.env
        )
        self.S = np.concatenate((self.S, new_S), axis=0)
        self.A = np.concatenate((self.A, new_A), axis=0)
        self.R = np.concatenate((self.R, new_R), axis=0)
        self.Snext = np.concatenate((self.Snext, new_Snext), axis=0)
        self.Term = np.concatenate((self.Term, new_Term), axis=0)
        self.evals.append(eval)

    def learn(self, iter: int = 10):
        self.evals = []
        self.times = []
        self.init_real_data()
        start = time.time()
        for i in trange(iter):
            # Last arg is target
            self.transi.fit(self.S, self.A, self.Snext)
            self.reward.fit(self.S, self.A, self.Snext, self.R)
            self.done.fit(self.S, self.A, self.R, self.Snext, self.Term)

            self.model_env = make_env(
                self.env, self.S, self.transi, self.reward, self.done, self.k
            )
            if i < 1:
                ### Init agent for first time ####
                if self.no_params:
                    agent_kwargs = dict(
                        policy="MlpPolicy",
                        env=self.model_env,
                        train_freq=(1, "step")
                    )
                
                else:
                    agent_kwargs = dict(
                        policy="MlpPolicy",
                        env=self.model_env,
                        learning_starts=256,
                        gradient_steps=32,
                        train_freq=128,
                        batch_size=16,
                    )

                    if self.agent == SAC:
                        agent_kwargs.update(dict(target_update_frequency=256))

                self.agent = self.agent(**agent_kwargs)
                ### Init agent for first time ####

            else:
                self.agent.env = self.model_env
                
            self.agent.learn(total_timesteps=256)
            self.add_new_real_data()

            print("Perf Real Env {}".format(self.evals[-1]))
            self.times.append(time.time() - start)

    def save(self, fname):
        fname = "Experience_Results/" + fname
        os.makedirs(fname, exist_ok=True)
        np.savetxt(fname + "/times", self.times)
        np.savetxt(fname + "/evals", self.evals)
        self.agent.save(fname + "/policy")
        dump(self.transi, fname + "/transi.joblib")
        dump(self.reward, fname + "/reward.joblib")
        dump(self.done, fname + "/done.joblib")
