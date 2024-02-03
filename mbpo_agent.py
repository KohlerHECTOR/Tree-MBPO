from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import numpy as np
import gymnasium as gym
from real_data_collection import collect_real_data, init_rng_data


class MBPOAgent:
    def __init__(
        self,
        real_env: gym.Env,
        transi_mod,
        reward_mod,
        done_mod,
        make_env_fn,
        policy_optim: OffPolicyAlgorithm,
    ):
        self.env = real_env
        self.transi = transi_mod
        self.reward = reward_mod
        self.done = done_mod
        self.make_env = make_env_fn
        self.agent = policy_optim

    def init_real_data(self):
        self.S, self.A, self.R, self.Snext, self.Term = init_rng_data(self.env)

    def add_new_real_data(self):
        new_S, new_A, new_R, new_Snext, new_Term, eval = collect_real_data(
            self.agent, self.env
        )
        self.S = np.concatenate((self.S, new_S), axis=1)
        self.A = np.concatenate((self.A, new_A), axis=1)
        self.R = np.concatenate((self.R, new_R), axis=1)
        self.Snext = np.concatenate((self.Snext, new_Snext), axis=1)
        self.Term = np.concatenate((self.Term, new_Term), axis=1)
        self.evals.append(eval)

    def learn(self, iter: int = 10):
        self.evals = []
        self.init_real_data()
        for i in range(iter):
            # Last arg is target
            self.transi.fit(self.S, self.A, self.Snext)
            self.reward.fit(self.S, self.A, self.Snext, self.R)
            self.done.fit(self.S, self.A, self.R, self.Snext, self.Term)

            self.model_env = self.make_env(
                self.env, self.S, self.transi, self.reward, self.done
            )
            if i < 1:
                self.agent = self.agent(
                    "MlpPolicy", self.model_env, learning_starts=0, gradient_steps=20
                )
            else:
                self.agent.env = self.model_env
            self.agent.learn(total_timesteps=1_000, progress_bar=True)
            self.add_new_real_data()
