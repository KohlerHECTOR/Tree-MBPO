from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3 import SAC, TD3
import numpy as np
import gymnasium as gym
from mbpo.real_data_collection import collect_real_data, init_rng_data
import time
import os
from tqdm.rich import trange
from joblib import dump
from mbpo.model_estimators import FullTransitionModel, DoneModel
from mbpo.model_env import make_env
import torch as th


class MBPOAgent:
    def __init__(
        self,
        real_env: gym.Env,
        transi_mod: FullTransitionModel,
        done_mod: DoneModel,
        policy_optim: OffPolicyAlgorithm,
        length_model_rollouts: int = 1,
    ):
        self.env = real_env
        self.transi = transi_mod
        self.done = done_mod
        self.agent = policy_optim
        self.k = length_model_rollouts

    def init_real_data(self):
        self.S, self.A, self.R, self.Snext, self.Term = init_rng_data(self.env)

    def add_new_transi(self, s: np.ndarray):
        action = self.agent.predict(th.FloatTensor(s.reshape(1, -1)), deterministic=False)[0][0]
        s_next, r, term, trunc, _ = self.env.step(action)
        return action, r, s_next, term, trunc
        
    def learn(self, iter: int = 10):
        self.evals = []
        self.times = []
        self.init_real_data()
        start = time.time()
        for i in trange(iter):
            self.transi.fit(self.S, self.A, self.R, self.Snext)
            self.done.fit(self.S, self.A, self.R, self.Snext, self.Term)
            self.model_env = make_env(
                self.env, self.S, self.transi, self.done, self.k
            )
            if i < 1:
                ### Init agent for first time ####
                agent_kwargs = dict(
                    policy="MlpPolicy",
                    env=self.model_env,
                    train_freq=(1, "step"),
                    gradient_steps=40
                )
                self.agent = self.agent(**agent_kwargs)
                ### Init agent for first time ####
            else:
                self.agent.env = self.model_env
            s, _ = self.env.reset()
            cum_r = 0
            for step in range(1000):
                a, r, snext, term, trunc = self.add_new_transi(s)
                cum_r += r
                self.S = np.concatenate((self.S, s.reshape(1, -1)))
                self.A = np.concatenate((self.A, a.reshape(1, -1)))
                self.R = np.concatenate((self.R, np.array(r).reshape(-1, 1)))
                self.Snext = np.concatenate((self.Snext, snext.reshape(1, -1)))
                self.Term = np.concatenate((self.Term, np.array(term, dtype=np.int8).reshape(-1, 1)))
                if term or trunc:
                    s, _ = self.env.reset()
                    self.evals.append(cum_r)
                    print("Perf Real Env {}".format(self.evals[-1]))
                    self.times.append(time.time() - start)
                    cum_r = 0
                

                self.model_env = make_env(self.env, self.S, self.transi, self.done, self.k)   
                self.agent.env = self.model_env

                self.agent.learn(total_timesteps=400)

                s = snext
                

    def save(self, fname):
        fname = "Experience_Results/" + fname
        os.makedirs(fname, exist_ok=True)
        np.savetxt(fname + "/times", self.times)
        np.savetxt(fname + "/evals", self.evals)
        self.agent.save(fname + "/policy")
        dump(self.transi, fname + "/transi.joblib")
        dump(self.done, fname + "/done.joblib")
