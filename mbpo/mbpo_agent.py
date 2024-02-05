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
        # no need for no_grad()
        action = self.agent.predict(th.FloatTensor(s.reshape(1, -1)), deterministic=False)[0][0]
        s_next, r, term, trunc, _ = self.env.step(action)

        self.S = np.concatenate((self.S, s.reshape(1, -1)))
        self.A = np.concatenate((self.A, action.reshape(1, -1)))
        self.R = np.concatenate((self.R, np.array(r).reshape(-1, 1)))
        self.Snext = np.concatenate((self.Snext, s_next.reshape(1, -1)))
        self.Term = np.concatenate((self.Term, np.array(term, dtype=np.int8).reshape(-1, 1)))
        return action, r, s_next, term, trunc
        
    def learn(self, iter: int = 10):
        self.evals = []
        self.times = []
        self.init_real_data()
        start = time.time()
        for i in trange(iter):
            self.transi.fit(self.S, self.A, self.R, self.Snext)
            self.done.fit(self.S, self.A, self.R, self.Snext, self.Term)
            self.model_env = make_env(self.env.observation_space, self.env.action_space, self.S, self.transi, self.done, self.k)   

            if i < 1:
                ### Init agent for first time ####
                agent_kwargs = dict(
                    policy="MlpPolicy",
                    env=self.model_env,
                    train_freq=(400, "step"), # 400
                    gradient_steps=40, # 40,
                    learning_starts=0,
                )
                self.agent = self.agent(**agent_kwargs)
                ### Init agent for first time ####
            else:
                self.agent.env = self.model_env
            # self.agent.gradient_steps += 1
            self.agent.learn(total_timesteps=400) #400


            cum_r = 0
            # shoudl reset here
            s, _ = self.env.reset()
            for j in range(1000): # Real env steps #1000
                _, r, snext, term, trunc = self.add_new_transi(s)
                cum_r += r
                if term or trunc:
                    s, _ = self.env.reset()
                    self.evals.append(cum_r)
                    self.times.append(time.time() - start)
                    cum_r = 0
                # sanity check
                # for param in self.agent.policy.actor.latent_pi.parameters():
                #     continue
                # print(param[0])
                s = snext
            print("Perf Real Env {}".format(self.evals[-1]))
            

    def save(self, fname):
        fname = "Experience_Results/" + fname
        os.makedirs(fname, exist_ok=True)
        np.savetxt(fname + "/times", self.times)
        np.savetxt(fname + "/evals", self.evals)
        self.agent.save(fname + "/policy")
        dump(self.transi, fname + "/transi.joblib")
        dump(self.done, fname + "/done.joblib")
