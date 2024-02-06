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
from stable_baselines3.common.evaluation import evaluate_policy


class MBPOAgent:
    def __init__(
        self,
        real_env: gym.Env,
        transi_mod: FullTransitionModel,
        done_mod: DoneModel,
        policy_optim: OffPolicyAlgorithm,
        length_model_rollouts: int = 1,
        gsteps: int = 40,
        nb_pol_optim_per_learned_model: int = 1,
        nb_model_rollout_per_optim: int = 400,
        nb_pol_update_per_optim: int = 1
    ):
        self.env = real_env
        self.transi = transi_mod
        self.done = done_mod
        self.agent = policy_optim
        self.k = length_model_rollouts
        self.gsteps = gsteps
        self.nb_optims = nb_pol_optim_per_learned_model
        assert self.nb_optims <= 1000, "max optim per model is 1000"
        self.optim_freq = 1000 // self.nb_optims
        self.nb_rollouts = nb_model_rollout_per_optim
        assert nb_pol_update_per_optim <= self.nb_rollouts, "nb pol update?optim sould be <= nb model rollout"
        self.update_freq = self.nb_rollouts // nb_pol_update_per_optim

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

        # Init Models #
        # if not self.separate_transi_r:
        self.transi.fit(self.S, self.A, self.R, self.Snext) # Separate ?
        self.done.fit(self.S, self.A, self.R, self.Snext, self.Term)
        self.model_env = make_env(self.env.observation_space, self.env.action_space, self.S, self.transi, self.done, self.k)

        # Init agent for first time #
        agent_kwargs = dict(
            policy="MlpPolicy",
            env=self.model_env,
            train_freq=(self.update_freq, "step"),
            gradient_steps=self.gsteps,
            learning_starts=0,
        )
        self.agent = self.agent(**agent_kwargs)

        for i in trange(iter):
            cum_r = 0
            # shoudl reset here
            s, _ = self.env.reset()
            for j in range(1000): # Real env steps #1000
                _, r, snext, term, trunc = self.add_new_transi(s)
                if ((j+1) % self.optim_freq) == 0:
                    self.model_env = make_env(self.env.observation_space, self.env.action_space, self.S, self.transi, self.done, self.k)
                    self.agent.learn(total_timesteps=self.nb_rollouts)
                cum_r += r
                if term or trunc:
                    s, _ = self.env.reset()
                    # self.evals.append(cum_r)
                    # self.times.append(time.time() - start)
                    cum_r = 0
                s = snext
            if i % 10 == 0:
                mn, std_ = evaluate_policy(self.agent, self.env)
                self.evals.append(mn)
                self.times.append(time.time() - start)
                print("Perf Real Env {}".format((mn, std_)))

            self.transi.fit(self.S, self.A, self.R, self.Snext) # Separate ?
            self.done.fit(self.S, self.A, self.R, self.Snext, self.Term)
            

    def save(self, fname):
        fname = "Experience_Results/" + fname
        os.makedirs(fname, exist_ok=True)
        np.savetxt(fname + "/times", self.times)
        np.savetxt(fname + "/evals", self.evals)
        self.agent.save(fname + "/policy")
        dump(self.transi, fname + "/transi.joblib")
        dump(self.done, fname + "/done.joblib")
