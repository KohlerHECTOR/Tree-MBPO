import gymnasium as gym
import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import torch as th
from statistics import mean


def init_rng_data(
    real_env: gym.Env,
    nb_data: int = 1000,
):
    s, _ = real_env.reset()
    S = np.zeros((nb_data, real_env.observation_space.shape[0]))
    A = np.zeros((nb_data, real_env.action_space.shape[0]))
    R = np.zeros((nb_data, 1))
    SN = np.zeros((nb_data, real_env.observation_space.shape[0]))
    Term = np.zeros((nb_data, 1), dtype=np.int8)
    for i in range(nb_data):
        S[i] = s
        A[i] = real_env.action_space.sample()
        SN[i], R[i, 0], Term[i, 0], trunc, _ = real_env.step(A[i])
        if Term[i, 0]:
            s, _ = real_env.reset()
        else:
            s = SN[i]
    return S, A, R, SN, Term


def collect_real_data(
    agent: OffPolicyAlgorithm, env: gym.Env, nb_steps: int = 1000, deterministic=False
):
    """collect real env transitions as seperate np arrays from the
    real model using policy trained on estimated model.

    Args:
        agent (OffPolicyAlgorithm): An SB3 off policy algorithm with actor
        env (gym.Env): Real environment model
        nb_steps

    Returns:
        S, A, R, Snext, Term, evals: np arrays of transitions and mean cum reward of the agent.
    """
    S = np.zeros((nb_steps, env.observation_space.shape[0]))
    A = np.zeros((nb_steps, env.action_space.shape[0]))
    R = np.zeros((nb_steps, 1))
    Snext = np.zeros((nb_steps, env.observation_space.shape[0]))
    Term = np.zeros((nb_steps, 1), dtype=np.int8)
    avg_cum_r = []
    stp = 0
    s, _ = env.reset()
    done = False
    cum_r = 0
    while stp < nb_steps:
        with th.no_grad():
            mean_actions = agent.predict(
                th.FloatTensor(s.reshape(1, -1)), deterministic=deterministic
            )[0]
        S[stp] = s
        action = mean_actions[0]
        A[stp] = action
        s_next, r, term, trunc, infos = env.step(action)
        R[stp, 0] = r
        Term[stp, 0] = term
        Snext[stp] = s_next
        cum_r += r
        stp += 1
        s = s_next
        done = term or trunc
        if done:
            avg_cum_r.append(cum_r)
            cum_r = 0
            s, _ = env.reset()

    return (S, A, R, Snext, Term, mean(avg_cum_r))
