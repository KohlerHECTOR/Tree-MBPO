import gymnasium as gym
import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from tqdm import rich
import torch as th


def init_rng_data(
    real_env: gym.Env,
    nb_data: int = 2048,
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


def collect_real_data(agent: OffPolicyAlgorithm, env: gym.Env, nb_trajs: int = 100):
    """collect real env transitions as seperate np arrays from the
    real model using policy trained on estimated model.

    Args:
        agent (OffPolicyAlgorithm): An SB3 off policy algorithm with actor
        env (gym.Env): Real environment model
        nb_trajs (int, optional): Number of trajectories on real env by agent.
        Defaults to 100.

    Returns:
        S, A, R, Snext, Term, evals: np arrays of transitions and mean cum reward of the agent.
    """
    S, A, R, Snext, Term = [], [], [], [], []
    avg_cum_r = 0
    for i in rich(range(nb_trajs)):
        s, _ = env.reset()
        done = False
        cum_r = 0
        while not done:
            with th.no_grad():
                mean_actions, _, _ = agent.policy.actor.get_action_dist_params(
                    th.FloatTensor(s.reshape(1, -1))
                )
            S.append(s)
            action = mean_actions[0].numpy()
            A.append(action)
            s_next, r, term, trunc, infos = env.step(action)
            R.append(r)
            cum_r += r
            Term.append(term)
            Snext.append(s_next)
            s = s_next
            done = term or trunc
        avg_cum_r += cum_r

    return (
        np.array(S),
        np.array(A),
        np.array(R),
        np.array(Snext),
        np.array(Term, dtype=np.int8),
        avg_cum_r / nb_trajs,
    )
