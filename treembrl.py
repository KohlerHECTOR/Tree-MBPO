from tqdm import tqdm
from stable_baselines3 import SAC
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import gymnasium as gym
import torch as th
from stable_baselines3.common.env_util import make_vec_env

def collect_data(agent, env, nb_trajs = 100):
    S, A, R, Snext, Term= [], [], [], [], []
    avg_cum_r = 0
    for i in tqdm(range(nb_trajs)):
        s,_ = env.reset()
        done = False
        cum_r = 0
        while not done:

            with th.no_grad():
                mean_actions, _, _ = agent.policy.actor.get_action_dist_params(th.FloatTensor(s.reshape(1,-1)))
            S.append(s)
            action = mean_actions.numpy()
            A.append(action[0])
            s_next, r, term, trunc, infos = env.step(action[0])
            R.append(r)
            cum_r += r
            Term.append(term)
            Snext.append(s_next)
            s = s_next
            done = term or trunc
        avg_cum_r += cum_r
    
    return np.array(S), np.array(A),  np.array(R), np.array(Snext), np.array(Term), avg_cum_r/nb_trajs


def fit_tree_models(s, a, r, snext, term):
    sa = np.concatenate((s, a), axis = 1)
    concat_transi = np.concatenate((sa, r.reshape(-1,1), snext), axis = 1)

    transi_model = DecisionTreeRegressor(max_leaf_nodes=1024 ,random_state=42) # is hyperparam
    transi_model.fit(sa, snext)
    reward_model = DecisionTreeRegressor(max_leaf_nodes=1024,random_state=42)
    reward_model.fit(sa, r)
    term_model = DecisionTreeClassifier(max_leaf_nodes=1024,random_state=42) # Should be perfect
    # imbalanced data of done not done
    term_model.fit(concat_transi, term)

    return transi_model, reward_model, term_model

class GymTreeModel(gym.Env):
    def __init__(self, model_tree, reward_tree, done_tree, obs_space, action_space, real_states_buffer, render_mode=None):
        self.observation_space, self.action_space = obs_space, action_space
        self.model_tree = model_tree
        self.reward_tree = reward_tree
        self.done_tree = done_tree
        self.real_states = real_states_buffer
        self.sstate = None
        self.nb_steps = 0

    def reset(self, **kwargs):
        # sample real start states at random ?
        start_state_idx = np.random.choice(np.arange(len(self.real_states)))
        self.state = self.real_states[start_state_idx]
        self.nb_steps = 0
        return self.state, {}
    
    def step(self, action):
        term, trunc = False, False
        sa = np.concatenate((self.state, action))
        snext = self.model_tree.predict([sa])[0]
        r = self.reward_tree.predict([sa])
        sarsn = np.concatenate((sa, r, snext))
        term = self.done_tree.predict([sarsn])[0]
        self.nb_steps += 1
        if self.nb_steps >= 2: # Is hyperparam
            trunc = True
        self.state = snext
        return self.state, r[0], term, trunc, {}
    



if __name__ == "__main__":
    ## Hopper-v4
    print("Getting random real env transitions")
    real_env = gym.make("Hopper-v4")
    s, _ = real_env.reset()
    S = np.zeros((10_000, real_env.observation_space.shape[0]))
    A = np.zeros((10_000, real_env.action_space.shape[0]))
    R = np.zeros((10_000, 1))
    SN = np.zeros((10_000, real_env.observation_space.shape[0]))
    Term = np.zeros((10_000, 1), dtype=np.int8)
    for i in range(10_000):
        S[i] = s
        A[i] = real_env.action_space.sample()
        SN[i], R[i], Term[i], trunc, _ = real_env.step(A[i])
        if Term[i]:
            s, _ = real_env.reset()
        else:
            s = SN[i]

    print("Fitting tree model")
    transi_tree, reward_tree, done_tree = fit_tree_models(S, A, R, SN, Term)

    env_kwargs = dict(
            model_tree=transi_tree, 
            reward_tree=reward_tree, 
            done_tree=done_tree, 
            obs_space=real_env.observation_space, 
            action_space=real_env.action_space, 
            real_states_buffer=S
        )
    env = make_vec_env(GymTreeModel, env_kwargs=env_kwargs, n_envs=8)
    agent = SAC(env=env, policy="MlpPolicy", learning_starts=0)
    print("Training sac on Tree models")
    agent.learn(10_000, progress_bar=True)

    for i in range(9):
        print("Getting real env transitions with sac policy")
        S_new, A_new, R_new, SN_new, Term_new, evals = collect_data(agent, real_env)
        print("On real env, sac trained on models gets :{}".format(evals))
        S = np.concatenate((S_new, S))
        A = np.concatenate((A_new, A))
        R = np.concatenate((R_new.reshape(-1,1), R))
        SN = np.concatenate((SN_new, SN))
        Term = np.concatenate((Term_new.reshape(-1,1), Term))
        print("Fitting new tree models")
        transi_tree, reward_tree, done_tree = fit_tree_models(S, A, R, SN, Term)
        env_kwargs = dict(
            model_tree=transi_tree, 
            reward_tree=reward_tree, 
            done_tree=done_tree, 
            obs_space=real_env.observation_space, 
            action_space=real_env.action_space, 
            real_states_buffer=S
        )
        env = make_vec_env(GymTreeModel, env_kwargs=env_kwargs, n_envs=8)
        agent.env = env
        print("Training sac on Tree models")
        # TODO: only train on most recent tree models data ?
        agent.learn(10_000, progress_bar=True)