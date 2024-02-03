import gymnasium as gym
from tree_model import DoneTreeModel, RewardTreeModel, TransitionTreeModel
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


class GymTreeModel(gym.Env):
    def __init__(
        self,
        model_tree: TransitionTreeModel,
        reward_tree: RewardTreeModel,
        done_tree: DoneTreeModel,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        real_states_buffer: np.ndarray,
        rollout_length: int = 5,
    ):
        self.observation_space, self.action_space = obs_space, action_space
        self.model_tree = model_tree
        self.reward_tree = reward_tree
        self.done_tree = done_tree
        self.real_states = real_states_buffer
        self.sstate = None
        self.nb_steps = 0
        self.k = rollout_length

    def reset(self, **kwargs):
        start_state_idx = np.random.choice(np.arange(len(self.real_states)))
        self.state = self.real_states[start_state_idx]
        self.nb_steps = 0
        return self.state, {}

    def step(self, action):
        term, trunc = False, False
        snext = self.model_tree.predict(self.state, action)
        r = self.reward_tree.predict(self.state, action, snext)
        term = self.done_tree.predict(self.state, action, r, snext)
        self.nb_steps += 1
        if self.nb_steps >= self.k:  # Is hyperparam
            trunc = True
        self.state = snext
        return self.state, r, term, trunc, {}


def make_tree_env(
    real_env: gym.Env,
    real_states: np.ndarray,
    transi_tree: TransitionTreeModel,
    reward_tree: RewardTreeModel,
    done_tree: DoneTreeModel,
):
    env_kwargs = dict(
        model_tree=transi_tree,
        reward_tree=reward_tree,
        done_tree=done_tree,
        obs_space=real_env.observation_space,
        action_space=real_env.action_space,
        real_states_buffer=real_states,
    )
    return make_vec_env(GymTreeModel, env_kwargs=env_kwargs, n_envs=8)
