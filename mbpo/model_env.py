import gymnasium as gym
from mbpo.model_estimators import DoneModel, FullTransitionModel
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


class GymModel(gym.Env):
    def __init__(
        self,
        transi: FullTransitionModel,
        done: DoneModel,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        real_states_buffer: np.ndarray,
        rollout_length: int = 5,
    ):
        self.observation_space, self.action_space = obs_space, action_space
        self.transi = transi
        self.done = done
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
        r, snext = self.transi.predict(self.state, action)
        term = self.done.predict(self.state, action, r, snext)
        self.nb_steps += 1
        if self.nb_steps >= self.k:  # Is hyperparam
            trunc = True
        self.state = snext
        return self.state, r, term, trunc, {}


def make_env(
    observation_space,
    action_space,
    real_states: np.ndarray,
    transi: FullTransitionModel,
    done: DoneModel,
    rollout_length: int = 1,
):
    env_kwargs = dict(
        transi=transi,
        done=done,
        obs_space=observation_space,
        action_space=action_space,
        real_states_buffer=real_states,
        rollout_length=rollout_length,
    )
    return make_vec_env(GymModel, env_kwargs=env_kwargs, n_envs=1)
