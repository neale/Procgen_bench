#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import gym3
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from .wrappers import make_atari, wrap_deepmind
from .wrappers import FrameStack as FrameStack_
from .vec_env import SubprocVecEnv, VecEnv

from .vec_env import VecExtractDictObs, VecMonitor, VecNormalize

from procgen import ProcgenEnv

from ..utils import *


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(
    env_id,
    seed,
    rank,
    episode_life=True,
    num_levels=200,
    dist='easy'):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            is_procgen = False
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id.startswith('procgen'):
            is_procgen = True
            env = gym.make(
                    env_id,
                    start_level=0,
                    num_levels=num_levels,
                    distribution_mode=dist)
        else:
            is_procgen = False
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
            env.seed(seed + rank)

        env = OriginalReturnWrapper(env)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            env = TransposeImage(env)
        if not is_procgen:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            env = FrameStack(env, 4)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        return


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 n_levels=200,
                 mode='easy',
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        
        envs = [make_env(name, seed, i, episode_life, n_levels, dist=mode) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))
        
        self.name = name

        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
