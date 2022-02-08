import os
from copy import copy

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper, VecMonitor)


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        elif "Montezuma" in env_id:
            env = MontezumaInfoWrapper(env)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = EpisodicLifeEnv(env)
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WarpFrame(env, width=84, height=84)
                env = ClipRewardEnv(env)
                obs_shape = env.observation_space.shape

                if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                    env = TransposeImage(env, op=[2, 0, 1])
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError

        return env

    return _thunk


def make_vec_envs(config):
    log_dir = config['log_dir'] + '\\' + config['run_id'] if 'log_dir' in config else 'logs\\' + config['run_id']
    envs = [
        make_env(config['env']['env_name'], config['env']['seed'], i)
        for i in range(config['env']['num_process'])
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    envs = VecMonitor(envs, os.path.join(log_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    envs = VecPyTorch(envs, device)

    if len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)
    elif 'obs_stack' in config['env']:
        envs = VecPyTorchFrameStack(envs, config['env']['obs_stack'], device)

    return envs


def make_play_env(config):
    env = [
        make_env(config['env']['env_name'], 3, 0)
    ]

    env = DummyVecEnv(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = VecPyTorch(env, device)
    if len(env.observation_space.shape) == 3:
        env = VecPyTorchFrameStack(env, 4, device)
    elif 'obs_stack' in config['env']:
        env = VecPyTorchFrameStack(env, config['env']['obs_stack'], device)

    return env


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address=3, max_step_per_episode=4500):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()
        self.max_step_per_episode = max_step_per_episode
        self.steps = 0

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.steps += 1
        self.visited_rooms.add(self.get_current_room())
        if self.max_step_per_episode < self.steps:
            done = True
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            done = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


"""
[num_process,84,84,3]
"""


class VecFrameStack(VecEnvWrapper):
    def __init__(self, vec_env, num_stack):
        self.venv = vec_env
        self.num_stack = num_stack
        self.raw_obs_shape = self.vec_env.observation.shape
        print(self.raw_obs_shape)

        wrapped_observation = vec_env.observation_space  # wrapped ob space
        self.shape_dim0 = wrapped_observation.shape[0]

        low = np.repeat(wrapped_observation.low, self.num_stack, axis=0)
        high = np.repeat(wrapped_observation.high, self.num_stack, axis=0)

        self.stacked_obs = np.zeros((vec_env.num_envs,) + low.shape)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=vec_env.observation_space.dtype)
        VecEnvWrapper.__init__(self, vec_env, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        #
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = np.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, vec_env, num_stack, device=None):
        self.venv = vec_env
        self.num_stack = num_stack
        self.is_stacked = True

        wrapped_observation = vec_env.observation_space  # wrapped ob space
        self.shape_dim0 = wrapped_observation.shape[0]

        low = np.repeat(wrapped_observation.low, self.num_stack, axis=0)
        high = np.repeat(wrapped_observation.high, self.num_stack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((vec_env.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=vec_env.observation_space.dtype)
        VecEnvWrapper.__init__(self, vec_env, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        #
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
