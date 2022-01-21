from typing import Dict

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd

from models import IntrinsicModel
from storage import Memory
from utils.enums import RewardKey

early = [RewardKey.RND]
lazy = [RewardKey.EXTRINSIC, RewardKey.GAIL]


class RewardValueFactory:
    def __init__(self, intrinsic_models: Dict[RewardKey, IntrinsicModel], memory: Memory, config):

        self._memory = memory
        self._on_rewards = []
        self._strengths = {}
        self._reward_config = config['reward']
        self._r_step = 0
        self._v_step = 0
        self._buffer_size = config['buffer_size']
        self._intrinsic_models = intrinsic_models
        self._device = config['device']

        self._reward_normalizers: Dict[RewardKey, RewardNormalizer] = {}
        self._rewards: Dict[RewardKey, torch.Tensor] = {}
        self._values: Dict[RewardKey, torch.Tensor] = {}
        for key in self._reward_config.keys():
            reward_key = RewardKey(key)
            self._on_rewards.append(reward_key)

            self._rewards[reward_key] = torch.zeros((config['buffer_size'], config['env']['num_process'], 1)).to(
                self._device)
            self._values[reward_key] = torch.zeros((config['buffer_size'] + 1, config['env']['num_process'], 1)).to(
                self._device)

            self._strengths[reward_key] = self._reward_config[key]['strength']
            self._reward_normalizers[reward_key] = RewardNormalizer(self._reward_config[key]['gamma'])

    def push_extrinsic_reward(self, reward: torch.Tensor):
        if RewardKey.EXTRINSIC in self._rewards.keys():
            self._rewards[RewardKey.EXTRINSIC][self._r_step % self._buffer_size].copy_(reward)
            self._r_step += 1

    def push_values(self, values):
        for k, v in values.items():
            if k in self._values.items():
                self._values[k][self._v_step % self._buffer_size].copy_(v.detach())

        self._v_step += 1

    def _prep_reward(self, key: RewardKey):
        if key not in self._rewards.keys():
            return

        if key == RewardKey.EXTRINSIC:
            self._r_step = 0
            self._v_step = 0
            rewards = self._rewards[key]
        else:
            model = self._intrinsic_models[key]
            inputs = self._memory.get_input(key)
            rewards = model.make_intrinsic_reward(inputs)
            rewards.to(self.device)
        prep_rewards = self._reward_normalizers[key](rewards)
        self._rewards[key] = prep_rewards.to(self.device) * self._strengths[key]

    @property
    def reward_config(self):
        return self._reward_config

    @property
    def device(self):
        return self._device

    @property
    def rewards(self):
        return self._rewards

    @property
    def values(self):
        return self._values

    def get_rewards_mean(self):
        rewards = {}
        for k, v in self._rewards.items():
            rewards[k] = v.view(-1, 1).mean().cpu().item()
        return rewards

    @property
    def on_rewards(self):
        return self._on_rewards

    def make_early_reward(self):
        for name in early:
            self._prep_reward(name)

    def make_lazy_reward(self):
        for name in lazy:
            self._prep_reward(name)


class RewardNormalizer:
    def __init__(self, gamma):
        self._rms = RunningMeanStd()
        self._returns = None
        self._gamma = gamma

    def __call__(self, rewards):
        return self._normalize_reward(rewards)

    def _normalize_reward(self, rewards):
        if rewards.is_cuda:
            rewards = rewards.cpu()
        for i in range(rewards.size(0)):
            if self._returns is not None:
                self._returns = self._returns * self._gamma + rewards[i]
                self._rms.update(self._returns.numpy())
            else:
                self._returns = rewards[i].clone()
        return rewards / np.sqrt(self._rms.var + 1e-11)
