import gzip
import pickle
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from utils.enums import MemoryKey, RewardKey


class Memory:
    def __init__(self, nrm):
        if 'gail' in self._config['reward']:
            self._demo_controller = DemoController(self._config)
        self._config = nrm.config
        self._device = self._config['device']
        self._memory: Dict[MemoryKey, torch.Tensor] = self.__init_memory(nrm.env.observation_space.shape, self._config)
        self._batch_size = self._config['batch_size']
        self._buffer_size = self._config['buffer_size']
        self._num_process = self._config['env']['num_process']
        self.base_shape = (self._buffer_size, self._num_process)
        self._obs_filter = nrm.obs_filter
        self._made_return = False
        self._step = 0

    def __init_memory(self, obs_shape, config):
        memory = defaultdict(torch.Tensor)
        buffer_size = config['buffer_size']
        num_process = config['env']['num_process']

        memory[MemoryKey.STATE] = torch.zeros((buffer_size + 1, num_process, *obs_shape)).to(self._device)
        memory[MemoryKey.ACTION] = torch.zeros((buffer_size, num_process, 1)).long().to(self._device)
        memory[MemoryKey.LOG_PROB] = torch.zeros((buffer_size, num_process, 1)).to(self._device)
        memory[MemoryKey.MASK] = torch.ones((buffer_size + 1, num_process, 1)).to(self._device)

        return memory

    def to(self, device):
        for k, v in self._memory.items():
            self._memory[k] = v.to(device)

    def push(self, **data):
        for key, value in data.items():
            if not isinstance(key, MemoryKey):
                key = MemoryKey(key)

            if key == MemoryKey.STATE or key == MemoryKey.MASK:
                self._memory[key][self._step + 1].copy_(value)
            else:
                self._memory[key][self._step].copy_(value)
        self._step += 1

    def push_hard(self, key, tensors):
        self._memory[key] = tensors

    def get(self, key: MemoryKey, flatten=False):
        if flatten:
            return self.__flatten(key)
        else:
            return self._memory[key]

    def make_trajectory(self):
        trajectory = torch.cat([self._memory[MemoryKey.STATE][:-1], self._memory[MemoryKey.ACTION]], dim=-1)
        self._memory[MemoryKey.TRAJECTORY] = trajectory

    def prep_demo_observation(self):
        self._demo_controller.prep_observation(self.obs_filter)

    def get_batch(self, *args, **kwargs):
        total_num = self._buffer_size * self._num_process
        sampler = BatchSampler(SubsetRandomSampler(range(total_num)), self._batch_size, True)

        for indices in sampler:
            batch_dic = {'indices': indices}
            for key in args:
                batch_dic[key] = self.__flatten(key)[indices]
            for key, tensor in kwargs.items():
                batch_dic[key] = tensor[indices]
            yield batch_dic

    def get_input(self, reward_key: RewardKey):
        if reward_key == RewardKey.GAIL:
            return self._memory[MemoryKey.TRAJECTORY]
        elif reward_key == RewardKey.RND:
            # no stacked obs
            return self._memory[MemoryKey.STATE][1:, :, -1:, ]

    def ready(self, obs):
        self._memory[MemoryKey.STATE][0].copy_(obs)

    def after_update(self):
        self._made_return = False
        self._memory[MemoryKey.STATE][0].copy_(self._memory[MemoryKey.STATE][-1])
        self._memory[MemoryKey.MASK][0].copy_(self._memory[MemoryKey.MASK][-1])
        self._step = 0

    # return agent's states & actions
    def get_demo_trajectory_batches(self):
        return self._demo_controller.get_batch()

    def __flatten(self, key: MemoryKey):
        if key == MemoryKey.STATE:
            return self._memory[key][:-1].view(-1, *self._memory[key].size()[2:])
        elif key == MemoryKey.MASK:
            return self._memory[key][:-1].view(-1, 1)
        elif key == MemoryKey.NEXT_STATE:
            return self._memory[MemoryKey.STATE][1:, :, -1:, ].view(-1, 1, *self._memory[MemoryKey.STATE].size()[3:])
        else:
            return self._memory[key].view(-1, 1)

    @property
    def obs_filter(self):
        return self._obs_filter

    @obs_filter.setter
    def obs_filter(self, obs_filt):
        self._obs_filter = obs_filt


class DemoController:
    def __init__(self, config):
        self.pointer = 0
        with gzip.open(config['reward']['gail']['demo_path'], 'r') as f:
            raw_demo_data = pickle.load(f)
        self._demo_state = torch.tensor(raw_demo_data['state']).float()
        self._demo_action = torch.tensor(raw_demo_data['action']).unsqueeze(-1).float()

        self.num_stack = config['env']['obs_stack']
        self.batch_size = config['batch_size']
        self.indices = np.arange(len(self._demo_action))
        self._device = config['device']
        np.random.shuffle(self.indices)
        self.num_batches = len(self._demo_action) // self.batch_size
        self._demo_trajectory = torch.zeros(())

    def prep_observation(self, obs_filter):
        prep_state = obs_filter(self._demo_state)
        self._demo_trajectory = torch.cat([prep_state, self._demo_action], dim=-1).to(self._device)

    def get_batch(self):
        if self.pointer + self.batch_size > len(self.indices):
            self.pointer = 0
            np.random.shuffle(self.indices)

        trajectory_batch = self._demo_trajectory[self.pointer:self.pointer + self.batch_size]
        self.pointer += self.batch_size

        return trajectory_batch

    @property
    def demo_trajectory(self):
        return self._demo_trajectory
