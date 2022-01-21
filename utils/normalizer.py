import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd


class ObsFilter:
    def __init__(self):
        self._obs_rms = RunningMeanStd()
        self._training = True

    def __call__(self, inputs, update=True):
        _is_gpu = False
        if inputs.is_cuda:
            _is_gpu = True
            inputs = inputs.cpu()
        if update and self._training:
            self._update(inputs)

        if _is_gpu:
            return torch.clip(((inputs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var[0] + 1e-11)), -5, 5).cuda()
        else:
            return torch.clip(((inputs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var[0] + 1e-11)), -5, 5)

    def _update(self, inputs):
        if not isinstance(inputs, np.ndarray):
            inputs = inputs.numpy()
        self._obs_rms.update(inputs)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False
