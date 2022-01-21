import enum
import warnings
from abc import *
from typing import List

import numpy as np
import torch.nn as nn

from utils.enums import RewardKey

EPSILON = 1e-7


class NetKey(enum.Enum):
    BASE = 'base'
    ACTOR = 'actor'
    CRITIC = 'critic'

    ACTOR_BASE = 'actor_base'
    CRITIC_BASE = 'critic_base'

    ACTOR_OUT = 'actor_out'
    CRITIC_OUT = 'critic_out'

    TARGET = 'target'
    PREDICT = 'predict'


# region init helpers
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, feature):
        return feature.view(feature.size(0), -1)


# endregion

class Net(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def get_layer(self, net_key: NetKey):
        pass


# TODO: Make super class for (Policy, Policy2)
class Policy(Net):
    def __init__(self, obs_space, action_space, num_hidden_unit=256, critics=None):
        super(Policy, self).__init__()

        if critics is None:
            critics = [RewardKey.EXTRINSIC]
        elif RewardKey.EXTRINSIC not in critics:
            critics.append(RewardKey.EXTRINSIC)

        simple_actor_critic = SimpleActorCritic(obs_space, action_space, num_hidden_unit, critics)
        self._base = simple_actor_critic.base
        self._actor_out_layer = simple_actor_critic.actor_out_layer
        self._critic_out_layers = simple_actor_critic.critic_out_layers
        self._has_same_base = True

    @property
    def has_same_base(self):
        return self._has_same_base

    def get_layer(self, net_key: NetKey):
        if net_key == NetKey.ACTOR_BASE or net_key == NetKey.CRITIC_BASE or net_key == NetKey.BASE:
            return self._base
        elif net_key == NetKey.ACTOR_OUT:
            return self._actor_out_layer
        elif net_key == NetKey.CRITIC_OUT:
            return self._critic_out_layers
        else:
            raise NotImplementedError


class Policy2(Net):
    def __init__(self, obs_space, action_space, num_hidden_unit=256, critics=None):
        super(Policy2, self).__init__()

        if critics is None:
            critics = [RewardKey.EXTRINSIC.value]
        elif RewardKey.EXTRINSIC not in critics:
            critics.append(RewardKey.EXTRINSIC.value)

        actor = Actor(obs_space, action_space, num_hidden_unit)
        critic = Critic(obs_space, num_hidden_unit, critics)

        self._actor_base = actor.base
        self._critic_base = critic.base
        self._actor_out_layer = actor.out_layer
        self._critic_out_layers = critic.out_layers

        self._has_same_base = False

    def get_layer(self, net_key: NetKey):
        if net_key == NetKey.ACTOR_BASE:
            return self._actor_base
        elif net_key == NetKey.CRITIC_BASE:
            return self._critic_base
        elif net_key == NetKey.ACTOR_OUT:
            return self._actor_out_layer
        elif net_key == NetKey.CRITIC_OUT:
            return self._critic_out_layers
        else:
            raise NotImplementedError

    @property
    def has_same_base(self):
        return self._has_same_base


class Discriminator(Net, metaclass=ABCMeta):
    def __init__(self, input_dim, num_hidden_unit=256):
        super().__init__()
        input_dim = input_dim
        self._base = nn.Sequential(
            Base(input_dim, num_hidden_unit),
            nn.Linear(num_hidden_unit, 1)
        )

    def get_layer(self, net_key: NetKey):
        if net_key != NetKey.BASE:
            warnings.warn(f"{self.__class__.__name__} has only base layer")
        return self._base


class RNDNet(Net):
    def __init__(self, obs_space, num_hidden_unit=256):
        super(RNDNet, self).__init__()
        self._target_net = TargetNet(obs_space, num_hidden_unit)
        self._predict_net = PredictNet(obs_space, num_hidden_unit)

    def get_layer(self, net_key: NetKey):
        if net_key == NetKey.TARGET:
            return self._target_net
        elif net_key == NetKey.PREDICT:
            return self._predict_net
        else:
            raise NotImplementedError


class TargetNet(nn.Module):
    def __init__(self, obs_space, num_hidden_unit=512):
        super(TargetNet, self).__init__()
        if len(obs_space.shape) == 1:
            base_ = Base(obs_space.shape[0], num_hidden_unit)
        elif len(obs_space.shape) == 3:
            base_ = CNNBase(1, num_hidden_unit)
        else:
            raise NotImplementedError
        self.base = nn.Sequential(
            base_
        )
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        return self.base(inputs)


class PredictNet(nn.Module):
    def __init__(self, obs_space, num_hidden_unit=512):
        super(PredictNet, self).__init__()
        if len(obs_space.shape) == 1:
            base_ = Base(obs_space.shape[0], num_hidden_unit)
        elif len(obs_space.shape) == 3:
            base_ = CNNBase(1, num_hidden_unit)
        else:
            raise NotImplementedError
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.base = nn.Sequential(
            base_
            , nn.ReLU()
            , init_(nn.Linear(num_hidden_unit, num_hidden_unit))
            , nn.ReLU()
            , init_(nn.Linear(num_hidden_unit, num_hidden_unit))
        )

    def forward(self, inputs):
        return self.base(inputs)


class SimpleActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, num_hidden_unit, critics: List[str] = None):
        super(SimpleActorCritic, self).__init__()

        if critics is None:
            critics = [RewardKey.EXTRINSIC.value]
        elif 'extrinsic' not in critics:
            critics.append(RewardKey.EXTRINSIC.value)

        if len(obs_space.shape) == 1:
            self.base = Base(obs_space.shape[0], num_hidden_unit)
        elif len(obs_space.shape) == 3:
            self.base = CNNBase(obs_space.shape[0], num_hidden_unit)
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.actor_out_layer = init_(nn.Linear(num_hidden_unit, act_space.n))
        self.critic_out_layers = nn.ModuleDict()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        for reward_key in critics:
            self.critic_out_layers[RewardKey(reward_key).value] = init_(nn.Linear(num_hidden_unit, 1))


class Actor(nn.Module):
    def __init__(self, obs_space, act_space, num_hidden_unit):
        super(Actor, self).__init__()
        if len(obs_space.shape) == 1:
            self.base = Base(obs_space.shape[0], num_hidden_unit)
        elif len(obs_space.shape) == 3:
            self.base = CNNBase(obs_space.shape[0], num_hidden_unit)
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        self.out_layer = init_(nn.Linear(num_hidden_unit, act_space.n))


class Critic(nn.Module):
    def __init__(self, obs_space, num_hidden_unit, critics: List[str] = None):
        super(Critic, self).__init__()

        if critics is None:
            critics = [RewardKey.EXTRINSIC.value]
        elif 'extrinsic' not in critics:
            critics.append(RewardKey.EXTRINSIC.value)

        self.out_layers = nn.ModuleDict()
        if len(obs_space.shape) == 1:
            self.base = Base(obs_space.shape[0], num_hidden_unit)
        elif len(obs_space.shape) == 3:
            self.base = CNNBase(obs_space.shape[0], num_hidden_unit)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        for reward_key in critics:
            self.out_layers[RewardKey(reward_key).value] = init_(nn.Linear(num_hidden_unit, 1))


class Base(nn.Module):
    def __init__(self, num_input, num_hidden_unit, do_init=True):
        super(Base, self).__init__()
        self.input_layer = nn.Linear(num_input, num_hidden_unit)
        if do_init:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), np.sqrt(2))

            self.__layer = nn.Sequential(
                init_(nn.Linear(num_input, num_hidden_unit))
                , nn.Tanh()
                , init_(nn.Linear(num_hidden_unit, num_hidden_unit))
                , nn.Tanh()
            )
        else:
            self.__layer = nn.Sequential(
                nn.Linear(num_input, num_hidden_unit)
                , nn.Tanh()
                , nn.Linear(num_hidden_unit, num_hidden_unit)
                , nn.Tanh()

            )

    def forward(self, obs):
        return self.__layer(obs)


class CNNBase(nn.Module):
    def __init__(self, num_input, num_hidden_unit):
        super(CNNBase, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.__layer = nn.Sequential(
            init_(nn.Conv2d(num_input, 32, 8, stride=4))
            , nn.ReLU()
            , init_(nn.Conv2d(32, 64, 4, stride=2))
            , nn.ReLU()
            , init_(nn.Conv2d(64, 32, 3, stride=1))
            , nn.ReLU()
            , Flatten()
            , init_(nn.Linear(32 * 7 * 7, num_hidden_unit))
            , nn.ReLU()
        )

    def forward(self, obs):
        return self.__layer(obs / 255)
