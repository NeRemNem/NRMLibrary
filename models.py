from abc import *

import torch
import torch.optim as optim
from torch.distributions import Categorical

import nets
from nets import NetKey
from utils.enums import RewardKey

EPSILON = 1e-7


def create_agent(env, config):
    model_params = config['model']['params']

    on_rewards = list(config['reward'].keys())
    if len(on_rewards) < 1:
        raise

    def create_t1_agent():
        net = nets.Policy(env.observation_space, env.action_space, model_params['hidden_unit'], on_rewards)
        net.to(config['device'])
        optimizer = optim.Adam(net.parameters(), lr=model_params['lr'])
        return net, optimizer

    def create_t2_agent():
        net = nets.Policy2(env.observation_space, env.action_space, model_params['hidden_unit'], on_rewards)
        net.to(config['device'])
        optimizer = optim.Adam(net.parameters(), lr=model_params['lr'])
        return net, optimizer

    if len(on_rewards) > 1:
        net, optimizer = create_t2_agent()
    else:
        net, optimizer = create_t1_agent()
    agent = ActorCriticAgent(net, optimizer, config['device'])

    return agent


def create_intrinsic_model(env, config):
    reward_config = config['reward']

    def create_gail_model():
        net = nets.Discriminator(env.observation_space.shape[0] + 1, reward_config['gail']['hidden_unit'])
        net.to(config['device'])
        optimizer = optim.Adam(net.parameters(), lr=reward_config['gail']['lr'])
        return GAILModel(net, optimizer, config['device'])

    def create_rnd_model():
        net = nets.RNDNet(env.observation_space, reward_config['rnd']['hidden_unit'])
        net.to(config['device'])
        optimizer = optim.Adam(net.get_layer(NetKey.PREDICT).parameters(), lr=reward_config['rnd']['lr'])
        return RNDModel(net, optimizer, config['device'])

    intrinsic_models = {}
    if 'gail' in reward_config:
        intrinsic_models[RewardKey.GAIL] = create_gail_model()

    if 'rnd' in reward_config:
        intrinsic_models[RewardKey.RND] = create_rnd_model()

    return intrinsic_models


class Model:
    def __init__(self, net: nets.Net, optimizer, device):
        self._net = net
        self._optimizer = optimizer
        self._device = device

    @property
    def net(self):
        return self._net

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optim):
        self._optimizer = optim

    @property
    def device(self):
        return self._device

    def to(self, device_name):
        self._net.to(device_name)
        self._optimizer.to(device_name)
        self._device = device_name


class ActorCriticAgent(Model):
    def __init__(self, net, optimizer, device):
        super().__init__(net, optimizer, device)

    @torch.no_grad()
    def act(self, inputs):
        actor_out_layer = self._net.get_layer(NetKey.ACTOR_OUT)
        actor_base = self._net.get_layer(NetKey.ACTOR_BASE)

        feature = actor_base(inputs)
        logits = actor_out_layer(feature)
        dist = Categorical(logits=logits)
        sample = dist.sample()
        action = sample.unsqueeze(-1)
        log_prob = dist.log_prob(sample).unsqueeze(-1)

        return action, self.get_value(inputs), log_prob

    def get_value(self, inputs):
        values = {}
        critic_out_layers = self._net.get_layer(NetKey.CRITIC_OUT)

        base = self._net.get_layer(NetKey.CRITIC_BASE)

        feature = base(inputs)
        for key, layer in critic_out_layers.items():
            values[RewardKey(key)] = layer(feature)

        return values

    def evaluate(self, inputs, action):
        actor_base = self._net.get_layer(NetKey.ACTOR_BASE)
        actor_out_layer = self._net.get_layer(NetKey.ACTOR_OUT)

        critic_base = self._net.get_layer(NetKey.CRITIC_BASE)
        critic_out_layers = self._net.get_layer(NetKey.CRITIC_OUT)

        feature = actor_base(inputs)
        logits = actor_out_layer(feature)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action.squeeze(1))

        feature = critic_base(inputs)
        values = []
        for k, layer in critic_out_layers.items():
            values.append(layer(feature))
        values = torch.stack(values).squeeze(0)
        dist_entropy = dist.entropy()
        return log_prob.unsqueeze(1), values, dist_entropy.mean()


class IntrinsicModel(Model, metaclass=ABCMeta):
    def __init__(self, net, optimizer, device):
        super(IntrinsicModel, self).__init__(net, optimizer, device)

    @abstractmethod
    def make_intrinsic_reward(self, inputs):
        pass

    @abstractmethod
    def make_intrinsic_logit(self, inputs):
        pass


class GAILModel(IntrinsicModel):
    def __init__(self, net, optimizer, device):
        super(GAILModel, self).__init__(net, optimizer, device)

    def make_intrinsic_logit(self, inputs):
        base = self.net.get_layer(NetKey.BASE)
        return base(inputs)

    @torch.no_grad()
    def make_intrinsic_reward(self, inputs):
        with torch.no_grad():
            base = self.net.get_layer(NetKey.BASE)
            logit = base(inputs)
            prob = torch.sigmoid(logit)
            # reward = -torch.log(1 - prob * (1.0 - EPSILON))
            reward = torch.log(prob) - torch.log(1 - prob)
            return reward


class RNDModel(IntrinsicModel):
    def __init__(self, net, optimizer, device):
        super(RNDModel, self).__init__(net, optimizer, device)

    def make_intrinsic_logit(self, inputs):
        pass

    @torch.no_grad()
    def make_intrinsic_reward(self, inputs):
        reward_shape = (*inputs.size()[:2], 1)
        prep_inputs = inputs.view(-1, *inputs.size()[2:])

        target = self.net.get_layer(NetKey.TARGET)
        predict = self.net.get_layer(NetKey.PREDICT)

        rewards =(predict(prep_inputs)- target(prep_inputs)).pow(2).mean(-1).unsqueeze(-1)
        rewards = rewards.view(reward_shape)
        # print(rewards.mean())
        return rewards
