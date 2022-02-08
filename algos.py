from typing import Dict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import *
from reward_value_factory import RewardValueFactory
from storage import Memory
from utils.enums import MemoryKey

ACTOR = 0
CRITIC = 1


def make_returns_advantages(rewards: Dict[RewardKey, torch.Tensor], values: Dict[RewardKey, torch.Tensor],
                            mask: torch.Tensor,
                            device,
                            properties, lambda_=0.95, use_gae=True):
    temp_advantage = []
    temp_returns = []
    for key in rewards.keys():
        reward = rewards[key]
        value = values[key]

        local_return = []
        local_advantage = []

        if key != RewardKey.EXTRINSIC:
            mask = torch.ones((reward.size(0) + 1, 1)).to(device)
        gae = 0
        gamma = properties[key]['gamma']
        for i in reversed(range(reward.size(0))):
            delta = reward[i] + gamma * value[i + 1] * mask[i + 1] - value[i]
            gae = delta + gamma * lambda_ * mask[i + 1] * gae
            local_return.insert(0, (gae + value[i]).tolist())
            local_advantage.insert(0, gae.tolist())
        temp_returns.append(local_return)
        temp_advantage.append(local_advantage)

    returns = torch.tensor(np.mean(np.array(temp_returns, dtype=np.float32), axis=0))
    advantages = torch.tensor(np.mean(np.array(temp_advantage, dtype=np.float32), axis=0))
    return returns, advantages


def create_algorithm(config):
    algos_fn = {
        'ppo': PPO
        , 'gail': GAIL
        , 'rnd': RND
    }
    main_algos = {}
    intrinsic_algos = {}
    algo_fn = algos_fn[config['algo']]

    main_algos[config['algo']] = algo_fn(config)

    if 'gail' in config['reward']:
        algo_fn = algos_fn['gail']
        intrinsic_algos[RewardKey.GAIL] = algo_fn(config)
    if 'rnd' in config['reward']:
        algo_fn = algos_fn['rnd']
        intrinsic_algos[RewardKey.RND] = algo_fn(config)

    return main_algos, intrinsic_algos


class Algo(metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self._device = config['device']
        self.batch_num = config['buffer_size'] // config['batch_size']

    @abstractmethod
    def update(self, memory: Memory, reward_factory: RewardValueFactory, model):
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device


class GAIL(Algo):
    def __init__(self, config):
        super(GAIL, self).__init__(config)
        self.epoch = config['reward']['gail']['epoch']
        self.agent_trajectory = torch.zeros((0))

    def update(self, memory: Memory, reward_factory: RewardValueFactory, model):
        discrim = model
        optimizer = model.optimizer
        memory.prep_demo_observation()
        memory.make_trajectory()
        count = 0
        loss_epoch = 0
        for _ in range(self.epoch):
            for batch in memory.get_batch(MemoryKey.STATE,MemoryKey.ACTION):
                demo_trajectory = memory.get_demo_trajectory_batches()
                agent_trajectory = torch.cat([batch[MemoryKey.STATE], batch[MemoryKey.ACTION]], dim=1)

                demo_logit = discrim.make_intrinsic_logit(demo_trajectory.to(self.device))
                agent_logit = discrim.make_intrinsic_logit(agent_trajectory.to(self.device))

                demo_loss = F.binary_cross_entropy_with_logits(demo_logit
                                                               , torch.ones(demo_trajectory.size(0))
                                                               .unsqueeze(1)
                                                               .to(self.device)
                                                               )
                agent_loss = F.binary_cross_entropy_with_logits(agent_logit,
                                                                torch.zeros(agent_trajectory.size(0))
                                                                .unsqueeze(1)
                                                                .to(self.device)
                                                                )
                loss = (demo_loss + agent_loss).item()
                loss_epoch += loss
                count += 1
                optimizer.zero_grad()
                (demo_loss + agent_loss).backward()
                optimizer.step()

        return "GAIL discrim loss", loss_epoch / (self.epoch * count)


class RND(Algo):
    def __init__(self, config):
        super(RND, self).__init__(config)
        self.epoch = 1

    def update(self, memory: Memory, reward_factory: RewardValueFactory, model):
        target = model.net.get_layer(NetKey.TARGET)
        predict = model.net.get_layer(NetKey.PREDICT)
        optimizer = model.optimizer
        loss_epoch = 0

        for _ in range(self.epoch):
            for batch in memory.get_batch(MemoryKey.NEXT_STATE):
                next_state = batch[MemoryKey.NEXT_STATE]
                predict_logit = predict(next_state)
                target_logit = target(next_state)

                loss = F.mse_loss(predict_logit, target_logit)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                loss_epoch += loss.item()

        return "RND predict loss", loss_epoch


class PPO(Algo):
    def __init__(self, config):
        super(PPO, self).__init__(config)
        params = config['model']['params']
        self.batch_num = config['buffer_size'] // config['batch_size']
        self.eps = params['eps']
        self.epoch = params['epoch']
        self.c_value = params['c_value']
        self.c_entropy = params['c_entropy']

    def update(self, memory: Memory, reward_factory: RewardValueFactory, model):
        optimizer = model.optimizer

        returns, advantages = make_returns_advantages(reward_factory.rewards, reward_factory.values,
                                                      memory.get(MemoryKey.MASK),
                                                      reward_factory.device
                                                      , reward_factory.properties)

        value_loss_epoch = 0
        policy_loss_epoch = 0
        dist_entropy_epoch = 0
        returns = returns.view(-1, 1).to(reward_factory.device)
        advantages = advantages.view(-1, 1).to(reward_factory.device)

        for i in range(self.epoch):
            for batch in memory.get_batch(MemoryKey.STATE, MemoryKey.ACTION, MemoryKey.LOG_PROB):
                indices = batch['indices']
                new_log_probs, new_values, entropy = model.evaluate(batch[MemoryKey.STATE], batch[MemoryKey.ACTION])
                ratio = torch.exp(new_log_probs - batch[MemoryKey.LOG_PROB])

                sur1 = ratio * advantages[indices]
                sur2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages[indices]

                p_loss = -torch.min(sur1, sur2).mean()
                v_loss = (returns[indices] - new_values).pow(2).mean()

                if type(optimizer) == tuple:
                    actor_optimizer = optimizer[ACTOR]
                    critic_optimizer = optimizer[CRITIC]

                    actor_optimizer.zero_grad()
                    p_loss.backward()
                    actor_optimizer.step()

                    critic_optimizer.zero_grad()
                    v_loss.backward()
                    critic_optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss = (self.c_value * v_loss + p_loss - self.c_entropy * entropy)
                    nn.utils.clip_grad_norm_(model.net.parameters(),
                                             0.5)
                    loss.backward()
                    optimizer.step()

                value_loss_epoch += v_loss.item()
                policy_loss_epoch += p_loss.item()
                dist_entropy_epoch += entropy.mean().item()
        return {"value loss": value_loss_epoch / (self.epoch * self.batch_num)
            , "policy loss": policy_loss_epoch / (self.epoch * self.batch_num)
            , "dist entropy ": dist_entropy_epoch / (self.epoch * self.batch_num)
                }
