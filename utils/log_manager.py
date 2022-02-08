from collections import defaultdict

import numpy as np

from utils.enums import RewardKey


class LogManager:
    def __init__(self, config):
        self._main_loss_log = defaultdict(list)
        self._sub_loss_log = defaultdict(list)
        self._reward_log = defaultdict(list)
        self._log_interval = config['log_interval']

    def print_log(self, step, is_last=False):
        if is_last:
            self._print_reward_log(step)
            self._print_loss_log()
            return
        if self._check_print_log_time(step):
            self._print_reward_log(step)
            self._print_loss_log()

    def add_main_algo_loss(self, losses):
        for k, v in losses.items():
            self._main_loss_log[k].append(v)

    def add_sub_algo_loss(self, name, loss):
        self._sub_loss_log[name].append(loss)

    def add_reward(self, rewards):
        for k, v in rewards.items():
            self._reward_log[k].append(v)

    def _print_reward_log(self, step):
        print(f"{step} step elapsed\n")
        for k, v in self._reward_log.items():
            if k == RewardKey.EXTRINSIC:
                print(f"\t\t{RewardKey(k).value} avg score : {np.mean(self._reward_log[RewardKey(k)]):.1f}\n"
                      f"\t\t\tmin / max score : {np.min(self._reward_log[RewardKey(k)]):.1f} / {np.max(self._reward_log[RewardKey(k)]):.1f}\n")
            else:
                print(f"\t\t{RewardKey(k).value} avg score : {np.mean(self._reward_log[RewardKey(k)]):.3f}\n"
                      f"\t\t\tmin / max score : {np.min(self._reward_log[RewardKey(k)]):.3f} / {np.max(self._reward_log[RewardKey(k)]):.3f}\n")
        self._reward_log.clear()

    def _print_loss_log(self):
        loss_names = list(self._main_loss_log.keys())
        losses = list(self._main_loss_log.values())
        loss_msg = "\t"
        for i in range(len(loss_names)):
            loss_msg += f'{loss_names[i]} : {np.mean(losses[i]):.5f} \t'

        print(loss_msg)
        self._main_loss_log.clear()

        loss_names = list(self._sub_loss_log.keys())
        losses = list(self._sub_loss_log.values())
        loss_msg = "\t"
        for i in range(len(loss_names)):
            loss_msg += f'{loss_names[i]} : {np.mean(losses[i]):.5f} \t'

        print(loss_msg, "\n")
        self._sub_loss_log.clear()

    def _check_print_log_time(self, step):
        if step % self._log_interval == 0 and step > 0:
            return True
        else:
            return False
