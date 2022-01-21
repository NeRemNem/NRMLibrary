import gzip
import os
import pickle

import gym
from gym.utils import play
from utils.util import FrameStack

env = gym.make('MountainCar-v0')
stack = 32
env = FrameStack(env, stack)
"""
states = [s0, s1, s2, s3 ...]
actions = [a0, a1, a2, a3 ...]

"""


class RecordStudio:
    def __init__(self, env, path, key=None, count=3):
        self.states = []
        self.actions = []
        self._is_record = False
        self.count = count
        self.env = env
        self.key = key
        self._path = path

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        self.states.append(obs_t.copy())
        self.actions.append(action)
        if done:
            self.count += 1
        if self.count > 2 and done:
            if not self._is_record:
                self._is_record = True
                d = {'state': self.states, 'action': self.actions}
                with gzip.open(self._path, "wb") as f:
                    pickle.dump(d, f)
                    print("done!")

    def record(self, fps=140, zoom=2, callback=None):
        if callback is None:
            callback = self.callback
        if self.key is None:
            self.key = env.get_keys_to_action()

        while not self._is_record:
            play.play(self.env, True, fps=fps, zoom=zoom, callback=callback, keys_to_action=self.key)
        exit()


path = f"{os.pardir}/Demos/mountain_car_32.pickle"
k = {(ord('a'),): 0, (ord('d'),): 2, (): 1}
studio = RecordStudio(env, path, k)
studio.record()
