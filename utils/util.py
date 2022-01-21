from gym import Wrapper
from gym.spaces import Box
import numpy as np

from collections.abc import Sequence


def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst),)

    # recurse
    shape = get_shape(lst[0], shape)

    return shape


class FrameStack(Wrapper):
    def __init__(self, env, num_stack):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack

        self.origin_shape = env.observation_space.shape[0]
        self.stack = np.zeros(env.observation_space.shape[0] * num_stack)

        low = np.repeat(env.observation_space.low, num_stack, axis=0)
        high = np.repeat(env.observation_space.high, num_stack, axis=0)

        self.observation_space = Box(low=low
                                     , high=high
                                     , shape=(env.observation_space.shape[0] * num_stack,)
                                     , dtype=self.observation_space.dtype
                                     )

    def _get_observation(self):
        return self.stack.copy()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.stack[:-self.origin_shape] = self.stack[self.origin_shape:]
        self.stack[-self.origin_shape:] = observation
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.stack = np.zeros(self.env.observation_space.shape[0] * self.num_stack)
        self.stack[-self.origin_shape:] = observation
        return self._get_observation()


