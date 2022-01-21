from enum import Enum


class MemoryKey(Enum):
    STATE = "state"
    NEXT_STATE = "next_state"
    ACTION = "action"
    LOG_PROB = "log_prob"
    MASK = "mask"
    TRAJECTORY = "trajectory"


class RewardKey(Enum):
    EXTRINSIC = 'extrinsic'
    GAIL = 'gail'
    RND = 'rnd'

