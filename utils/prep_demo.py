import gzip
import os
import pickle

import numpy as np

num_stack = 64
OBS = 0
ACT = 1
"""
demo = 
    [Episode 1, Episode 2, Episode 3]
    Episode = 
        [ [obs, action],[obs, action], ... ]
            2      1        

prep_episode =
    sa = cat (state, action) 
    [ sa1, sa2, sa3 ... ]
            
all_episode =
    [Prep Episode 1, Prep Episode 2, Prep Episode 3]
    
"""
with gzip.open(f"{os.getcwd()}/Demos/mountain_car_32.pickle", "r") as f:
    demo = pickle.load(f)
    all_episode = []
    prep_episode = []
    for episode in demo:
        prep_episode = []
        for t in episode:
            action = np.array([t[ACT]])
            state = t[OBS]
            sa = np.concatenate((state,action),axis=0)
            prep_episode.append(sa)
        print('prep_episode', len(prep_episode))
        all_episode.append(prep_episode)
    with gzip.open(f"{os.getcwd()}/Demos/mountain_car_trajectory_32_prep.pickle", "wb") as ff:
        pickle.dump(all_episode,ff)
        print('prep Done')

