import os
import numpy as np
import torch
import collections
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict


agents = A2CAgent(5, 20, 128, [512, 512, 512], 27, learning_rate=1.0e-5)

env    = Env(['SPY', 'IWD', 'IWC'], 20)

n_episodes = 1000


if __name__ == "__main__":

    run(env, n_episodes, agent, batch_size = 64, 
        window_size = 200, write_output_every = 5000)

