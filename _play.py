import os
import numpy as np
import torch
import collections
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from rl_portfolio import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'

lstm_hidden = 10
time_window = 10
# agent = A2CAgent(time_window, [1024, 1024, 1024, 512], 27, learning_rate=1.0e-5)

agent = A2CAgent(time_window, [1000, 500], 27,
            learning_rate = 0.001, gamma=0.95)

# agent = DQNAgent(time_window, lstm_hidden * 3 + 4, lstm_hidden, [100, 100], 27,
#             learning_rate = 0.0001, gamma=0.95, epsilon_decay=0.99, 
#             min_eps = 0.05, target_update_freq = 5, max_memory_size = 50_000)

env   = Env(['MSFT', 'JPM', 'WMT'], time_window)

n_episodes = 2000

if __name__ == "__main__":

    # run1(env, n_episodes, agent, batch_size=64)

    run2(env, n_episodes, agent, batch_size = 64, 
        window_size = 200, write_output_every = 5000)

