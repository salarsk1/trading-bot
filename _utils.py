import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as data_reader

__all__ = ["ReplayBuffer", "StockPrep"]

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class StockPrep(object):

    def __init__(self, stocks, window):

        self.n_stocks = len(stocks)
        self.stocks = stocks
        self.window = window
        self.prepared_data =[] 
        self.prepared_data = self.prepare_data()

    def prepare_data(self):

        for s in range(self.n_stocks):
            temp = []
            datas = data_reader.DataReader(self.stocks[s], data_source="yahoo")
            for year in range(2015, 2019):
                past = datas.loc[str(year-1)+'-01-01':str(year-1)+'-12-31']
                past = past.ix[-self.window:]
                new  = datas.loc[str(year)+'-01-01':str(year)+'-12-31']
                temp.append(pd.concat([past, new]))
            self.prepared_data.append(temp)
        return self.prepared_data

    def extract_data(self, year, t):

        k_c = np.zeros((self.n_stocks, self.window))
        k_o = np.zeros((self.n_stocks, self.window))
        k_h = np.zeros((self.n_stocks, self.window))
        k_l = np.zeros((self.n_stocks, self.window))
        k_v = np.zeros((self.n_stocks, self.window))
        for n in range(self.n_stocks):
            high_p   = np.array(self.prepared_data[n][year-2015]['High'])
            low_p    = np.array(self.prepared_data[n][year-2015]['Low'])
            open_p   = np.array(self.prepared_data[n][year-2015]['Open'])
            close_p  = np.array(self.prepared_data[n][year-2015]['Close'])
            volume   = np.array(self.prepared_data[n][year-2015]['Volume'])
            for i in range(t, self.window+t):
                k_c[n, i-t] = close_p[i]#(close_p[i] - close_p[i-1]) / close_p[i-1]
                k_o[n, i-t] = open_p[i]#(open_p[i]  - close_p[i-1]) / close_p[i-1]
                k_h[n, i-t] = high_p[i]#(close_p[i] - high_p[i])    / high_p[i]
                k_l[n, i-t] = low_p[i]#(close_p[i] - low_p[i])     / low_p[i]
                k_v[n, i-t] = volume[i]#(volume[i]  - volume[i-1])  / volume[i-1]

        return k_c, k_o, k_h, k_l, k_v

    def build_matrix(self, year, t):
        k_c, k_o, k_h, k_l, k_v = self.extract_data(year, t)
        return np.stack([k_c, k_o, k_h, k_l, k_v])

if __name__ == "__main__":
    stock = ['MSFT', 'JPM', 'WMT']
    prep  = StockPrep(stock, 4)
    data  = prep.prepare_data()
    print(prep.build_matrix(2017, 10))
    



