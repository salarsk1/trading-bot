from collections import namedtuple
from itertools import product
import itertools
import numpy as np
import pickle
from _utils import *

__all__ = ["Box", "Env"]

class Box(object):
    def __init__(self, low, high, shape):
        self.low   = low
        self.high  = high
        self.shape = shape

class Tuple(object):
    def __init__(self, spaces):
        self.space = []
        while spaces:
            self.space.append(spaces.pop(0))

class Env(object):

    def __init__(self, stocks, window, delta=10_000):
        
        self.window = window
        self.stock_env = StockPrep(stocks, window)
        self.initial_cash = 1_000_000
        self.state = np.zeros(self.stock_env.n_stocks + 1)
        self.share_of_each_stock = np.zeros(self.stock_env.n_stocks)
        self.delta = delta
        self.map_dict = {0 : (-1, -1, -1),
                         1 : (-1, -1, 0),
                         2 : (-1, -1, 1),
                         3 : (-1, 0, -1),
                         4 : (-1, 0, 0),
                         5 : (-1, 0, 1),
                         6 : (-1, 1, -1),
                         7 : (-1, 1, 0),
                         8 : (-1, 1, 1),
                         9 : (0, -1, -1),
                         10: (0, -1, 0),
                         11: (0, -1, 1),
                         12: (0, 0, -1),
                         13: (0, 0, 0),
                         14: (0, 0, 1),
                         15: (0, 1, -1),
                         16: (0, 1, 0),
                         17: (0, 1, 1),
                         18: (1, -1, -1),
                         19: (1, -1, 0),
                         20: (1, -1, 1),
                         21: (1, 0, -1),
                         22: (1, 0, 0),
                         23: (1, 0, 1),
                         24: (1, 1, -1),
                         25: (1, 1, 0),
                         26: (1, 1, 1)
                        }
        self.trans_percentage = 0.0025
        self.trans_amount     = 10_000
        self.portfolio = np.zeros(4)
        self.initial_portfolio = [0 for _ in range(4)]

    def reset(self, year):
        self.year = year
        for i in range(self.stock_env.n_stocks):
            self.share_of_each_stock[i] = np.floor(250_000 / self.stock_env.prepared_data[i][year-2015].iloc[self.window-1]['Close'])
            self.portfolio[i+1] = self.share_of_each_stock[i]  * self.stock_env.prepared_data[i][year-2015].iloc[self.window-1]['Close']
        self.portfolio[0] = self.initial_cash - np.sum(self.portfolio[1:])
        self.initial_portfolio = np.copy(self.portfolio)
        self.state = self.portfolio.reshape(1, -1) / np.sum(self.portfolio)
        return self.state

    def look_up_dict(self, action):
        return self.map_dict[action]

    def step(self, t, action):

        action = self.look_up_dict(action)
        current_portfolio_value = np.sum(self.portfolio)
        for i, a in enumerate(action):
            if a==-1:
                price = self.stock_env.prepared_data[i][self.year-2015].iloc[self.window+t]['Close']
                share = np.floor(self.trans_amount / price)
                if share > self.share_of_each_stock[i]:
                    pass
                    self.portfolio[0] += (1.0 - self.trans_percentage) * price * self.share_of_each_stock[i]
                    self.share_of_each_stock[i] = 0
                    self.portfolio[i+1] = price * self.share_of_each_stock[i]
                else:
                    self.share_of_each_stock[i] -= share
                    self.portfolio[i+1] = self.share_of_each_stock[i] * price
                    self.portfolio[0] += (1.0 - self.trans_percentage) * price * share

        for i, a in enumerate(action):
            if a==1:
                price = self.stock_env.prepared_data[i][self.year-2015].iloc[self.window+t]['Close']
                share = np.floor(self.trans_amount / price)
                if self.portfolio[0] < price * share:
                    pass
                    self.share_of_each_stock[i] += np.floor(self.portfolio[0] / price)
                    self.portfolio[i+1] = self.share_of_each_stock[i] * price
                    self.portfolio[0] -= (1.0 + self.trans_percentage) * price * np.floor(self.portfolio[0] / price)
                else:
                    self.share_of_each_stock[i] += share
                    self.portfolio[i+1] = self.share_of_each_stock[i] * price
                    self.portfolio[0] -= (1.0 + self.trans_percentage) * price * share
        for i in range(self.stock_env.n_stocks):
            self.portfolio[i+1] = self.share_of_each_stock[i] * \
                                self.stock_env.prepared_data[i][self.year-2015].iloc[self.window+t]['Close']

        self.reward = (np.sum(self.portfolio) - current_portfolio_value) / current_portfolio_value
        self.state = self.portfolio.reshape(1, -1) / np.sum(self.portfolio)
        return self.state, self.reward

if __name__ == "__main__":

    env = Env(['SPY', 'IWD', 'IWC'], 20)
    env.reset(2011)
    for i in range(3):
        env.step(30, 3)
    print(np.sum(env.portfolio))

    

    

