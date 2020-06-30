import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from _utils import *

__all__ = ["run1", "run2"]



def run1(env, n_episodes, agent, batch_size = 64, 
        window_size = 200, write_output_every = 5000):
    rewards = []
    CR = []
    SR = []
    for episode in range(n_episodes):
        # year = np.random.choice([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018], 
        #                         p=[0.01, 0.02, 0.03, 0.06, 0.08, 0.2, 0.25, 0.35])

        period = int(np.random.choice(list(np.linspace(0, 7, 8)),
                                  p=[*[0.05]*2, *[0.1]*2, *[0.15]*2, *[0.2]*2]))
        if period < 2 and period >=0:
            year = 2015
        if period < 4 and period >=2:
            year = 2016
        if period < 6 and period >=4:
            year = 2017
        if period < 8 and period >=6:
            year = 2018
        print(year, period)
        env.reset(year)
        episode_length = env.stock_env.prepared_data[0][year-2015].shape[0]
        denom = (episode_length - env.window) // 2
        done = False
        state = env.state
        trajectory = []
        stocks_indicators = []
        for i in range(env.stock_env.n_stocks):
            stocks_indicators.append(env.stock_env.build_matrix(year, period%2*denom, i, state))


        state = [stocks_indicators, env.state]
        episode_reward = 0

        # denom = 30
        rho_t = []
        for t in range(period%2*denom+1, (period%2+1)*denom+1):
            action = agent.get_action(state, episode)
            new_state, reward = env.step(t-1, action)
            stocks_indicators = []
            for i in range(env.stock_env.n_stocks):
                stocks_indicators.append(env.stock_env.build_matrix(year, t, i, state))

            new_state = [stocks_indicators, env.state]
            # new_state = np.hstack([indicator_matrix, new_state])
            if t == (period%2+1)*denom:
                done = True
            agent.replay.push(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward
            if len(agent.replay) > batch_size:
                agent.update(batch_size, episode)
            rho_t.append(reward - 0.0001)

        rewards.append(episode_reward)

        print('episode {}, reward {}'.format(episode+1, episode_reward))

        print('portfolio {}, shares {}'.format(env.portfolio, env.share_of_each_stock))

        cr = (np.sum(env.portfolio) - np.sum(env.initial_portfolio)) / np.sum(env.initial_portfolio) * 100

        sr = np.mean(rho_t) / np.std(rho_t) * np.sqrt(denom)

        CR.append(cr)
        SR.append(sr)


        print('episode {}, CR {}, SR {}'.format(episode + 1, cr, sr))

        print('***********************************')

        if (episode + 1) % agent.target_update_freq == 0:
            agent.target.load_state_dict(agent.policy.state_dict())
        if (episode + 1) % 100 == 0:
            plt.plot(CR, label="CR")
            plt.plot(SR, label="SR")
            plt.legend()
            plt.show() 

def run2(env, n_episodes, agent, batch_size = 64, 
        window_size = 200, write_output_every = 5000):

    rewards = []
    CR = []
    SR = []
    for episode in range(n_episodes):
        year = int(np.random.choice([2015, 2016, 2018],
                                  p=[0.2, 0.3, 0.5]))

        year = 2018
        print(year)
        env.reset(year)
        episode_length = env.stock_env.prepared_data[0][year-2015].shape[0] - env.window
        done = False
        state = env.state
        trajectory = []
        stocks_indicators = env.stock_env.build_matrix(year, 0)

        state = [stocks_indicators, env.state]
        episode_reward = 0

        # denom = 30
        rho_t = []
        print(episode)
        for t in range(1, episode_length):
            action = agent.get_action(state)
            new_state, reward = env.step(t, action)
            stocks_indicators = env.stock_env.build_matrix(year, t)
            new_state = [stocks_indicators, new_state]
            # new_state = np.hstack([indicator_matrix, new_state])
            if t == episode_length - 1:
                done = True

            trajectory.append([state, action, reward, new_state, done])
            state = new_state
            episode_reward += reward
            rho_t.append(reward - 0.0001)
        rewards.append(episode_reward)

        print('episode {}, reward {}'.format(episode+1, episode_reward))

        print('portfolio {}, shares {}'.format(env.portfolio, env.share_of_each_stock))

        cr = (np.sum(env.portfolio) - np.sum(env.initial_portfolio)) / np.sum(env.initial_portfolio) * 100

        sr = np.mean(rho_t) / np.std(rho_t) * np.sqrt(episode_length)
        CR.append(cr)
        SR.append(sr)

        # print(agent.actor.encoder1.lstm.parameters())
        print('episode {}, CR {}, SR {}'.format(episode + 1, cr, sr))

        print('***********************************')

        agent.update(trajectory)
        if (episode + 1) % 500 == 0:
            plt.plot(CR, label="CR")
            plt.plot(SR, label="SR")
            plt.legend()
            plt.show() 

