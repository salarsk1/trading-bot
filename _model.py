import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from _utils import *
from _environment import *

__all__ = ["Critic", "Actor", "DQN"]

class Critic(nn.Module):
    def __init__(self, num_indc, lstm_hidden, hidden_layers):
        
        super(Critic, self).__init__()

        self.num_indc = num_indc

        self.lstm_hidden = lstm_hidden

        self.encoder1 = Encoder(self.num_indc, self.lstm_hidden)

        self.encoder2 = Encoder(self.num_indc, self.lstm_hidden)

        self.encoder3 = Encoder(self.num_indc, self.lstm_hidden)

        self.hidden_layers = hidden_layers

        self.linear = nn.ModuleList()

        self.input_size = self.lstm_hidden * 3 + 4
        
        self.linear.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        
        for i in range(1, len(self.hidden_layers)):
            self.linear.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))

        self.linear.append(nn.Linear(self.hidden_layers[-1], 1))

    def forward(self, state, action):
        """
        state and action parameters are torch tensors
        """
        x = torch.cat([state, action], 1)

        for layer in range(len(self.linear)-1):
            x = F.relu(self.linear[layer](x))
            x = nn.Dropout(0.3)(x)

        x = self.linear[-1](x)

        return x

class Actor(nn.Module):
    def __init__(self, window_size, hidden_layers, output_size):
        
        super(Actor, self).__init__()
        
        self.window_size = window_size 
        
        self.hidden_layers = hidden_layers

        self.output_size = output_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Dropout(p = 0.2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 2, stride=1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (1,2), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 1),
            nn.Dropout(p = 0.2))

        self.linear1 = nn.Sequential(
            nn.Linear(512+4, self.hidden_layers[0]),
            nn.Tanh(),
            nn.Dropout(0.2))
        
        self.linear2 = nn.Sequential(
            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
            nn.Tanh(),
            nn.Dropout(0.2))

        self.linear3 = nn.Sequential(
            nn.Linear(self.hidden_layers[1], self.output_size))
        
    def forward(self, s, w):
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        s = s.view(s.size(0), 1, -1)
        feature = torch.cat([s, w], axis=2)
        feature = self.linear1(feature)
        feature = self.linear2(feature)
        out     = self.linear3(feature)
        return out



class DQN(nn.Module):

    def __init__(self, window_size, input_size, lstm_hidden, hidden_layers, output_size):
        
        super(DQN, self).__init__()

        self.window_size = window_size
        
        self.input_size = input_size

        self.lstm_hidden = lstm_hidden
        
        self.hidden_layers = hidden_layers

        self.output_size = output_size

        self.linear = nn.ModuleList()

        self.encoder1 = Encoder(5, self.lstm_hidden)
        

        self.encoder2 = Encoder(5, self.lstm_hidden)
        

        self.encoder3 = Encoder(5, self.lstm_hidden)
        

        self.linear.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        
        for i in range(1, len(self.hidden_layers)):
            self.linear.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))

        self.linear.append(nn.Linear(self.hidden_layers[-1], self.output_size))

    def forward(self, S1, S2, S3, W):
        h1 = self.encoder1(S1).squeeze(0)
        h2 = self.encoder2(S2).squeeze(0)
        h3 = self.encoder3(S3).squeeze(0)
        
        state = torch.cat([h1, h2, h3, W], axis=1)
        for layer in range(len(self.linear)-1):
            state = F.relu(self.linear[layer](state))
            state = F.dropout(state, 0.3)

        return self.linear[-1](state)


# class DQN(nn.Module):

#     def __init__(self, input_size, hidden_layers, output_size):
        
#         super(DQN, self).__init__()
        
#         self.input_size = input_size
        
#         self.hidden_layers = hidden_layers

#         self.output_size = output_size

#         self.linear = nn.ModuleList()

#         self.linear.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        
#         for i in range(1, len(self.hidden_layers)):
#             self.linear.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))

#         self.linear.append(nn.Linear(self.hidden_layers[-1], self.output_size))

#     def forward(self, state):

#         state = torch.FloatTensor(state)
#         for layer in range(len(self.linear)-1):
#             state = F.relu(self.linear[layer](state))
#             state = F.dropout(state, 0.3)

#         # return F.softmax(self.linear[-1](softmaxte), dim=1)
#         return self.linear[-1](state)

# The encoder class to get the representation of the time series on 
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers=1):

        super(Encoder, self).__init__()

        self.input_size = input_size

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers, batch_first = True)

    def forward(self, inputs):
        out, (self.hidden, cell) = self.lstm(inputs)#.reshape(-1, -1, self.input_size)
        return self.lstm(inputs)

if __name__ == "__main__":

    actor = Actor(10, [100, 100], 27)
    stock = ['MSFT', 'JPM', 'WMT']
    prep  = StockPrep(stock, 10)
    data  = prep.prepare_data()
    inds  = torch.FloatTensor(prep.build_matrix(2017, 10)).unsqueeze(0)
    w = torch.rand(1,1,4)
    print(actor.forward(inds, w))

# num_indc, window_size, lstm_hidden, hidden_layers, output_size
    








