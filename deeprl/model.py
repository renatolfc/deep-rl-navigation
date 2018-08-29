#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    'Policy Model for solving RL tasks with deep learning.'

    def __init__(self, state_size: int, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: int
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):
        'Runs the state through the network to generate action values.'

        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x

class AdvantageNetwork(nn.Module):
    'Policy Model with Dueling Networks for solving RL tasks with deep learning.'


    def __init__(self, state_size: int, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: int
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)

        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, action_size)

    def forward(self, state):
        'Runs the state through the network to generate action values.'

        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        value = self.value(x)
        advantage = self.advantage(x)

        return value + (advantage - advantage.mean())


class VisualAdvantageNetwork(nn.Module):
    'Policy Model with Dueling Networks for solving RL tasks with deep learning.'


    def __init__(self, state_size: tuple, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: tuple
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        _, self.width, self.height, self.channels = state_size

        self.conv1 = nn.Conv2d(self.channels, 32, 4, 2)
        self.h1, self.w1 = self._conv_size(self.width, self.height, 4, 0, 1, 2)

        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.h2, self.w2 = self._conv_size(self.h1, self.w1, 4, 0, 1, 2)

        self.conv3 = nn.Conv2d(64, 128, 4, 2)
        self.h3, self.w3 = self._conv_size(self.h2, self.w2, 4, 0, 1, 2)

        self.value_fc = nn.Linear(self.h3 * self.w3 * 128, 512)
        self.advantage_fc = nn.Linear(self.h3 * self.w3 * 128, 512)

        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_size)

    def _conv_size(self, h, w, kernel_size, padding, dilation, stride):
        return math.floor((h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1), \
                math.floor((w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def forward(self, state):
        'Runs the state through the network to generate action values.'

        state = state.view(state.shape[0], self.channels, self.height, self.width)

        x = F.leaky_relu(self.conv1(state))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size()[0], -1)

        value = self.value(F.leaky_relu(self.value_fc(x)))
        advantage = self.advantage(F.leaky_relu(self.advantage_fc(x)))

        return value + (advantage - advantage.mean())


class VisualQNetwork(nn.Module):
    'Policy Model with Dueling Networks for solving RL tasks with deep learning.'

    def __init__(self, state_size: tuple, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: tuple
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        _, self.width, self.height, self.channels = state_size

        self.conv1 = nn.Conv2d(self.channels, 32, 8, 4)
        self.h1, self.w1 = self._conv_size(self.width, self.height, 8, 0, 1, 4)

        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.h2, self.w2 = self._conv_size(self.h1, self.w1, 4, 0, 1, 2)

        self.conv3 = nn.Conv2d(64, 128, 4, 2)
        self.h3, self.w3 = self._conv_size(self.h2, self.w2, 4, 0, 1, 2)

        self.fc1 = nn.Linear(self.h3 * self.w3 * 128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, action_size)

    def _conv_size(self, h, w, kernel_size, padding, dilation, stride):
        return math.floor((h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1), \
                math.floor((w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def forward(self, state):
        'Runs the state through the network to generate action values.'

        state = state.view(state.shape[0], self.channels, self.height, self.width)

        x = F.leaky_relu(self.conv1(state))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        return self.fc3(x)
