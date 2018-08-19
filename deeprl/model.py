#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
