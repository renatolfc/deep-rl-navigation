#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import logging
import numpy as np
from collections import namedtuple, deque

import torch
import torch.optim as optim
import torch.nn.functional as F

from .model import QNetwork

BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64  # Minibatch size
GAMMA = 0.99  # Discount factor
TAU = 1e-3  # target parameters soft update
LR = 5e-4  # learning rate
UPDATE_EVERY = 5  # how often to update the network


class DeviceAwareClass(object):
    'Sets appropriate device for computation based on GPU availability.'

    if torch.cuda.is_available():
        logging.info(
            'Using first CUDA device for computation. '
            'Tune visibility by setting the CUDA_VISIBLE_DEVICES env var.'
        )
        device = torch.device('cuda:0')
    else:
        logging.info(
            'Solely using CPU for computation.'
        )
        device = torch.device('cpu')


class Agent(DeviceAwareClass):
    'Interacts with and learns from the environment'

    def __init__(self, state_size: int, action_size: int, seed: int):
        '''Initializes an Agent object.

        Parameters
        ----------
            state_size: int
                Dimension of each state
            action_size: int
                Dimension of each action
            seed: int
                Random seed to use
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network(s) {{{
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        # }}}

        # Gradient descent optimizer {{{
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # }}}

        # Replay memory {{{
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # }}}

        # Current time step
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps {{{
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Only sample experiences if they can fill a batch
            # (makes sure we don't try to learn at the very beginning)
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        # }}}

    def act(self, state, epsilon=0.0):
        '''Returns actions for a given state as per current policy.

        Parameters
        ----------
            state: array_like
                current state the agent is in
            epsilon: float
                probability of selecting random actions in epsilon-greedy
        '''

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Policy evaluation {{{
        self.qnetwork_local.eval()
        with torch.no_grad():
            # We don't want to compute gradients here, since we're evaluating
            # our policy. 
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # }}}

        # Epsilon-greedy action selection
        if random.random() > eps:
            return randargmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''Updates value parameters using given batch of experience tuples.

        Parameters
        ----------
            experiences: Tuple[torch.Tensor]
                tuple of (s, a, r, s', done) tuples
            gamma: float
                Discount factor
        '''
        states, actions, rewards, next_states, dones = experiences

        # Estimates the TD target R + γ max_a q(S′, a, w−) {{{
        targets_next = self.qnetwork_target.forward(next_states).max(1)[0].unsqueeze(1)
        # By definition, all future rewards after reaching a terminal states are zero.
        # Hence, we use the `dones` booleans to properly assign value to states.
        targets = rewards + (gamma * targets_next * (1 - dones))
        # }}}

        # Now we get what our current policy thinks are the values of the actions
        # we've taken in the past
        estimated = self.qnetwork_local.forward(states).gather(1, actions)

        # We want to minimize the MSE (although this part is super confusing in
        # the DQN paper. Some people say this should be the Huber loss, but that's
        # not what I understood from the code attached to the paper.
        # For more context:
        # https://stackoverflow.com/a/43720168
        loss = F.mse_loss(estimated, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Updating the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Goes slowly from the local weights to the target weights. Is this
        a contraction? 🤔

        Parameters
        ----------
            local_model: PyTorch model
                Model from which weights will be copied
            target_model: PyTorch model
                Model to which weights will be copied
            tau: float
                step size
        '''

        for target, local in zip(target_model.parameters(), local_model.parameters()):
            target.data.copy_(tau * local.data + (1.0 - tau) * target.data)


class ReplayBuffer(DeviceAwareClass):
    'Fixed-size buffer to store experience tuples.'

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initializes a ReplayBuffer object.

        Parameters
        ----------
            action_size: int
                dimention of each action
            buffer_size: int
                maximum size of buffer
            batch_size: int
                size of each training batch
            seed: int
                random seed
        '''

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            'Experience',
            field_names='state action reward next_state done'.split()
        )

    def add(self, state, action, reward, next_state, done):
        'Adds a new experience to memory.'

        self.memory.append(
            self.experience(state, action, reward, next_state, done)
        )

    def sample(self):
        'Randomly sample a batch of experiences from memory.'

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        'Returns the current size of the buffer.'

        return len(self.memory)


def randargmax(a):
    return np.random.choice(np.flatnonzero(a == a.max()))
