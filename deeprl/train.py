#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import argparse
from collections import deque

import torch
import numpy as np

from .util import load_environment
from .agent import Agent


def dqn(env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.001,
        eps_decay=0.995, solution_threshold=13.0, checkpointfn='checkpoint.pth'):
    """Function that uses Deep Q Networks to learn environments.

    Parameters
    ----------
        n_episodes: int
            maximum number of training episodes
        max_t: int
            maximum number of timesteps per episode
        eps_start: float
            starting value of epsilon, for epsilon-greedy action selection
        eps_end: float
            minimum value of epsilon
        eps_decay: float
            multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = Agent(state_size, action_size, 123)

    scores = []  # All episodes seen over training
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        logging.debug(
            'Episode {}\tAverage Score: {:.2f}'
            .format(i_episode + 1, np.mean(scores_window))
        )
        if i_episode % 100 == 0:
            logging.info(
                'Episode {}\tAverage Score: {:.2f}'
                .format(i_episode + 1, np.mean(scores_window))
            )
        if np.mean(scores_window) >= solution_threshold:
            logging.info(
                'Environment solved in {:d} episodes!'
                .format(i_episode - 99)
            )
            logging.info(
                'Saving checkpoint file at %s', checkpointfn
            )
            agent.save(checkpointfn)
            break
    return agent, scores, i_episode - 99

def main():
    parser = argparse.ArgumentParser(description='Trains a learning agent')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store',
                        default='checkpoint.pth')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = load_environment()
    dqn(env, checkpointfn=args.checkpoint)

if __name__ == '__main__':
    main()
