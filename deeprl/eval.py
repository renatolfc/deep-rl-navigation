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


def evaldqn(env, checkpointfn='checkpoint.pth'):
    """Function that uses Deep Q Networks to learn environments.

    Parameters
    ----------
        env: Environment
            execution environment
        checkpointfn: str
            Name of the file to load network parameters
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = Agent.load(checkpointfn)

    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done, False)
        state = next_state
        score += reward
        if done:
            break
    logging.info(
        'Final score: %g', score
    )
    return score

def main():
    parser = argparse.ArgumentParser(description='Evaluates a learned agent')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store',
                        default='checkpoint.pth')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = load_environment()
    evaldqn(env, checkpointfn=args.checkpoint)

if __name__ == '__main__':
    main()
