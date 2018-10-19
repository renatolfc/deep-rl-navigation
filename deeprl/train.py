#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import logging
import argparse
from collections import deque

import torch
import numpy as np

from .agent import DoubleDQNAgent as Agent, BUFFER_SIZE
from .util import load_environment, UnityEnvironmentWrapper, get_state
from .util import STACK_SIZE, FRAME_SKIP

BATCH = 0
CHANNELS = 1
DEPTH = 2
HEIGHT = 3
WIDTH = 4


def rgb2gray(img):
    img = img.squeeze()
    if len(img.shape) == 3:
        return img.dot([0.299, 0.587, 0.114]).reshape(
            1, img.shape[0], img.shape[1], 1
        )
        #return (img ** 2).dot(
        #    [0.299, 0.587, 0.114]
        #).reshape(1, img.shape[0], img.shape[1], 1)
    else:
        raise ValueError('Image in some color space not known')


def reset_deque(state):
    state_deque = deque(maxlen=STACK_SIZE)

    for _ in range(STACK_SIZE):
        state_deque.append(np.zeros(
            state.shape
        ))

    return state_deque

def dqn(env, n_episodes=1001, max_t=1000 * FRAME_SKIP, eps_start=1.0,
        eps_end=0.001, eps_decay=0.995, solution_threshold=13.0,
        checkpointfn='checkpoint.pth', load_checkpoint=False,
        reload_every=None):
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

    if state_size == 0:
        use_visual = True
        initial_state = get_state(env_info, use_visual)
        state_size = list(initial_state.shape)
        state_size.insert(2, STACK_SIZE)
        state_size = tuple(state_size)

    if load_checkpoint:
        try:
            agent = Agent.load(checkpointfn, use_visual)
        except Exception:
            logging.exception('Failed to load checkpoint. Ignoring...')
            agent = Agent(state_size, action_size, 0, use_visual)
    else:
        agent = Agent(state_size, action_size, 0, use_visual)

    if agent.episode:
        eps = (eps_start * eps_decay) ** agent.episode
    else:
        eps = eps_start

    for i_episode in range(agent.episode, n_episodes):
        state_deque = reset_deque(initial_state)

        env_info = env.reset(train_mode=True)[brain_name]
        state = get_state(env_info, use_visual)
        state_deque.append(state)

        score = 0
        for t in range(max_t):
            state = np.stack(state_deque, axis=-1) \
                    .squeeze(axis=0).transpose(0, -1, 1, 2)

            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]

            next_state = get_state(env_info, use_visual)
            state_deque.append(next_state)
            next_state = np.stack(state_deque, axis=-1) \
                    .squeeze(axis=0).transpose(0, -1, 1, 2)

            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(
                state,
                action,
                reward,
                next_state,
                done,
            )

            score += reward
            if done:
                break
        agent.scores.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        agent.episode += 1

        logging.debug(
            'Episode {}\tAverage Score: {:.2f}\tCurrent Score: {:.2f}\tEpsilon: {:.4f}'
            .format(i_episode, np.mean(agent.scores[-100:]), score, eps)
        )
        if (i_episode + 1) % 100 == 0:
            logging.info(
                'Episode {}\tAverage Score: {:.2f}'
                .format(i_episode, np.mean(agent.scores[-100:]))
            )
            logging.info(
                'Saving checkpoint file...'
            )
            agent.save(checkpointfn)
        if np.mean(agent.scores[-100:]) >= solution_threshold:
            logging.info(
                'Environment solved in {:d} episodes!'
                .format(i_episode - 99)
            )
            logging.info(
                'Saving checkpoint file at %s', checkpointfn
            )
            agent.save(checkpointfn)
            break
        if reload_every and i_episode and (i_episode + 1) % reload_every == 0:
            env.close()
            reload_process()

    return agent


def reload_process():
    if '--load-checkpoint' not in sys.argv:
        sys.argv.append('--load-checkpoint')
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    os.execv('/proc/self/exe', 'python -m deeprl.train'.split() + sys.argv[1:])


def main():
    parser = argparse.ArgumentParser(description='Trains a learning agent')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store',
                        default='checkpoint.pth')
    parser.add_argument('--load-checkpoint', dest='load_chkpt', action='store_true',
                        default=False)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    env = UnityEnvironmentWrapper(load_environment())
    dqn(env, eps_decay=0.995, n_episodes=2000, reload_every=1000,
        checkpointfn=args.checkpoint, load_checkpoint=args.load_chkpt)

if __name__ == '__main__':
    main()
