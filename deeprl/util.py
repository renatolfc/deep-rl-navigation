#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pygame
import inspect
import logging
import zipfile
import platform

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_agg as agg
from matplotlib.ticker import FuncFormatter, MaxNLocator

from unityagents import UnityEnvironment

try:
    from IPython.display import Markdown, display
except Exception:
    logging.exception('Failed to import jupyter modules. Ignoring')

here = os.path.dirname(os.path.abspath(__file__))

STACK_SIZE = 4
FRAME_SKIP = 1
PYGAME_INITIALIZED = False
VIEW_RESOLUTION = 1280, 720
ACTIONS = {
    0: '↑',
    1: '↓',
    2: '←',
    3: '→',
}


class UnityEnvironmentWrapper(object):
    def __init__(self, env, frameskip=FRAME_SKIP):
        self.env = env
        self.frameskip = frameskip

    def step(self, action):
        for i in range(self.frameskip):
            env = self.env.step(action)
        return env

    def __getattr__(self, attr):
        return getattr(self.env, attr)


def get_state(env_info, use_visual):
    if use_visual:
        state = env_info.visual_observations[0].transpose(0, 3, 1, 2)
    else:
        state = env_info.vector_observations[0]
    return state


def tick_formatter(tick_val, tick_pos):
    return ACTIONS.get(tick_val, '')


def show_agent(state, next_state, action, screen):
    global PYGAME_INITIALIZED
    if not PYGAME_INITIALIZED:
        pygame.init()
        screen = pygame.display.set_mode(VIEW_RESOLUTION, pygame.DOUBLEBUF)
        PYGAME_INITIALIZED = True

    fig = plt.figure(0, figsize=(VIEW_RESOLUTION[0]/96, VIEW_RESOLUTION[1]/96), dpi=96)

    for i in range(4):
        ax = plt.subplot2grid((3, 5), (0, i), rowspan=2)
        ax.imshow(state[:, i, :, :].transpose(1, 2, 0))
        ax.set_title('Frame - %d' % (3 - i))
    ax = plt.subplot2grid((3, 5), (0, 4), rowspan=2)
    ax.imshow(next_state[:, -1, :, :].transpose(1, 2, 0))
    ax.set_title('Next state')

    a = np.zeros((1, 4))
    a[0, action] = 1
    ax = plt.subplot2grid((3, 5), (2, 0), colspan=5)
    ax.imshow(a, cmap='gray')
    ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    size = canvas.get_width_height()

    surf = pygame.image.fromstring(raw_data, size, "RGB")
    surf_pos = surf.get_rect()
    screen.blit(surf, surf_pos)
    pygame.display.update()
    plt.close(fig)


def build_environment(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        zipped = directory + '.zip'
        if os.path.exists(zipped):
            with zipfile.PyZipFile(zipped) as zfp:
                logging.info('Extracting environment...')
                zfp.extractall()
        else:
            raise Exception('Unable to proceed. Cannot find environment.')
    os.chmod(path, 0o755)
    return UnityEnvironment(file_name=path)

def get_executable_path():
    parent = os.path.dirname(here)
    if platform.system() == 'Linux':
        if '64' in platform.architecture()[0]:
            return os.path.join(parent, 'VisualBanana_Linux', 'Banana.x86_64')
        else:
            return os.path.join(parent, 'VisualBanana_Linux', 'Banana.x86')
    elif platform.system() == 'Darwin':
            return os.path.join(parent, 'Banana.app')
    elif platform.system() == 'Windows':
        if '64' in platform.architecture()[0]:
            return os.path.join(parent, 'Banana_Windows_x86_64', 'Banana.exe')
        else:
            return os.path.join(parent, 'Banana_Windows_x86', 'Banana.exe')
    else:
        logging.error('Unsupported platform!')
        raise ValueError('Unsupported platform')

def print_source(obj):
    source = inspect.getsource(obj)
    display(Markdown('```python\n' + source + '\n```'))

def load_environment():
    path = get_executable_path()
    return build_environment(path)
