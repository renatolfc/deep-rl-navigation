# "Deep" Reinforcement Learning for Navigation

This repository implements an agent that solves a simplified version of the
[Banana Collector](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/master/docs/images/banana.png)
environment of the [Unity ML agents](https://github.com/Unity-Technologies/ml-agents) framework.

## The environment

In particular, in this repository we solve the custom Udacity-built environment
created as part of the Deep Reinforcement Learning nanodegree, whose original
repository can also be [found on
github](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

 * Set-up: In this environment, there is a single agent located in a box world in which
   the agent has to collect bananas.
 * Goal: The agent must learn to move to as many yellow bananas as possible,
   while avoiding blue bananas
 * Agents: Only a single agent is linked to a single brain
 * Reward function:
  * +1 for "collecting" (colliding with) yellow bananas
  * -1 for "collecting" (colliding with) blue bananas
 * Observation space (some would call state space, but these are not
   necessarily the same): a continuous set of 37 variables, containing the
   agent's velocity, along with ray-based perception of objects around the
   agent's forward direction.
 * Action space: four discrete actions:
  * 0: walk forward
  * 1: walk backward
  * 2: turn left
  * 3: turn right

# Getting Started

This project requires Python 3.6+. To install all dependencies, ensure you are
on an activated [virtualenv](https://virtualenv.pypa.io/en/stable/).

Once that's the case, performing the command `pip install -r requirements.txt`
will install all software dependencies to run the code.

Users of macOS and Linux should be good after installing requirements. If
execution fails for some reason, unzipping the files Banana.app.zip or
Banana_Linux.zip should fix any issues found.

## Requirements for Windows users

Windows users should also download the Banana environment for your appropriate
architecture
([32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
or
[64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)).
If you don't know whether you're running a 32-bit or a 64-bit system, please
proceed to [Microsoft
support](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64).

# Running the code

## Training the agent

With all requirements installed, the preferred method for training the agent is
by executing the `train.py` module. The easiest way to run it is by calling

```
python -m deeprl.train
```

In the command line. For additional arguments, please run `python -m
deeprl.train --help`.


## Evaluating the agent

Similarly, to evaluate the agent, please call `python -m deeprl.eval`. The code
will search for a file called `checkpoint.pth` in the current working
directory. You can specify other paths by using the command line interface.
Failure to load a checkpoint file will imply in evaluating a random agent.


## Jupyter notebook

As an alternative, you can call the code interactively in the
[Navigation](Navigation.ipynb) notebook.
