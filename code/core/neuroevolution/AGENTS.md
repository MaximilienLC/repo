# Neuroevolution

## Overview

### High-level description

This directory hosts neuroevolution methods that we are developing.

All methods rest upon a genetic algorithm with no crossovers and 50% truncation selection.

Every iteration of the algorithm is characterized by three stages:

- variation: agents in the population are randomly mutated. If agents are mutators (explained below), they then ought to alter themselves afterwards.
- evaluation: agents perform a given task and are assigned a fitness score.
- selection: agents with the top 50% fitness scores are selected and duplicated, taking over the population slots of the lower 50% scoring agents.

### Implementation details

Computations are applied on the population as a whole, with each agent component being one of its components.

All three iteration stages are then ran on the population tensor all at once.

## Agent roles

### High-level description

For a given algorithmic run, population agents can take up to three, non-exclusive roles:
- actors: take "actions" in their environment (e.g. steer right, select what word to say next, ...).
- discriminators: compare the behaviour of other actors with some target behaviour.
- mutators: observe their inner workings to decide which parts of themselves to update.

In the simplest optimization setting, population agents are only actors. They optimize a hand-crafted metric.
In adversarial behaviour imitation settings, population agents are both actors and discriminators.

When agents are not mutators, agents only change their architecture and parameters through random mutations.
When agents are mutators, agents also pick out some changes to apply to themselves.

### Implementation details

TODO

## Agent networks

### Overivew

Agents make use of neural networks to perform computations.

We experiment with two modes:
1. The agents' neural network architecture is frozen. Only its parameters can be perturbed.
2. The agents' neural network architecture is dynamic.

### Implementation

In practice, all networks in the populations are maintained in a large 

TODO

## Agent inputs

Agents receive input values either from:
- environments: e.g. CartPole observations in OpenAI Gym.
- deep learning network outputs/activations.

