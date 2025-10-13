# Neuroevolution

## Table of Contents

1. Overview
2. Agent roles
3. Agent networks
4. Agent inputs

## 1. Overview

### a. High-level description

This directory hosts neuroevolution methods that we are developping.

All methods rest upon a genetic algorithm with no crossover operation and 50% truncation selection.

Every iteration of the algorithm is characterized by three stages:
- variation: agents in the population are randomly mutated. If agents are mutators (explained below), they then ought to alter themselves.
- evaluation: agents perform a given task and are assigned a fitness score.
- selection: agents with the top 50% fitness scores are selected and duplicated, taking over the population slots of the lower 50% scoring agents.

### b. Implementation details

* All computations, over the three stages, are batched, meaning that computation is applied on the population as a whole.
* The 50% truncation selection is fixed. 

## 2. Agent roles

### High-level description

For a given algorithmic run, agents in the population can take up to three, non-exclusive roles:
- actors: take "actions" in their environment (e.g. steer right, select what word to say next, ...).
- discriminators: compare the behaviour of actor agents with some target behaviour.
- mutators: observe their inner workings to decide which parts of themselves to update.

In the simplest optimization setting, population agents are only actors. They optimize a hand-crafted metric.
In adversarial behaviour imitation settings, population agents are both actors and discriminators.

When agents are not mutators, changes in agents' architectures and parameters solely occur through random mutations.
When agents are mutators, agents also get to pick out some changes to apply to themselves.

### Implementation details

* For a given algorithmic run, all agents possess the same roles.

## 3. Agent networks

### Overivew

Agents make use of neural networks to perform computations.

We experiment with two modes:
1. The agents' neural network architecture is static. Only its parameters can be perturbed.
2. The agents' neural network architecture is dynamic.

### Implementation

* In practice, all networks in the population, regardless of mode, are maintained in population-wise tensors.

## 4. Agent inputs

### Overview

Agents receive input values either from:
- environments: e.g. CartPole observations in OpenAI Gym.
- Deep Learning network outputs/activations.
