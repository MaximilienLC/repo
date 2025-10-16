# Neuroevolution

## Table of Contents

1. Overview
2. Agent roles
3. Agent networks
4. Agent playground

## 1. Overview

### a. Description

This directory hosts neuroevolution methods that we are developping.

All methods rest upon a genetic algorithm with no crossover operation and 50% truncation selection.

Every iteration of the algorithm is characterized by three stages:
- variation: agents in the population are randomly mutated. If agents are mutators (explained below), they then ought to alter themselves.
- evaluation: agents perform a given task and are assigned a fitness score.
- selection: agents with the top 50% fitness scores are selected and duplicated, taking over the population slots of the lower 50% scoring agents.

### c. Details

* The 50% truncation selection is fixed.
* In order to best leverage GPU computation, the population is the point of focus: agents are mostly abstracted away from the implementation. Agents' information is held in indexable population-wise tensors.

## 2. Agent roles

### a. Description

For a given algorithmic run, agents in the population can take up to three, non-exclusive roles:
- actors: take "actions" in their environment (e.g. steer right, select what word to say next, ...).
- discriminators: compare the behaviour of actor agents with some target behaviour.
- mutators: observe their inner workings to decide which parts of themselves to update.

In the simplest optimization setting, population agents are only actors. They optimize a hand-crafted metric.
In adversarial behaviour imitation settings, population agents are both actors and discriminators.

When agents are not mutators, changes in agents' architectures and parameters solely occur through random mutations.
When agents are mutators, agents also get to pick out some changes to apply to themselves.

### b. Details

* For a given algorithmic run, all agents possess the same roles.

## 3. Agent networks

### a. Description

Agents make use of neural networks to perform computations.

We experiment with two modes:
1. The agents' neural network architectures are static. Only parameters can be perturbed.
2. The agents' neural network architectures are dynamic. More info `@DYNAMIC NETWORKS.md`.

### b. Details

* Given that networks are components of agents, they also are maintained in population-wide tensors. During evaluation for instance, input information of format `population_size x (input_dimensions)` is exposed to a weight metrix of format `population_size x (weight_dimensions)` through operations the likes of `torch.bmm`.

## 4. Agent playground

### a. Description

Agents operate in environments. They either:
1) Operate independentely in the environment (e.g. on tasks like CartPole that have low-dimensional input/output spaces)
or
2) Operate through processed Deep Learning model outputs/activations. In the simplest of cases agents map from the DL model action/output space back into that same space.

### b. Details

* Environments are always maintained in a `torchrl.envs.ParallelEnv`.