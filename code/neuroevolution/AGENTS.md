# Neuroevolution

## Introduction

This directory is meant to host neuroevolution methods that I am developing.

All methods rest upon a genetic algorithm with no crossovers and 50% truncation selection.

I like to divide every iteration into three stages:

- variation: agents are randomly mutated. If agents are mutators (explained below), then they ought to alter themselves afterwards.
- evaluation: agents perform a given task and are assigned a fitness score.
- selection: the agents with the top 50% fitness scores are selected and duplicated over the slots of the lower 50% scoring agents.

## Agent roles

There are three types of non-exclusive roles that agents in the population can posess:
- actors: take "actions" in their environment (e.g. steer right, select what word to say next, ...)
- discriminators: compare the behaviour of other actors with some target behaviour
- mutators: observe their inner workings to decide which parts of themselves to update

In the simplest optimization setting, population agents are only actors. They optimize a hand-crafted metric.
In adversarial behaviour imitation settings, population agents are both actors and discriminators.

When agents are not mutators, agents only change their architecture and parameters through random mutations.
When agents are mutators, agents also pick out some changes to apply to themselves.

## Agents inner workings

Agents make use of neural networks to perform computations.

There are two types of neural networks:
- static: only weights and biases are mutated
- dynamic: the architecture also mutates.

## Agent inputs

Agents receive input values either from:
- environments: e.g. CartPole observations in OpenAI Gym.
- deep learning network outputs/activations.

## Misc

All rank 1+ tensors ought to be processed in PyTorch, with all matrix operations being performed on the select GPUs.