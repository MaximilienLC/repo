# TODO

This directory is meant to host methods that I have came up with.

All methods rest upon a 50% truncation selection genetic algorithm with no crossovers.

## Agent roles

There are three types of non-exclusive roles that agents in the population can posess:
- actors: take "actions" in their environment (e.g. steer right, select what word to say next, ...)
- discriminators: compare the behaviour of other actors with some target behaviour
- mutators: observe their inner workings to decide which parts of themselves to update

In simple optimization settings, population agents are only actors.
In behaviour imitation settings, population agents are both actors and discriminators.

When agents are not mutators, all mutations performed on agents are random.
When agents are mutators, mutations performed on agents are both random and decided by themselves.

## Agents inner workings

Agents make use of neural networks to perform computations.

There are two types of neural networks:
- static: only weights and biases are mutated
- dynamic: the architecture also mutates.

## Agent environments

Agents receive one of two forms of input values:
- environments:
- deep learning networks that

## Misc

All rank 1+ tensors ought to be processed in pytorch, with all matrix operations being performed on the select GPUs.

# Instructions

Have
```
# ruff: noqa  # noqa: PGH004
# type: ignore
```
as python file header

