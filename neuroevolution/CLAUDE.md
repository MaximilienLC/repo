
- [Specifications](#specifications)
  - [Overview](#overview)
    - [Description](#description)
      - [Evolution stages](#evolution-stages)
      - [Types of evolutionary algorithms](#types-of-evolutionary-algorithms)
    - [Details](#details)
  - [Agent roles](#agent-roles)
    - [Description](#description-1)
      - [Optimization types](#optimization-types)
    - [Mutator role and variation stage](#mutator-role-and-variation-stage)
  - [Agent networks](#agent-networks)
    - [Description](#description-2)
      - [Dynamically complexifying networks](#dynamically-complexifying-networks)
    - [Details](#details-1)
  - [Agent playground](#agent-playground)
    - [Description](#description-3)
    - [Details](#details-2)
- [Gotchas](#gotchas)

# Specifications

## Overview

### Description

This folder hosts **neuroevolution** methods (**evolutionary algorithms** optimizing **artificial neural networks**)  that we are developping. This folder does not contain any experimentation code.

#### Evolution stages

Every iteration of any of these algorithms is characterized by three stages:
- **variation**: agents in the population are **randomly perturbed**. If agents are **mutators** (explained below), they then ought to **alter themselves**.
- **evaluation**: agents **perform a given task** and are **assigned a fitness score**.
- **selection**: given the **fitness scores**, certain low performing agents are replaced by duplicates of high performing agents.

#### Types of evolutionary algorithms

We implement various **evolutionary algorithms**:
- `simple_ga`: during selection, agents with the **top 50% fitness scores** are **selected and duplicated**, taking over the population slots of the lower 50% scoring agents.
- `simple_es`: during selection, the population's fitness scores are standardized and turned into a probability distribution through a softmax operation. A single agent is then created through a weighted sum of existing agents' parameters. It is then duplicated over the entire population size.

### Details

* In order to most naturally leverage GPU computation, the **population is the point of focus** in this codebase: agents are mostly abstracted away from the implementation. Agents' information is instead held in indexable population-wise tensors.

## Agent roles

### Description

For a given algorithmic run, agents in the population can take **up to three, non-exclusive roles**:
- **actors**: take "actions" in their environment (e.g. steer right, select what word to say next, ...).
- **discriminators**: compare the behaviour of actor agents with some target behaviour.
- **mutators**: observe their inner workings to decide which parts of themselves to update.

#### Optimization types

In the simplest optimization setting, population agents are **actors**. They optimize a **hand-crafted metric**.
In **adversarial behaviour imitation** settings, population agents are both **actors and discriminators**.

### Mutator role and variation stage

When agents are **not mutators**, changes in agents' architectures and parameters **solely** occur through **random perturbations**.
When agents are **mutators**, agents also get to **pick out some changes to apply to themselves**.

## Agent networks

### Description

Agents make use of neural networks to perform computations.

Leveraging the flexibility of **genetic algorithms**, we heavily focus on neural networks with **dynamically complexifying architectures** ([more info](#a-dynamically-complexifying-networks)), however we also implement standard **static architecture** neural networks for more general use across all our implemented **evolutionary algorithms**.

#### Dynamically complexifying networks

Implemented in `dynamic_net.py`.

### Details

* Given that networks are components of agents, they are **also maintained in population-wide tensors**. During evaluation for instance, input information of format `population_size x (input_dimensions)` is exposed to a weight metrix of format `population_size x (weight_dimensions)` through operations the likes of `torch.bmm()`.

## Agent playground

### Description

Agents operate in environments. They either:
1) Operate independentely in the environment (e.g. on tasks like CartPole that have low-dimensional input/output spaces)
or
2) Operate through processed Deep Learning model outputs/activations. In the simplest of cases agents map from the DL model action/output space back into that same space.

### Details

* Environments are always maintained in a `torchrl.envs.ParallelEnv`.

# Gotchas

N/A