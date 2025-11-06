# neuroevolution/

## Table of Contents
- [neuroevolution/](#neuroevolution)
  - [Table of Contents](#table-of-contents)
  - [I. Overview](#i-overview)
    - [1. Description](#1-description)
    - [2. Details](#2-details)
  - [II. Agent roles](#ii-agent-roles)
    - [1. Description](#1-description-1)
  - [III. Agent networks](#iii-agent-networks)
    - [1. Description](#1-description-2)
      - [A. Dynamically complexifying networks](#a-dynamically-complexifying-networks)
    - [2. Details](#2-details-1)
  - [IV. Agent playground](#iv-agent-playground)
    - [1. Description](#1-description-3)
    - [2. Details](#2-details-2)

## I. Overview

### 1. Description

This directory hosts **neuroevolution** methods (**evolutionary algorithms** optimizing **artificial neural networks**)  that we are developping.

Every iteration of any of these algorithms is characterized by three stages:
- **variation**: agents in the population are **randomly perturbed**. If agents are **mutators** (explained below), they then ought to **alter themselves**.
- **evaluation**: agents **perform a given task** and are **assigned a fitness score**.
- **selection**: given the **fitness scores**, certain low performing agents are replaced by duplicates of high performing agents.

We implement various **evolutionary algorithms**:
- `simple_ga`: during selection, agents with the **top 50% fitness scores** are **selected and duplicated**, taking over the population slots of the lower 50% scoring agents.
- `simple_es`: during selection, the population's fitness scores are standardized and turned into a probability distribution through a softmax operation. A single agent is then created through a weighted sum of existing agents' parameters. It is then duplicated over the entire population size.

### 2. Details

* In order to most naturally leverage GPU computation, the **population is the point of focus** in this codebase: agents are mostly abstracted away from the implementation. Agents' information is instead held in indexable population-wise tensors.

## II. Agent roles

### 1. Description

For a given algorithmic run, agents in the population can take **up to three, non-exclusive roles**:
- **actors**: take "actions" in their environment (e.g. steer right, select what word to say next, ...).
- **discriminators**: compare the behaviour of actor agents with some target behaviour.
- **mutators**: observe their inner workings to decide which parts of themselves to update.

In the simplest optimization setting, population agents are **actors**. They optimize a **hand-crafted metric**.
In **adversarial behaviour imitation** settings, population agents are both **actors and discriminators**.

When agents are **not mutators**, changes in agents' architectures and parameters **solely** occur through **random perturbations**.
When agents are **mutators**, agents also get to **pick out some changes to apply to themselves**.

## III. Agent networks

### 1. Description

Agents make use of neural networks to perform computations.

Leveraging the flexibility of **genetic algorithms**, we heavily focus on neural networks with **dynamically complexifying architectures** ([more info](#a-dynamically-complexifying-networks)), however we also implement standard **static architecture** neural networks.

#### A. Dynamically complexifying networks

`@---III.1.A.DYNAMIC_NETWORKS.md`

### 2. Details

* Given that networks are components of agents, they are **also maintained in population-wide tensors**. During evaluation for instance, input information of format `population_size x (input_dimensions)` is exposed to a weight metrix of format `population_size x (weight_dimensions)` through operations the likes of `torch.bmm()`.

## IV. Agent playground

### 1. Description

Agents operate in environments. They either:
1) Operate independentely in the environment (e.g. on tasks like CartPole that have low-dimensional input/output spaces)
or
2) Operate through processed Deep Learning model outputs/activations. In the simplest of cases agents map from the DL model action/output space back into that same space.

### 2. Details

* Environments are always maintained in a `torchrl.envs.ParallelEnv`.