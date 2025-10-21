# Neuroevolution

## Table of Contents
- [Neuroevolution](#neuroevolution)
  - [Table of Contents](#table-of-contents)
  - [I. Overview](#i-overview)
    - [1. Description](#1-description)
    - [2. Details](#2-details)
  - [II. Agent roles](#ii-agent-roles)
    - [1. Description](#1-description-1)
  - [III. Agent networks](#iii-agent-networks)
    - [1. Description](#1-description-2)
      - [A. Dynamically complexifying networks](#a-dynamically-complexifying-networks)
        - [i. Structure](#i-structure)
        - [ii. Original structure](#ii-original-structure)
        - [iii. Mutations](#iii-mutations)
        - [iv. Representation for computation](#iv-representation-for-computation)
        - [v. Node computation](#v-node-computation)
        - [vi. Computation pipeline](#vi-computation-pipeline)
    - [2. Details](#2-details-1)
  - [IV. Agent playground](#iv-agent-playground)
    - [1. Description](#1-description-3)
    - [2. Details](#2-details-2)

## I. Overview

### 1. Description

This directory hosts **neuroevolution** methods that we are developping.

All methods rest upon a **genetic algorithm** with **no crossover** operation and a fixed **50% truncation selection**.

Every iteration of the algorithm is characterized by three stages:
- **variation**: agents in the population are **randomly perturbed**. If agents are **mutators** (explained below), they then ought to **alter themselves**.
- **evaluation**: agents **perform a given task** and are **assigned a fitness score**.
- **selection**: agents with the **top 50% fitness scores** are **selected and duplicated**, taking over the population slots of the lower 50% scoring agents.

### 2. Details

* In order to best leverage GPU computation, the **population is the point of focus**: agents are mostly abstracted away from the implementation. Agents' information is instead held in indexable population-wise tensors.

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

Leveraging the flexibility of genetic algorithms, we focus on neural networks with **dynamically complexifying architectures** (more info `@DYNAMIC NETWORKS.md`), however we also implement standard **static architecture** neural networks.

#### A. Dynamically complexifying networks

Disclaimer: the words `node` and `neuron` are used interchangeably in this document.

##### i. Structure

The networks have three types of neurons: input, hidden and output.
Input and output neurons have no biases.

##### ii. Original structure

The network begins with one input node per value it inputs, one output node per value it outputs, and no hidden neurons.

A network's original structure differs whether or not it is remapping some signal.

In all cases, it begins with no hidden neurons.

If no deep learning, there is no original connection between the input and output nodes.
Since the first variation stage occurs before the first evaluation stage, the network will have an opportunity to grow neural circuits at that point.

If not, the network starts with 

##### iii. Mutations

These networks are dynamic in structure: they contract and expand through two mutation types: `grow_node` and `prune_node`.

##### iv. Representation for computation

N/A.

##### v. Node computation

A node N performs a simple `tanh(wx+b)` with `x` being a vector concatenating outputs from the nodes that connect to node N (which includes node N itself).

In practice, all nodes in the network

Input, hidden, output neurons

if mapping from environment input space to output space:
start with no hidden neuron, and no connection between neurons.

if mapping from deep learning model output space back into output space:
start with no hidden neuron, and 

##### vi. Computation pipeline

(All computations are batched over the entire population)

1. Pre-network processing

This is where operations like standardization happen over the environment input vectors (or deep learning model output vectors).

2. Network processing

The results of Step 1 are multiplied against the sparse matrix representations of the networks' weights. The networks' biases are added to the results.

3. Post-network processing

This is where operations like producing a discrete action with `argmax` happen.

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