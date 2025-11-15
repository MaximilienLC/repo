
- [Specifications](#specifications)
  - [Overview](#overview)
    - [Description](#description)
    - [Details](#details)
  - [Agent networks](#agent-networks)
    - [Description](#description-1)
      - [Static architecture networks](#static-architecture-networks)
      - [Dynamically complexifying networks](#dynamically-complexifying-networks)
        - [Initalization and mutation](#initalization-and-mutation)
        - [Components for computation](#components-for-computation)
    - [Details](#details-1)
  - [Agent playground](#agent-playground)
    - [Description](#description-2)
    - [Details](#details-2)
- [Gotchas](#gotchas)

# Specifications

## Overview

### Description

This folder hosts `neuroevolution` methods (`evolutionary algorithms` optimizing `artificial neural networks`) that we are developping. This folder does not contain any experimentation code.

At this point in time, we build upon a simple `genetic algorithm`. It is characterized by three stages:
- `variation`: agents in the population are `randomly perturbed`.
- `evaluation`: agents `perform a given task` and are `assigned a fitness score`.
- `selection`: agents with the `top 50% fitness scores` are `selected and duplicated`, taking over the population slots of the lower 50% scoring agents.

### Details

In order to most naturally leverage GPU computation, the `population is the point of focus` in this codebase: agents are mostly abstracted away from the implementation. Agents' information is instead held in indexable population-wise tensors.

## Agent networks

### Description

Agents make use of neural networks to perform computations.
We experiment with two types of networks: `static architecture` networks, and networks with `dynamically complexifying architectures`.

#### Static architecture networks

Static architecture networks take the shape of standard deep networks, but do not leverage their differentiation logic.

With `θ` representing all network parameters, `θᵢ` representing network parameter at index `i` (picturing all parameters flattened in a vector), we implement 3 ways of perturbing these networks:
1. Every generation, `∀θᵢ, εᵢ ~ N(0, 1e-3), θᵢ += εᵢ` (`1e-3` is picked to match the default learning rates for common `torch` optimizers like `SGD` and `AdamW`) <=> noise sampled from the `same gaussian distribution` is applied across network parameters `θ` .
2. Begin optimization by setting `∀θᵢ, σᵢ = 1e-3`. Every generation, `∀θᵢ, ξᵢ ~ N(0, 1e-2), σᵢ ×= (1 + ξᵢ), εᵢ ~ N(0, σᵢ²), θᵢ += εᵢ` <=> noise sampled from a gaussian distribution with `shifting per-parameter standard deviation σ` is applied across network parameters `θ`. The shifting of `σ` is driven by applying noise sampled from the `same gaussian distribution` (as in the first method).
3. Every generation, `∀θᵢ, εᵢ ~ N(0, 1e-3), θᵢ += AdamW_step(εᵢ)`.

#### Dynamically complexifying networks

The only component of the codebase 
Implemented in `dynamic_net.py`, we do not describe how these networks' architecture comes about here.

However we do describe how to interface with them.

##### Initalization and mutation

```
from dynamic_net import Net

net = Net(num_inputs, num_outputs) # Initialize
net.mutate() # Perturbs the network (to be called once per iteration)
```

##### Components for computation

The three key components to be retrieved for computation are `net.weights`,
`net.biases` and `net.in_nodes_indices`.

`net.weights` is a `float` `torch.Tensor` of shape `num_hidden_and_output_nodes x 3`. It contains all of the network's weights (each neuron has one weight per incoming connection, and at most 3 of them).

`net.biases` is a `float` `torch.Tensor` of shape `num_output_nodes x 3`. It contains all of the network's biases, given that only output nodes have biases.

`net.in_nodes_indices` is an `int` `torch.Tensor` of shape `num_hidden_and_output_nodes x 3`. It contains the indices of incoming connections for each neuron.

```
x = torch.hstack([
  torch.zeros(1), # Ghost node for where `net.in_nodes_indices == 0`
  torch.randn(len(net.nodes.input)), # Inputs
  torch.zeros(len(net.nodes.hidden + net.nodes.output)), # Hidden and output node outputs (empty on the first pass)
])
...
x_reshaped = torch.gather
```
 Only output nodes have biases.
- `net.biases`:
- `net.in_nodes_indices` is a `torch.Tensor` of shape `NHON x 3`
        # A tensor that contains all nodes' in nodes' mutable ids. Used during
        # computation to fetch the correct values from the `outputs` attribute.

### Details

* Given that networks are components of agents, they are `also maintained in population-wide tensors`. During evaluation for instance, input information of format `population_size x (input_dimensions)` is exposed to a weight metrix of format `population_size x (weight_dimensions)` through operations the likes of `torch.bmm()`.

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