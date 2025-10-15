# Neural Networks of Dynamic Complexity

## Overview

Disclaimer
----------

The words `node` and `neuron` are used interchangeably.

Structure
---------

These networks have three types of neurons: input, hidden and output.
Input and output neurons have no biases.

Modes
-----

These networks either:
- transform signal from one space to another input space to and output space, or
- remap signal from some output space back into  

Original Structure
------------------

The network begins with one input node per value it inputs, one output node per value it outputs, and no hidden neurons.



A network's original structure differs whether or not it is remapping some signal.

In all cases, it begins with no hidden neurons.

If no deep learning, there is no original connection between the input and output nodes.
Since the first variation stage occurs before the first evaluation stage, the network will have an opportunity to grow neural circuits at that point.

If not, the network starts with 

Mutations
---------

These networks are dynamic in structure: they contract and expand through two mutation types: `grow_node` and `prune_node`.

Representation for computation
------------------------------



Computation
-----------

A node N performs a simple `tanh(wx+b)` with `x` being a vector concatenating outputs from the nodes that connect to node N (which includes node N itself).

In practice, all nodes in the network

Input, hidden, output neurons



if mapping from environment input space to output space:
start with no hidden neuron, and no connection between neurons.

if mapping from deep learning model output space back into output space:
start with no hidden neuron, and 

## Computation pipeline

(All computations are batched over the entire population)

1. Pre-network processing

This is where operations like standardization happen over the environment input vectors (or deep learning model output vectors).

2. Network processing

The results of Step 1 are multiplied against the sparse matrix representations of the networks' weights. The networks' biases are added to the results.

3. Post-network processing

This is where operations like producing a discrete action with `argmax` happen.