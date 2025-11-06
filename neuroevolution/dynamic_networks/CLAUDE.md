- [Specifications](#specifications)
  - [Nodes](#nodes)
  - [Original structure](#original-structure)
  - [Mutations](#mutations)
  - [Representation for computation](#representation-for-computation)
  - [Node computation](#node-computation)
  - [Computation pipeline](#computation-pipeline)
- [Gotchas](#gotchas)

# Specifications

## Nodes

The networks have three types of nodes: `input`, `hidden` and `output` nodes.
`hidden` nodes have biases wheras `input` and `output` nodes do not.

## Original structure

The network begins with one `input` node per value it inputs, one `output` node per value it outputs, and no `hidden` nodes.

A network's original structure differs whether or not it is **mapping** or **remapping**.

If it is **mapping**, there is no starting **edge** between the `input` and `output` nodes.
Since the first variation stage occurs before the first evaluation stage, the network will have an opportunity to grow neural circuits at that point.

If it is **remapping**, there is an equivalent amount of `input` and `output` nodes, and an edge with a weight of 1 is created between each `input` node `i` and `output` node `i`.

## Mutations

The networks contract and expand in structure through two architectural mutations: `grow_node` and `prune_node`.

When the `grow_node` mutation is called, the following occurs:
1) A new `hidden` node is created.
2) A first `in` node is randomly sampled
3) A second `in` node is sampled with higher probability placed on nodes near the first `in` node.
4) An `out` node is sampled with higher probability placed on nodes near the second `in` node.
5) Edges are grown from the `in` nodes to the new `hidden` node and from the new `hidden` node to the `out` node.

When the `prune_node` mutation is called, the following occurs:
1) A `hidden` node is randomly sampled.
2) All of its edges, followed by itself, are pruned.
3) The `prune_node` logic is then applied on any other `hidden` node that no longer **receives** or **emits** information.

## Representation for computation

N/A.

## Node computation

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

# Gotchas

N/A