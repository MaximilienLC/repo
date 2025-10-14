# Neural Networks of Dynamic Complexity


Input, hidden, output neurons

input neurons and output neurons have no bias

if mapping from environment input space to output space:
start with no hidden neuron, and no connection between neurons.

if mapping from deep learning model output space back into output space:
start with no hidden neuron, and 

Computation pipeline:

(All computations are batched over the entire population)

1. Pre-network processing

This is where operations like standardization happen over the environment input vectors (or deep learning model output vectors).

2. Network processing

The results of Step 1 are multiplied against the sparse matrix representations of the networks' weights. The networks' biases are added to the results.

3. Post-network processing

This is where operations like producing a discrete action with `argmax` happen.