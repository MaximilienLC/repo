# Scaling Behaviour Cloning to Perfection

## Project Proposal

In this project, we aim to run scaling law analyses — investigations into how key performance metrics evolve as variables like dataset size or model capacity increase — in behaviour cloning tasks.

More specifically, we aim to perform these analyses on metrics that quantify perfection, i.e. that can saturate. This is somewhat in contrast with typical metrics in the scaling law literature — such as perplexity — that generally do not saturate.

We wish to do so in order to closely observe behaviour at the edge of saturation. Indeed, we hypothesize that various properties of gradient-based methods (relationship with data regime, differentiability constraint, etc) leads to practical limitations in their ability to saturate such metrics in various cases, with respect to other computational/machine learning techniques.

We propose to attempt to observe this phenomenon by benchmarking deep learning methods against genetic algorithms. Unlike deep learning, which is restricted to differentiable functions, genetic algorithms can optimize over any space of functions for which outputs can be ranked.

However, genetic algorithms' sole reliance on the selection mechanism to propagate data-driven information typically results in less efficient scaling compared to backpropagation. To overcome this limitation, we thus propose to also explore a hybrid approach: leveraging the representational power and information-rich outputs of deep learning models as inputs to genetic algorithms. We hypothesize that this integration will enable us to explore beyond the confines of gradient-based optimization, while still benefiting from its efficiency.

---

We will first wish to look for cases where gradient-based methods are less capable of saturating metrics than genetic algorithms. In order to do so, we plan to work our way up from simple 1) environments: classic control tasks in OpenAI Gym or simpler (e.g. grid-based environments) 2) behaviour targets: random or almost random behaviour 3) models: double-layer MLPs and 4) optimization objectives: output classification; and work our way up to more complexity.

Metrics will vary on a per-task basis.

When/If we find such cases, we then plan to experiment with the hybrid method and observe its behaviour towards saturation with respect to both previously explored underlying methods.
