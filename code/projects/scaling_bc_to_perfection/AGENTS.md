# Scaling Behaviour Cloning to Perfection

## Project Proposal

In this project, we aim to run scaling law analyses — investigations into how key performance metrics evolve as variables like dataset size or model capacity increase — in behaviour cloning tasks.

More specifically, we aim to perform these analyses on metrics that quantify perfection, i.e. that can saturate. This is somewhat in contrast with typical metrics in the scaling law literature — such as perplexity — that generally do not saturate.

We wish to do so in order to closely observe behaviour at the edge of saturation. Indeed, we hypothesize that various properties of gradient-based methods (relationship with data regime, differentiablity contraint, etc) leads to practical limitations in their ability to saturate such metrics, with respect to other computational/machine learning techniques.

We propose to attempt to observe this phenomenon by benchmarking deep learning methods against genetic algorithms. Unlike deep learning, which is restricted to differentiable functions, evolutionary algorithms can optimize over any space of functions for which outputs can be ranked.

However, genetic algorithms' sole reliance on the selection mechanism to propagate data-driven information typically results in less efficient scaling compared to backpropagation. To overcome this limitation, we thus propose to also explore a hybrid approach: leveraging the representational power and information-rich outputs of deep learning models as inputs to evolutionary algorithms. We hypothesize that this integration may enable us to explore beyond the confines of gradient-based optimization, while still benefiting from its efficiency.

We will begin with simple benchmarks, such as OpenAI Gym environments and behaviour targets derived from pre-trained reinforcement learning models (e.g., PPO). Our experimental setup will involve small MLP networks for basic behaviour cloning tasks, such as output classification. We will first analyze the scaling behaviour of deep learning models and neuroevolution (genetic algorithms + neural networks) in isolation, followed by experiments combining deep learning models with evolutionary search, where outputs of trained deep learning models are mapped into evolutionary algorithm-driven MLPs that reconstruct the action space.

In terms of evaluation, our primary metric will be the distributional difference in reward trajectories between target behaviors and their imitated counterparts. On a per-task basis, we also wish to observe the evolution of metrics such as episode length, position, etc. These will provide insight into both the fidelity of imitation and the effect of scaling across different optimization paradigms.

