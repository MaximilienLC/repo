- [Overview](#overview)
  - [Motivation](#motivation)
  - [Neuroevolution](#neuroevolution)
- [Comparing Deep Learning and Neuroevolution in low-dimensional tasks](#comparing-deep-learning-and-neuroevolution-in-low-dimensional-tasks)
  - [1. Modeling state-action pairs](#1-modeling-state-action-pairs)
    - [Experiment 1: Scaling Law Analysis - Dataset Size vs Performance](#experiment-1-scaling-law-analysis---dataset-size-vs-performance)
      - [Data](#data)
      - [Network setup](#network-setup)
      - [Methods](#methods)
      - [Evaluation Metrics](#evaluation-metrics)
      - [Results](#results)
      - [Analysis](#analysis)
      - [Plots](#plots)
      - [Issues](#issues)
    - [Experiment 2: Comparing methods more fairly](#experiment-2-comparing-methods-more-fairly)
      - [Data](#data-1)
      - [Network setup](#network-setup-1)
      - [Optimization/Evaluation functions](#optimizationevaluation-functions)
      - [Neuroevolution algorithms](#neuroevolution-algorithms)
      - [Optimization](#optimization)
      - [Plots](#plots-1)
      - [Hypothesis \& Results](#hypothesis--results)
        - [Performance gap between SGD \& NE Methods](#performance-gap-between-sgd--ne-methods)
        - [Remaining differences in compute use](#remaining-differences-in-compute-use)
  - [2. Modeling continual behaviour](#2-modeling-continual-behaviour)
      - [](#)

# Overview

## Motivation

Deep Learning methods have pretty much yielded the entirety modern progress in AI. However, we hypothesize that several of its properties (e.g., differentiability constraint, data hyperdependency, data hunger, lack of causal abstraction, overparameterization bias & representation entanglement) make it, standalone, a sub-optimal choice for various tasks. All computational methods have strengths and weaknesses and we believe there is unexplored value to be found in mixing them.

For instance, we believe Deep Learning-only methods to be sub-optimal for human behaviour cloning, which we focus on in this project.

## Neuroevolution

Neuroevolution is a distant cousin of Deep Learning. They both optimize artifical neural networks, however, the former makes use of evolutionary algorithms to optimize the networks.

Neuroevolution methods have several advantageous properties over Deep Learning (e.g. can optimize any functions whose outputs can be ranked, explicit exploration mechanisms that promote discovery beyond data-implied regions, more leeway to evolve modular structures that exhibit causal abstraction and reuse). However, their sole reliance on the selection mechanism to propagate information derived from data typically results in less efficient scaling compared to DL's backpropagation.

# Comparing Deep Learning and Neuroevolution in low-dimensional tasks

We propose to first try to find particular settings where Neuroevolution has direct practical benefits over Deep Learning.

Given that Neuroevolution is known not to scale well to information-rich domains, we propose to restrict ourselves to the following low-dimensional tasks: `Acrobot`, `CartPole`, `MountainCar` and `LunarLander`.

## 1. Modeling state-action pairs

We propose to begin our experiments with the task of `modeling state-action pair` datasets created from pre-trained Reinforcement Learning policies available on HuggingFace. The purpose here is to simply get a first feel of how the methods behave.

### Experiment 1: Scaling Law Analysis - Dataset Size vs Performance

#### Data

We use the CartPole-v1 dataset from HuggingFace:
- Dataset: `https://huggingface.co/datasets/NathanGavenski/CartPole-v1`
- Input space: 4 observations (cart position, cart velocity, pole angle, pole angular velocity)
- Output space: 2 discrete actions (0: push left, 1: push right)
- Total size: 500,000 state-action pairs
- Split: 495k train, 5k test
- Dataset sizes tested: [100, 300, 1000, 3000, 10000, 30000] samples from training set

#### Network setup

Both methods optimize a 2-layer MLP:
- Architecture: Input (4) → Hidden (64) → Hidden (32) → Output (2)
- Activations: ReLU
- Output: Raw logits converted to probability distribution via softmax

#### Methods

**Deep Learning:**
- Optimizer: Adam with learning rate 1e-3
- Loss function: Cross-entropy
- Training: 150 epochs, batch size 64

**Genetic Algorithm:**
- Population size: 100 individuals
- Generations: 100
- Selection: Tournament selection (k=3) with elitism (best individual preserved)
- Mutation: Gaussian noise with mutation rate 0.01
- Fitness function: Action match accuracy on whole training set

#### Evaluation Metrics

1. **Action Match Rate**: Percentage of test set actions that exactly match expert actions
2. **Episode Return**: Average cumulative reward over 50 evaluation episodes in CartPole-v1 environment (max possible: 500)
3. **FLOPs**: Total floating-point operations (forward + backward passes for DL; forward passes for GA fitness evaluations)

#### Results

**Deep Learning Saturation:**
- 100 samples: 82.08% match, 67.60 return, 2.13e+08 FLOPs
- 300 samples: 82.92% match, 288.62 return, 6.39e+08 FLOPs
- 1000 samples: 95.18% match, 500.00 return, 2.13e+09 FLOPs ← Near saturation
- 3000 samples: 98.50% match, 500.00 return, 6.39e+09 FLOPs ← Strong saturation
- 10000 samples: 99.28% match, 500.00 return, 2.13e+10 FLOPs ← Nearly perfect
- 30000 samples: 99.14% match, 500.00 return, 6.39e+10 FLOPs

**Genetic Algorithm Performance:**
- 100 samples: 82.16% match, 103.88 return, 4.78e+09 FLOPs
- 300 samples: 78.92% match, 101.82 return, 1.44e+10 FLOPs
- 1000 samples: 82.10% match, 46.04 return, 4.78e+10 FLOPs
- 3000 samples: 83.92% match, 44.64 return, 1.44e+11 FLOPs
- 10000 samples: 82.98% match, 293.14 return, 4.78e+11 FLOPs
- 30000 samples: 84.36% match, 190.08 return, 1.44e+12 FLOPs ← Plateaued at ~84%

#### Analysis

**1. DL Saturates, GA Does Not**
- Deep Learning achieves near-perfect performance (99%+ action match, 500 return) at 10k-30k samples
- Genetic Algorithm plateaus at ~84% action match and fails to reach expert-level performance
- DL shows clear power-law scaling behavior with predictable performance improvements as data increases
- GA exhibits erratic behavior, sometimes decreasing in performance with more data

**2. Compute Efficiency Disparity**
- At 30k samples: DL uses 6.39e+10 FLOPs vs GA uses 1.44e+12 FLOPs (22x difference)
- Despite using 22x more compute, GA achieves significantly worse performance
- DL is both more compute-efficient and more capable of reaching saturation

**3. Why GA Fails to Saturate**
- **Fitness-only feedback is too coarse**: GA receives only scalar fitness (match rate) per evaluation with no gradient information to guide parameter adjustments
- **Random mutations insufficient near optima**: Must rely on random mutations to improve, which becomes highly inefficient for fine-tuning
- **Population-based search inefficient for smooth landscapes**: CartPole behavior cloning has a relatively smooth loss landscape where gradient descent excels
- **Fixed mutation rate suboptimal**: Constant mutation rate (0.01) may be too large for fine-tuning near the optimum

**4. Hypothesis Reversal**
This experiment contradicts our initial hypothesis that "DL methods are less capable of saturating metrics than GAs in behavior cloning." Instead, we find:
- The differentiability constraint of DL is **beneficial** for this task
- Gradient information provides richer signal than fitness-based selection
- Backpropagation efficiently propagates information throughout the network
- GA's advantages (exploration, function space flexibility) don't materialize when the loss landscape is smooth and a simple MLP architecture suffices

#### Plots

Four-panel comparison showing:
1. **Top-left**: Action Match Rate vs Dataset Size - DL rapidly approaches perfect performance while GA plateaus
2. **Top-right**: Episode Return vs Dataset Size - DL achieves maximum return (500) while GA remains inconsistent
3. **Bottom-left**: Action Match Rate vs FLOPs - DL achieves better performance with fewer compute resources
4. **Bottom-right**: Episode Return vs FLOPs - Compute efficiency advantage of DL is clear

![Scaling Law Analysis Results](1_dl_vs_ga_scaling_dataset_size_flops/main.png)

#### Issues

This experiment was vibe-coded and thus has several flaws. The biggest is that GAs, every generation calculate over the whole training set in one-go, no mini-batches. This means that GAs had 100 update steps while DL had 150 x dataset_size / 64 updates.

Rather than attempt to address each of these issues, we propose to learn from it heading into experiment 2.

### Experiment 2: Comparing methods more fairly

Heading into experiment 2, we propose to more fairly calibrate both class of methods. We make the following changes:
- We make EAs optimize over mini-batches of the same size as DL methods.
- We propose to compare results based on total runtime (EA generations are much faster than DL optimization steps)
- We GPU-proof the EAs: In experiment 1, agents in the population get evaluated on the GPU one by one in a loop. This time around we run batch matrix multiplications to allow the whole evaluation to run on the GPU.
- We strip both EAs and GAs to their simplest (explained below).

#### Data

We use two datasets derived from PPO policies:
- `https://huggingface.co/datasets/NathanGavenski/CartPole-v1`
- `https://huggingface.co/datasets/NathanGavenski/LunarLander-v2`

* Input space: 4 observations for `CartPole`, 8 for `LunarLander`
* Output space: Both datasets' action space is discrete. `CartPole`'s possible actions are 0 and 1, `LunarLander`'s are 0,1,2,3
* Size: 500k for `CartPole`, 384k for `LunarLander`
* Processing: We shuffle the dataset, and start by training on 90% of it and test on the remaining 10%. We use batches of size 32.

#### Network setup

We begin with all methods optimizing over a double layer-MLP with hidden layer of size 50 and `tanh` activations. In the output layer, we set one neuron per possible action. Raw logits are converted into a probability distribution with a standard softmax.

#### Optimization/Evaluation functions

We optimize, for both Neuroevolution and Deep Learning, over the `cross entropy`. In order to quantify behaviour cloning perfection, comparing both methods' `macro F1 score` appears more pertinent. We thus use the `macro F1 score` as an evaluation metric.

Given that Neuroevolution can use the `macro F1 score` as a fitness score, we propose to (separately) also have it optimize over it.

In order to increase our fidelity when computing the `macro F1 score` (for both Neuroevolution optimization and evaluation), we run 10 sampling trials from the distribution generated by the networks.

#### Neuroevolution algorithms

We characterize every iteration of these algorithms by three stages:
- `variation`: agents in the population are `randomly perturbed`
- `evaluation`: agents `perform a given task` and are `assigned a fitness score`
- `selection`: given the `fitness scores`, certain low performing agents are replaced by duplicates of high performing agents

We implement the two following algorithms:
- `simple_ga`: during selection, agents with the `top 50% fitness scores` are `selected and duplicated`, taking over the population slots of the lower 50% scoring agents
- `simple_es`: during selection, the population's fitness scores are standardized and turned into a probability distribution through a softmax operation. A single agent is then created through a weighted sum of existing agents' parameters. It is then duplicated over the entire population size

We set the population size to 50 arbitrarily.

#### Optimization

We pick SGD with the default `torch` learning rate (`1e-3`) for Deep Learning.

We experiment with two optimization modes for neuroevolution methods:
1. Every generation, `∀θᵢ, εᵢ ~ N(0, 1e-3), θᵢ += εᵢ` <=> noise sampled from the `same gaussian distribution` is applied across network parameters `θ`
2. Begin optimization by setting `∀θᵢ, σᵢ = 1e-3`. Every generation, `∀θᵢ, ξᵢ ~ N(0, 1e-2), σᵢ ×= (1 + ξᵢ), εᵢ ~ N(0, σᵢ²), θᵢ += εᵢ` <=> noise sampled from a gaussian distribution with `shifting per-parameter standard deviation σ` is applied across network parameters `θ`. The shifting of `σ` is driven by applying noise sampled from the `same gaussian distribution` (as in the first method)

We believe the second mode to more closely resemble SGD in that weight update distributions have the opportunity of being weight-specific.

#### Plots

We generate the following plots for each dataset:

1. **CE Loss**: Line plot showing test cross-entropy loss vs. Runtime % for methods optimizing cross-entropy:
   - SGD (DL)
   - simple_ga + fixed_sigma + CE
   - simple_ga + adaptive_sigma + CE
   - simple_es + fixed_sigma + CE
   - simple_es + adaptive_sigma + CE

   Note: F1-optimizing NE methods are excluded from this plot as they optimize a different objective.

2. **Macro F1 Score**: Line plot showing macro F1 score on the test set vs. Runtime % for all 9 methods. This is the primary evaluation metric. The Runtime % x-axis (0-100%) allows direct comparison of methods regardless of their total number of iterations/generations.

3. **Final Macro F1 Score**: Bar chart comparing final macro F1 scores across all methods, sorted from best to worst performance, with numerical values displayed above each bar.

![Plot 2](2_dl_vs_ga_es/cartpole_v1.png)
![Plot 2](2_dl_vs_ga_es/lunarlander_v2.png)

#### Hypothesis & Results

##### Performance gap between SGD & NE Methods

All NE methods underperform
Regardless of the NE method used,

##### Remaining differences in compute use

- pop size
- resource utilization

## 2. Modeling continual behaviour

Human behaviour is not fixed. Not a list of frozen weights we can load and rollout hundreds of thousands of time

In the second section of this project, we propose to work on modeling the continual behaviour of human subjects.

We asked 3 subjects to perform 4 different virtual tasks/games over the course of several days and collected their data through a script we developed.

The tasks were:
- `Acrobot`: Swing a fixed two-link chain left/right to reach a
- `CartPole`: Move a cart left/right to prevent a pole from falling
- `MountainCar`: Roll a car left/right up/down a hill to reach a flag on the right
- `LunarLander`: Rocket boost down/left/right to land a lunar lander within a flagged region

We gave no instruction nor information about what the games are about to the participants. We asked the participants simply to play, whether or not they wanted to win or optimize their score was up to them.

In the data, we collect, for every episode the exact start time of the episode.

####

Our modeling task is: for each

We take episode 1, set its value to -1 take the last episode, set its value to 1 and everything else between -1 and 1.
