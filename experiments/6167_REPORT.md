- [Project Proposal](#project-proposal)
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
      - [Plots](#plots)
      - [Analysis](#analysis)
      - [Issues](#issues)
    - [Experiment 2: Comparing methods more fairly](#experiment-2-comparing-methods-more-fairly)
      - [Data](#data-1)
      - [Network setup](#network-setup-1)
      - [Optimization/Evaluation functions](#optimizationevaluation-functions)
      - [Neuroevolution algorithms](#neuroevolution-algorithms)
      - [Optimization](#optimization)
      - [Plots](#plots-1)
      - [Results](#results)
  - [2. Modeling continual human behaviour](#2-modeling-continual-human-behaviour)
      - [Experiment 3: On the Value of Giving Continual Learning information](#experiment-3-on-the-value-of-giving-continual-learning-information)
        - [Data](#data-2)
        - [Network Setup](#network-setup-2)
        - [Methods](#methods-1)
        - [Evaluation Metrics](#evaluation-metrics-1)
        - [Plots](#plots-2)
        - [Training Progress Tracking](#training-progress-tracking)
        - [Analysis](#analysis-1)
- [Citations](#citations)

# Project Proposal

In this project, we aim to run scaling law analyses — investigations into how key performance metrics evolve as variables like dataset size, model capacity and FLOPs increase — in human behaviour imitation tasks.

More specifically, we aim to perform these analyses on metrics that quantify perfection, i.e. that can saturate. This is somewhat in contrast with typical metrics in the scaling law literature — such as perplexity — that generally do not saturate.

We wish to do so in order to closely observe behaviour at the edge of saturation. Indeed, we hypothesize that several properties of deep learning (DL) harm its ability to accurately model the complexity of human behaviour. We list out some of these properties and their impact:

1. The differentiability constraint.
DL methods, being gradient-based, can only optimize over differentiable functions - a relatively narrow subspace of computable functions. This constraint demands of practitioners to proxy further from their desired objective, which leads to discrepancies between training success and behavioural fidelity.

2. Data hyperdependency
DL model updates are fully data-driven, leaving no room for exploration of parameters in non-data space.

3. Data hunger
DL methods are well-known to require large amounts of data to perform well. Their generalization relies heavily on distributional coverage, leading to overfitting on frequent patterns and poor handling of rare or unseen ones.

4. Lack of causal abstraction
DL models learn statistical associations rather than (at least not directly) causal structures (Li et al., 2024), limiting their ability to generalize under distributional shifts or to infer intent behind observed behaviour.

5. Overparameterization bias
While overparameterization aids optimization, it appears to encourage memorization and smooth interpolation over true understanding (Djiré et al., 2025), reducing robustness in low-data or out-of-distribution regimes.

6. Representation entanglement
Internal representations in deep models are highly distributed and entangled (Kumar et al., 2025), making them harder to interpret or manipulate, and hindering modular reuse of learned components.

We propose to attempt to observe this expected failure to saturate by benchmarking DL methods against genetic algorithms (GAs).

GAs, in contrast, can:
- optimize over any space of functions which outputs can be ranked
- incorporate explicit exploration mechanisms (mutation, crossover) that promote discovery beyond data-implied regions
- have more leeway to evolve modular, interpretable structures that exhibit causal abstraction and reuse

However, GAs' sole reliance on the selection mechanism to propagate information derived from data typically results in less efficient scaling compared to DL's backpropagation. To overcome this limitation, we thus propose to also explore a hybrid approach: leveraging the representational power and information-rich outputs of DL models as inputs to GAs. We hypothesize that this integration will enable us to explore beyond the confines of gradient-based optimization, while still benefiting from its efficiency.

---

We will first wish to look for cases where DL methods are less capable of saturating metrics than GAs. In order to do so, we plan to work our way up from simple 1) environments, e.g. classic control tasks in OpenAI Gym 2) models, e.g. double-layer MLPs and 3) optimization objectives, e.g. output classification; and work our way up to more complexity if needs be.

Metrics will vary on a per-task basis.

When/If we find saturation discrepancies, we plan to then experiment with the hybrid method and observe its behaviour towards saturation relative to both underlying methods.


# Overview

Abstract
Introduction and Motivation
Problem formulatuon
Methods
Experiments
Conclusion

## Motivation

Deep Learning methods have yielded close to the entirety of modern progress in AI. However, we hypothesize that several of its properties (e.g., differentiability constraint, data hyperdependency, data hunger, lack of causal abstraction, overparameterization bias & representation entanglement) make it, standalone, a sub-optimal choice for various tasks. All computational methods have strengths and weaknesses and we believe there is unexplored value to be found in mixing them.

For instance, we believe Deep Learning-only methods to be sub-optimal for human behaviour cloning (due to its computational complexity), which we focus on in this project.

## Neuroevolution

Neuroevolution is a distant cousin of Deep Learning. They both optimize artifical neural networks, however, the former makes use of evolutionary algorithms to optimize the networks.

Neuroevolution methods have several advantageous properties over Deep Learning (e.g. can optimize any functions whose outputs can be ranked, explicit exploration mechanisms that promote discovery beyond data-implied regions, more leeway to evolve modular structures that exhibit causal abstraction and reuse). However, their sole reliance on the selection mechanism to propagate information derived from data typically results in less efficient scaling compared to DL's backpropagation.

# Comparing Deep Learning and Neuroevolution in low-dimensional tasks

We propose to first try to find particular settings where Neuroevolution has direct practical benefits over Deep Learning.

Given that Neuroevolution is known not to scale well to information-rich domains, we propose to restrict ourselves to the following low-dimensional tasks: `Acrobot`, `CartPole`, `MountainCar` and `LunarLander`.

## 1. Modeling state-action pairs

We propose to begin our experiments with the task of `modeling state-action pair` datasets created from pre-trained Reinforcement Learning policies available on HuggingFace. The purpose here is to simply get a first feel of how the methods behave.

### Experiment 1: Scaling Law Analysis - Dataset Size vs Performance

`@experiments/1_dl_vs_ga_scaling_dataset_size_flops/main.py`

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

#### Plots

Four-panel comparison showing:
1. **Top-left**: Action Match Rate vs Dataset Size - DL rapidly approaches perfect performance while GA plateaus
2. **Top-right**: Episode Return vs Dataset Size - DL achieves maximum return (500) while GA remains inconsistent
3. **Bottom-left**: Action Match Rate vs FLOPs - DL achieves better performance with fewer compute resources
4. **Bottom-right**: Episode Return vs FLOPs - Compute efficiency advantage of DL is clear

![Scaling Law Analysis Results](1_dl_vs_ga_scaling_dataset_size_flops/main.png)

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

#### Issues

This experiment was vibe-coded with very little oversight and thus has several flaws. The biggest is that GAs, every generation, calculate over the whole training set in one-go, no mini-batches. This means that GAs had 100 update steps while DL had 150 x dataset_size / 64 updates.

Rather than attempt to address each of these issues, we propose to learn from it heading into experiment 2.

### Experiment 2: Comparing methods more fairly

`@experiments/2_dl_vs_ga_es/main.py`

Heading into experiment 2, we propose to more fairly calibrate both class of methods. We make the following changes:
- We make EAs optimize over mini-batches of the same size as DL methods.
- We propose to compare results based on total runtime (EA generations are much faster than DL optimization steps)
- We GPU-proof the EAs: In experiment 1, agents in the population get evaluated on the GPU one by one in a loop. This time around we run batch matrix multiplications to allow the whole evaluation to run on the GPU.
- We strip both EAs and GAs to their simplest forms (explained below).

#### Data

We use two datasets derived from PPO policies:
- `https://huggingface.co/datasets/NathanGavenski/CartPole-v1`
- `https://huggingface.co/datasets/NathanGavenski/LunarLander-v2`

* Input space: 4 observations for `CartPole`, 8 for `LunarLander`
* Output space: Both datasets' action space is discrete. `CartPole`'s possible actions are 0 and 1, `LunarLander`'s are 0,1,2,3
* Size: 500k for `CartPole`, 384k for `LunarLander`
* Processing: We shuffle the dataset, and train on 90%,30% and 10% of it and test on the remaining 10%. We use batches of size 32.

#### Network setup

We set all methods to now optimize over a MLP with dimensions [input, 50, output] with `tanh` activations on the hidden layer. Output selection remains the same as experiment 1.

#### Optimization/Evaluation functions

We optimize, once again, for both Neuroevolution and Deep Learning, over the `cross entropy`. In order to quantify behaviour cloning perfection, comparing both methods' `macro F1 score` appears more pertinent. We thus use the `macro F1 score` as an evaluation metric.

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

#### Results

Optimizing F1 didn't quite work, might be both too uniformative and noisy (only 10 trials)
N/A

## 2. Modeling continual human behaviour

Human behaviour quite complex. Compared to the behaviour we have been trying to imitate thus far, it is not a list of frozen weights we can load and rollout hundreds of thousands of time.

In the second section of this project, we propose to work on modeling the continual behaviour of human subjects.

We asked 2 subjects to perform 4 different virtual tasks/games over the course of several days and collected their data through `@data/collect.py`.

The tasks were:
- `Acrobot`: Swing a fixed two-link chain left/right to reach a certain height.
- `CartPole`: Move a cart left/right to prevent a pole attached to it from falling over.
- `MountainCar`: Roll a car left/right up/down a hill to reach a flag on the right hill.
- `LunarLander`: Rocket boost down/left/right to land a lunar lander within a flagged region.

We gave no instruction nor information about what the games are about to the participants. We asked the participants simply to play, whether or not they wanted to win or optimize their score was up to them.

In the data, we collect, for every episode the exact start time of the episode.

We create a simple categorization of the data:
If an episode starts at least 30 minutes after the previous episode, a new `session` has begun.
Within one `session` can be multiple episodes which we then call `runs` (e.g. `episode` 48 could be `session` 10 `run` 3).

We give that information to the networks by adding two input values (one for `session`, one for `run`).
We set the remap the `session` values from `1,2,3,...,n` to `-1,...,1`, with equal distance value between `sessions`.
For each `session` we set remap its `run` values in the same manner.
Ex: 5 sessions, session 1 has 1 runs, session 2 has 3 runs, session 3 has 5 runs -> session values = [-1.0, -0.5, 0.0, 0.5, 1.0], session 1 run values = [0.0], session 2 run values = [-1.0, 0.0, 1.0], session 1 run values = [-1.0, -0.5, 0.0, 0.5, 1.0].

#### Experiment 3: On the Value of Giving Continual Learning information

`@experiments/3_cl_info_dl_vs_ga/main.py`

We make the following changes from experiment 2:
- We drop `ES` given its underperformance relative to `GA`, only keep `adaptive GA CE & SGD`.
- We run ablation experiment: with/without `session` and `run` information.
- No scaling over dataset size: train on first 90% of data, test on last 10%

##### Data

We use human behavioral data collected from our two subjects playing 4 environments:
- **Environments**: CartPole (4 obs, 2 actions), MountainCar (2 obs, 3 actions), Acrobot (6 obs, 3 actions), LunarLander (8 obs, 4 actions)
- **Data structure**: Each episode includes timestamps, allowing us to derive session and run information
  - **Session**: Starts when >30 minutes have passed since previous episode
  - **Run**: Episode number within a session
- **Continual Learning features**: Session and run values are normalized to [-1, 1] range within their respective groups
  - Example: 5 sessions → session values = [-1.0, -0.5, 0.0, 0.5, 1.0]
  - Example: 3 runs in a session → run values = [-1.0, 0.0, 1.0]
- **Split**: Sample, within each session, 90% runs for optimization, 10% runs for testing

##### Network Setup

Same except input layer: `obs_dim` (without CL) OR `obs_dim + 2` (with CL features: session, run)

##### Methods

SGD & Adaptive Simple GA from above

##### Evaluation Metrics

Models are evaluated on their ability to replicate human behavior in the actual environments:

**1. Matched Episode Evaluation**
- Run the same number of episodes as the human participant
- Use identical random seeds for environment initialization to ensure fair comparison
- For models with CL features: use median session/run values from test set

**2. Key Metrics**
- **Mean return**: Average cumulative reward across episodes
- **Percentage difference from human**: `(model_return - human_return) / |human_return| × 100`
  - Closer to 0% = better match to human behavior
  - Positive = model performs better than human
  - Negative = model performs worse than human
- **Statistical significance**: Mann-Whitney U test comparing model vs human episode returns
  - p < 0.001 (***), p < 0.005 (**), otherwise not marked

**3. Test Set Performance**
- **Macro F1 score**: Evaluation metric for action prediction accuracy on held-out test data
- **Cross-entropy loss**: Training objective, also measured on test set

##### Plots

Three main visualizations generated for each environment × participant combination:

**1. Aggregate Returns Comparison**
- Bar chart showing mean percentage difference from human for all 4 methods
- Sorted by absolute percentage difference (best match first)
- Error bars show standard deviation across episodes
- Color coding: Blue for `_with_cl`, Red for `_no_cl`
- Statistical significance markers (**, ***) shown above bars
- Lower absolute values = better human behavior replication

**2. CL Impact Comparison**
- Side-by-side bar chart for each base method (SGD vs adaptive_ga_CE)
- Direct comparison of `_no_cl` vs `_with_cl` variants
- Shows percentage difference from human (0% = perfect match)
- Green dashed line at 0% indicates perfect match baseline
- Arrows indicate direction of change from no_cl to with_cl
- ✓/✗ symbols indicate whether adding CL info brought model closer to 0% (improvement)
- Shows numerical improvement in percentage points (pp)

**3. Return Distributions**
- Violin plots showing distribution of episode returns
- Compares human distribution vs each model's distribution
- Allows visual assessment of whether models capture variance in human behavior
- Shows both central tendency and spread of performance

##### Training Progress Tracking

Both methods track and save:
- Training loss history (cross-entropy)
- Test loss history
- Macro F1 score history (evaluated periodically)
- Model checkpoints (support resuming from previous runs)

##### Analysis

*[To be filled after running experiments across all environments and participants]*

Expected insights:
- Does providing continual learning information (session/run) improve model fit to human behavior?
- Does the impact of CL information differ between Deep Learning and Neuroevolution?
- Which environments show the strongest effect of CL information?
- Do models trained with CL info better capture temporal dynamics of human learning?

# Citations

Li et al., 2024: A Survey of Deep Causal Models and Their Industrial Applications
Djiré et al., 2025: Memorization or Interpolation ? Detecting LLM Memorization through Input Perturbation Analysis
Kumar et al., 2025: Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis
