# /projects/scaling bc to perfection/

- [/projects/scaling bc to perfection/](#projectsscaling-bc-to-perfection)
  - [Project Proposal](#project-proposal)


## Project Proposal

In this project, we aim to run scaling law analyses — investigations into how key performance metrics evolve as variables like dataset size, model capacity and FLOPs increase — in behaviour cloning tasks.

More specifically, we aim to perform these analyses on metrics that quantify perfection, i.e. that can saturate. This is somewhat in contrast with typical metrics in the scaling law literature — such as perplexity — that generally do not saturate.

We wish to do so in order to closely observe behaviour at the edge of saturation. Indeed, we hypothesize that several properties of deep learning (DL) harm its ability to accurately model behaviour. We list out some of these properties and their impact:

1. The differentiability constraint.
DL methods, being gradient-based, can only optimize over differentiable functions - a relatively narrow subspace of computable functions. This constraint demands of practitioners to proxy further from their desired objective, which leads to discrepancies between training success and behavioural fidelity.

2. Data hyperdependency
DL model updates are fully data-driven, leaving no room for exploration of parameters in non-data space.

3. Data hunger
DL methods are well-known to require large amounts of data to perform well. Their generalization relies heavily on distributional coverage, leading to overfitting on frequent patterns and poor handling of rare or unseen ones.

4. Lack of causal abstraction
DL models learn statistical associations rather than causal structures (Li et al., 2024) (at least not directly), limiting their ability to generalize under distributional shifts or to infer intent behind observed behaviour.

5. Overparameterization bias
While overparameterization aids optimization, it appears to encourage memorization and smooth interpolation over true understanding (Djiré et al., 2025), reducing robustness in low-data or out-of-distribution regimes.

6. Representation entanglement
Internal representations in deep models are highly distributed and entangled (Kumar et al., 2025), making them harder to interpret or manipulate, and hindering modular reuse of learned components.

We propose to attempt to observe this failure to saturate by benchmarking DL methods against genetic algorithms (GAs).

GAs, in contrast, can:
- optimize over any space of functions which outputs can be ranked
- incorporate explicit exploration mechanisms (mutation, crossover) that promote discovery beyond data-implied regions
- have more leeway to evolve modular, interpretable structures that exhibit causal abstraction and reuse

However, GAs' sole reliance on the selection mechanism to propagate information derived from data typically results in less efficient scaling compared to DL's backpropagation. To overcome this limitation, we thus propose to also explore a hybrid approach: leveraging the representational power and information-rich outputs of DL models as inputs to GAs. We hypothesize that this integration will enable us to explore beyond the confines of gradient-based optimization, while still benefiting from its efficiency.

---

We will first wish to look for cases where DL methods are less capable of saturating metrics than GAs. In order to do so, we plan to work our way up from simple 1) environments, e.g. classic control tasks in OpenAI Gym 2) behaviour targets, e.g. trained ML policies 3) models, e.g. double-layer MLPs and 4) optimization objectives, e.g. output classification; and work our way up to more complexity if needs be.

Metrics will vary on a per-task basis.

When/If we find such cases, we then plan to experiment with the hybrid method and observe its behaviour towards saturation with respect to both previously explored underlying methods.