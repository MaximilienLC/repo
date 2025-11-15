"""Contains example logic to drive the computation of a population of
networks.

Acronyms:
`TNMN` : Total (all networks) number of mutable (hidden and output) nodes.
`NON` : Number of output nodes.
 `TNN` : Total number of nodes.
"""

import random

import torch
from beartype import BeartypeConf
from beartype.claw import beartype_this_package
from jaxtyping import Bool, Float, Int  # Assuming jaxtyping is installed
from torch import Tensor

from .evolution import Net

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))


class WelfordRunningStandardizer:
    def __init__(
        self: "WelfordRunningStandardizer",
        n_mean_m2_x: Float[Tensor, "TNNplus1 4"],
        verbose: bool = False,
    ):
        self.n_mean_m2_x: Float[Tensor, "TNNplus1 4"] = n_mean_m2_x
        self.verbose = verbose
        if verbose:
            print("a. n_mean_m2_x")
            print(n_mean_m2_x)

    def __call__(
        self: "WelfordRunningStandardizer",
        x: Float[Tensor, "TNNplus1"],
    ) -> Float[Tensor, "TNNplus1"]:
        """
        Processes an input tensor 'x' containing a mix of old z-scores and
        new raw values.

        - If x[i] == prev_x[i] (old z-score) or x[i] == 0, stats are not updated.
        - If x[i] != prev_x[i] (new raw value), stats are updated using x[i].

        Returns a tensor where new raw values have been standardized and
        old z-scores remain the same.
        """

        # 1. Get previous state
        prev_n: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x[:, 0]
        prev_mean: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x[:, 1]
        prev_m2: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x[:, 2]
        # 'prev_x' holds the *previous z-score output*
        prev_x: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x[:, 3]

        # 2. Define the update mask. Update only for "new raw values",
        #    which are non-zero and not equal to the last z-score output.

        mask: Bool[Tensor, "TNNplus1"] = (x != 0) & (x != prev_x)
        if self.verbose:
            print("b. x")
            print(x)
            print("c. prev_x")
            print(prev_x)
            print("d. mask")
            print(mask)
        # 3. Calculate potential new values for the stats (based on raw x)
        n_potential = prev_n + 1.0
        delta = x - prev_mean
        mean_potential = prev_mean + delta / n_potential
        delta_potential = x - mean_potential
        M2_potential = prev_m2 + delta * delta_potential

        # 4. Conditionally update the stats.
        #    'n', 'mean', 'm2' now hold the *current* stats for this call.
        n = self.n_mean_m2_x[:, 0] = torch.where(mask, n_potential, prev_n)
        mean = self.n_mean_m2_x[:, 1] = torch.where(mask, mean_potential, prev_mean)
        m2 = self.n_mean_m2_x[:, 2] = torch.where(mask, M2_potential, prev_m2)

        # 5. Calculate z-score using the *updated* stats from step 4
        variance: Float[Tensor, "TNNplus1"] = m2 / n
        std_dev: Float[Tensor, "TNNplus1"] = torch.sqrt(variance)

        is_valid: Bool[Tensor, "TNNplus1"] = n >= 2
        safe_std_dev: Float[Tensor, "TNNplus1"] = torch.clamp(std_dev, min=1e-8)

        raw_z_score: Float[Tensor, "TNNplus1"] = (x - mean) / safe_std_dev

        # This is the z-score output for *all* elements (new and old)
        z_score_output: Float[Tensor, "TNNplus1"] = torch.where(
            is_valid, raw_z_score, torch.tensor(0.0)
        )

        # 6. Determine the final output
        #    - If mask is True (new raw value), use the new z_score_output.
        #    - If mask is False (old z-score or 0), use the original input x.
        x: Float[Tensor, "TNNplus1"] = torch.where(mask, z_score_output, x)

        # 7. Store the final output as the new 'prev_x' for the next call
        self.n_mean_m2_x[:, 3] = x
        if self.verbose:
            print("e. n_mean_m2_x")
            print(self.n_mean_m2_x)

        # 8. Return the final, processed tensor
        return x.clone()


def barebone_run(verbose: bool = True):
    """Simple working example to demonstrate how to run computation for the
    dynamic network."""
    torch.manual_seed(0)
    random.seed(0)

    POPULATION_SIZE = 4
    NUM_INPUTS = 3
    NUM_OUTPUTS = 2

    nets: list[Net] = [Net(NUM_INPUTS, NUM_OUTPUTS) for _ in range(POPULATION_SIZE)]

    for net in nets:
        for _ in range(5):  # grow the networks a bit
            net.mutate()
    nets[1].num_network_passes_per_input = 3
    nets[3].num_network_passes_per_input = 2
    if verbose:
        for i in range(POPULATION_SIZE):
            print(f"Net {i}")
            for node in nets[i].nodes.all:
                print(node)

    nets_num_nodes: Int[Tensor, "POPULATION_SIZE"] = torch.tensor(
        [len(net.nodes.all) for net in nets]
    )
    if verbose:
        print("1. nets_num_nodes")
        print(nets_num_nodes)

    # We add a value at the front to aid computation. This value at index 0
    # will always output 0. Empty in-node slots map to 0, meaning that node.
    n_mean_m2_x: Float[Tensor, "TNNplus1 4"] = torch.cat(
        ([torch.zeros(1, 4)] + [net.n_mean_m2_x for net in nets])
    )
    wrs = WelfordRunningStandardizer(n_mean_m2_x.clone(), verbose=True)
    if verbose:
        print("3. n_mean_m2_x")
        print(n_mean_m2_x)
        print(n_mean_m2_x.shape)

    input_nodes_start_indices: Int[Tensor, "POPULATION_SIZE"] = (
        torch.cat((torch.tensor([0]), torch.cumsum(nets_num_nodes[:-1], dim=0))) + 1
    )
    if verbose:
        print("4. input_nodes_start_indices")
        print(input_nodes_start_indices)

    input_nodes_indices: Int[Tensor, "NUM_INPUTSxPOPULATION_SIZE"] = (
        input_nodes_start_indices.unsqueeze(1) + torch.arange(NUM_INPUTS)
    ).flatten()
    if verbose:
        print("5. input_nodes_indices")
        print(input_nodes_indices)

    output_nodes_start_indices: Int[Tensor, "POPULATION_SIZE"] = (
        input_nodes_start_indices + NUM_INPUTS
    )
    if verbose:
        print("6. output_nodes_start_indices")
        print(output_nodes_start_indices)

    output_nodes_indices: Int[Tensor, "NUM_OUTPUTSxPOPULATION_SIZE"] = (
        output_nodes_start_indices.unsqueeze(1) + torch.arange(NUM_OUTPUTS)
    ).flatten()
    if verbose:
        print("7. output_nodes_indices")
        print(output_nodes_indices)

    nodes_indices = torch.arange(1, len(n_mean_m2_x))
    mutable_nodes_indices: Int[Tensor, "TNMN"] = nodes_indices[
        ~torch.isin(nodes_indices, input_nodes_indices)
    ]
    if verbose:
        print("8. mutable_nodes_indices")
        print(mutable_nodes_indices)

    nets_num_mutable_nodes: Int[Tensor, "4"] = nets_num_nodes - NUM_INPUTS
    nets_cum_num_mutable_nodes: Int[Tensor, "4"] = torch.cumsum(
        nets_num_mutable_nodes, 0
    )
    in_nodes_indices: Int[Tensor, "TNMN 3"] = torch.empty(
        (nets_num_mutable_nodes.sum(), 3), dtype=torch.int32
    )
    for i in range(POPULATION_SIZE):
        start: int = 0 if i == 0 else nets_cum_num_mutable_nodes[i - 1].item()
        end: int = nets_cum_num_mutable_nodes[i].item()
        net_in_nodes_indices: Int[Tensor, "NET_NMN 3"] = nets[i].in_nodes_indices
        in_nodes_indices[start:end] = (
            net_in_nodes_indices
            + (net_in_nodes_indices >= 0) * input_nodes_start_indices[i]
        )
    in_nodes_indices = torch.relu(in_nodes_indices)  # Map the -1s to 0s
    flat_in_nodes_indices: Int[Tensor, "TNMNx3"] = in_nodes_indices.flatten()
    if verbose:
        print("9. in_nodes_indices")
        print(in_nodes_indices)
        print(in_nodes_indices.shape)
        print(flat_in_nodes_indices)

    weights: Float[Tensor, "TNMN 3"] = torch.cat([net.weights for net in nets])
    if verbose:
        print("10. weights")
        print(weights)
        print(weights.shape)

    num_network_passes_per_input: Int[Tensor, "POPULATION_SIZE"] = torch.tensor(
        [net.num_network_passes_per_input for net in nets]
    )
    max_num_network_passes_per_input: int = max(num_network_passes_per_input).item()
    num_network_passes_per_input_mask = torch.zeros(
        (max_num_network_passes_per_input, nets_num_mutable_nodes.sum())
    )
    if verbose:
        print("11. num_network_passes_per_input")
        print(num_network_passes_per_input)
    for i in range(max_num_network_passes_per_input):
        for j in range(POPULATION_SIZE):
            if nets[j].num_network_passes_per_input > i:
                start = 0 if j == 0 else nets_cum_num_mutable_nodes[j - 1]
                end = nets_cum_num_mutable_nodes[j]
                num_network_passes_per_input_mask[i][start:end] = 1
    num_network_passes_per_input_mask = num_network_passes_per_input_mask.bool()

    if verbose:
        print("12. num_network_passes_per_input_mask")
        print(num_network_passes_per_input_mask)
        print(num_network_passes_per_input_mask.shape)

    for i in range(5):  # Imagine iterating through the environment

        if verbose:
            print(f"Iteration {i}")

        obs: Float[Tensor, "POPULATION_SIZE NUM_INPUTS"] = torch.randn(
            POPULATION_SIZE, NUM_INPUTS
        )
        if verbose:
            print("13. obs")
            print(obs)
            print(obs.shape)

        flat_obs: Float[Tensor, "POPULATION_SIZExNUM_INPUTS"] = obs.flatten()
        if verbose:
            print("14. flat_obs")
            print(flat_obs)
            print(flat_obs.shape)

        x: Float[Tensor, "TNNplus1"] = n_mean_m2_x[:, 3]
        x[input_nodes_indices] = flat_obs
        if verbose:
            print("15. x")
            print(x)
            print(x.shape)

        x: Float[Tensor, "TNNplus1"] = wrs(x)
        if verbose:
            print("16. x")
            print(x)
            print(x.shape)

        for j in range(max_num_network_passes_per_input):

            if verbose:
                print(f"Iteration {i}, Pass {j}")

            y: Float[Tensor, "TNMN 3"] = torch.gather(
                x, 0, flat_in_nodes_indices
            ).reshape(-1, 3)
            if verbose:
                print("17. y")
                print(y)
                print(y.shape)

            z: Float[Tensor, "TNMN"] = (y * weights).sum(dim=1)
            if verbose:
                print("18. z")
                print(z)
                print(z.shape)

            if verbose:
                print("19. num_network_passes_per_input_mask[j]")
                print(num_network_passes_per_input_mask[j])

            x[mutable_nodes_indices] = torch.where(
                num_network_passes_per_input_mask[j], z, x[mutable_nodes_indices]
            )
            if verbose:
                print("20. x")
                print(x)
                print(x.shape)

            x: Float[Tensor, "TNNplus1"] = wrs(x)
            if verbose:
                print("21. x")
                print(x)
                print(x.shape)

        actions: Float[Tensor, "POPULATION_SIZE NUM_OUTPUTS"] = x[
            output_nodes_indices
        ].reshape(POPULATION_SIZE, NUM_OUTPUTS)
        if verbose:
            print("22. actions")
            print(actions)
            print(actions.shape)

    for i in range(POPULATION_SIZE):
        start = input_nodes_start_indices[i]
        end = None if i + 1 == POPULATION_SIZE else input_nodes_start_indices[i + 1]
        nets[i].n_mean_m2_x = wrs.n_mean_m2_x[start:end]
