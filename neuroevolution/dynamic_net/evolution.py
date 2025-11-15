"""Contains logic to drive the evolution of a given network.

The logic contains many branches of execution and as a result is more sensible
to run per-network rather than per-population.

Also contains several components pertinent for network computation that get
altered during network evolution.

Network computation is to occur through population-wide operations and are
thus not implemented here.

Acronyms:
`NMN` : Number of mutable (hidden and output) nodes.
`NON` : Number of output nodes.
 `NN` : Number of nodes.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import torch
from jaxtyping import Float, Int
from ordered_set import OrderedSet
from torch import Tensor
from utils.beartype import ge, le, one_of


class Node:
    def __init__(
        self: "Node",
        role: An[str, one_of("input", "hidden", "output")],
        mutable_uid: An[int, ge(0)],
        immutable_uid: An[int, ge(0)],
    ) -> None:
        """Network node/neuron.

        Three types of nodes:

        Input nodes:
        ------------
        - There are as many input nodes as there are input signals.
        - Each input node is assigned an input value and forwards it to nodes
        that it connects to.
        - Input nodes are non-parametric and do not receive signal from other
        nodes.

        Hidden nodes:
        -------------
        - Hidden nodes are mutable parametric nodes that receive/emit signal
        from/to other nodes.
        - Hidden nodes have at most 3 incoming connections.
        - Hidden nodes' weights are randomly set when another node connects to
        it and then kept frozen. They do not have biases.
        - During a network pass, a hidden node runs the operation
        `standardize(weights · in_nodes' outputs)`

        Output nodes:
        -------------
        - Output nodes inherit all hidden nodes' properties.
        - There are as many output nodes as there are expected output signal
        values.
        """
        self.role: An[str, one_of("input", "hidden", "output")] = role
        # `mutable_uid` depends on the number of nodes presently in the network.
        self.mutable_uid: int = mutable_uid
        # `immutable_uid` depends on the total number of nodes ever grown.
        self.immutable_uid: int = immutable_uid
        self.out_nodes: list[Node] = []
        if self.role != "input":
            self.in_nodes: list[Node] = []
            self.weights: list[float] = [0, 0, 0]

    def __repr__(self: "Node") -> str:
        """Examples:
        Input node:  ('x',) → (0, '0') → ((6, '5'), (8, '7'))
        Hidden node: ((0, '0'), (6, '5')) → (8, '7') → ((3, '3'),)
        Output node: ((7, '6'), (8, '7')) → (3, '3') → ('y',)
        """
        in_nodes: str = ""
        if self.role == "input":
            in_nodes = "x"
        else:
            for node in self.in_nodes:
                in_nodes += f"{node.mutable_uid}-{node.immutable_uid},"
            in_nodes = in_nodes[:-1]

        out_nodes: str = ""
        if self.role == "output":
            out_nodes = "y"
        else:
            for node in self.out_nodes:
                out_nodes += f"{node.mutable_uid}-{node.immutable_uid},"
            out_nodes = out_nodes[:-1]
        return (
            str([] if self.role == "input" else self.weights)
            + " : "
            + in_nodes
            + " → "
            + f"{self.mutable_uid}-{self.immutable_uid}"
            + " → "
            + out_nodes
        )

    def sample_nearby_node(
        self: "Node",
        nodes_considered: OrderedSet["Node"],
        local_connectivity_probability: float,
    ) -> "Node":
        # Start with nodes within distance of 1.
        nodes_within_distance_i: OrderedSet[Node] = OrderedSet(
            ([] if self.role == "input" else self.in_nodes) + self.out_nodes
        )
        # Iterate while no node has been found.
        node_found: bool = False
        while not node_found:
            nodes_considered_at_distance_i: OrderedSet[Node] = (
                nodes_within_distance_i & nodes_considered
            )
            # 1) Having `nodes_considered_at_distance_i` be non-empty is not
            # sufficient to sample from it. `local_connectivity_probability`
            # controls how likely we are to sample from it at every iteration.
            # 2) If `nodes_within_distance_i` == `nodes_considered`: we've
            # exhausted the search, time to sample.
            if (
                local_connectivity_probability > random.random()
                and nodes_considered_at_distance_i
            ) or nodes_within_distance_i == nodes_considered:
                nearby_node: Node = random.choice(nodes_considered_at_distance_i)
                node_found: bool = True
            else:
                # Expand the search to nodes within distance of i+1.
                nodes_within_distance_iplus1: OrderedSet[Node] = (
                    nodes_within_distance_i.copy()
                )
                for node in nodes_within_distance_i:
                    nodes_within_distance_iplus1 |= OrderedSet(
                        ([] if node.role == "input" else node.in_nodes)
                        + node.out_nodes,
                    )
                if nodes_within_distance_iplus1 != nodes_within_distance_i:
                    nodes_within_distance_i = nodes_within_distance_iplus1
                # If we've reached the end of the connected sub-graph,
                # increase the search range to all nodes considered.
                else:
                    nodes_within_distance_i = OrderedSet(nodes_considered)

        return nearby_node

    def connect_to(self: "Node", node: "Node") -> None:
        weight: float = torch.randn(1).item()  # Standard random weight.
        node.weights[len(node.in_nodes)] = weight
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:
        i = node.in_nodes.index(self)
        # Reposition the node's weights
        if i == 0:
            node.weights[0] = node.weights[1]
        if i in (0, 1):
            node.weights[1] = node.weights[2]
        node.weights[2] = 0
        self.out_nodes.remove(node)
        node.in_nodes.remove(self)


@dataclass
class NodeList:
    """Holds `Node` instances for ease of manipulation."""

    all: list["Node"] = field(default_factory=list)
    input: list["Node"] = field(default_factory=list)
    hidden: list["Node"] = field(default_factory=list)
    output: list["Node"] = field(default_factory=list)
    # List of nodes that are receiving information from a source. Nodes appear
    # in this list once per source
    receiving: list["Node"] = field(default_factory=list)
    # List of nodes that are emitting information to a target. Nodes appear in
    # this list once per target
    emitting: list["Node"] = field(default_factory=list)
    # List of nodes currently being pruned. As a pruning operation can kickstart
    # a series of other pruning operations, this list is used to prevent
    # infinite loops
    being_pruned: list["Node"] = field(default_factory=list)

    def __iter__(
        self: "NodeList",
    ) -> Iterator[list["Node"] | list[list["Node"]]]:
        return iter(
            [
                self.all,
                self.input,
                self.hidden,
                self.output,
                self.receiving,
                self.emitting,
                self.being_pruned,
            ],
        )


class Net:
    """Network that expands/contracts through architectural mutations
    `grow_node` and `prune_node` called through the `mutate` method."""

    def __init__(
        self: "Net",
        num_inputs: An[int, ge(1)],
        num_outputs: An[int, ge(1)],
    ) -> None:
        self.num_inputs: An[int, ge(1)] = num_inputs
        self.num_outputs: An[int, ge(1)] = num_outputs
        self.total_num_nodes_grown: An[int, ge(0)] = 0
        self.nodes: NodeList = NodeList()
        # A list that contains all mutable nodes' weights.
        self.weights_list: list[list[float]] = []
        # A tensor that contains all nodes' computation parameters.
        # `n`, `mean` and `m2` are used for the Welford running
        # standardization. `x` are the node's computed outputs.
        self.n_mean_m2_x: Float[Tensor, "NN 4"] = torch.zeros((0, 4))
        # A mutable value that controls the average number of chained
        # `grow_node` mutations to perform per mutation call.
        self.avg_num_grow_mutations: An[float, ge(0)] = 1.0
        # A mutable value that controls the average number of chained
        # `prune_node` mutations to perform per mutation call.
        self.avg_num_prune_mutations: An[float, ge(0)] = 0.5
        # A mutable value that controls the number of passes through the network
        # per input.
        self.num_network_passes_per_input: An[int, ge(1)] = 1
        # A mutable value that controls increased/decreased chance for local
        # connectivity. More details in `Node.sample_nearby_node`.
        self.local_connectivity_probability: An[float, ge(0), le(1)] = 0.5
        self.initialize_architecture()

    def initialize_architecture(self: "Net") -> None:
        for _ in range(self.num_inputs):
            self.grow_node(role="input")
        for _ in range(self.num_outputs):
            self.grow_node(role="output")

    def grow_node(
        self: "Net",
        in_node_1: Node | None = None,
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> Node:
        """Method first called during initialization to grow the irremovable
        input and output nodes.

        Post-initialization, all calls create new hidden nodes.
        In such setting, three existing nodes are sampled: 2 to connect from
        and 1 to connect to."""
        new_node = Node(
            role,
            mutable_uid=len(self.nodes.all),
            immutable_uid=self.total_num_nodes_grown,
        )
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
        elif role == "output":
            self.nodes.output.append(new_node)
        else:  # role == "hidden"
            receiving_nodes_set: OrderedSet[Node] = OrderedSet(self.nodes.receiving)
            non_emitting_input_nodes: OrderedSet[Node] = OrderedSet(
                self.nodes.input
            ) - (OrderedSet(self.nodes.input) & OrderedSet(self.nodes.emitting))
            non_receiving_output_nodes: OrderedSet[Node] = OrderedSet(
                self.nodes.output
            ) - (OrderedSet(self.nodes.output) & OrderedSet(self.nodes.receiving))
            # 1) `in_node_1' → `new_node`
            if not in_node_1:
                # First focus on connecting input nodes to the rest of the
                # network.
                nodes_considered_for_in_node_1: OrderedSet[Node] = (
                    non_emitting_input_nodes
                    if non_emitting_input_nodes
                    else receiving_nodes_set
                )
                in_node_1 = random.choice(nodes_considered_for_in_node_1)
            non_emitting_input_nodes -= OrderedSet([in_node_1])
            self.grow_connection(in_node=in_node_1, out_node=new_node)
            # 2) `in_node_2' → `new_node`
            nodes_considered_for_in_node_2: OrderedSet[Node] = (
                non_emitting_input_nodes
                if non_emitting_input_nodes
                else receiving_nodes_set
            ) - OrderedSet([in_node_1])
            in_node_2: Node = in_node_1.sample_nearby_node(
                nodes_considered_for_in_node_2,
                self.local_connectivity_probability,
            )
            self.grow_connection(in_node=in_node_2, out_node=new_node)
            # 3) `new_node' → `out_node_1`
            if non_receiving_output_nodes:
                # First focus on connecting output nodes to the rest of the
                # network.
                nodes_considered_for_out_node_1: OrderedSet[Node] = (
                    non_receiving_output_nodes.copy()
                )
            else:
                nodes_considered_for_out_node_1: OrderedSet[Node] = OrderedSet()
                for node in self.nodes.hidden + self.nodes.output:
                    if len(node.in_nodes) < 3:
                        nodes_considered_for_out_node_1.add(node)
            out_node_1: Node = in_node_2.sample_nearby_node(
                nodes_considered_for_out_node_1,
                self.local_connectivity_probability,
            )
            self.grow_connection(in_node=new_node, out_node=out_node_1)
            self.nodes.hidden.append(new_node)
        if role in ["hidden", "output"]:
            self.weights_list.append(new_node.weights)
        self.n_mean_m2_x = torch.cat((self.n_mean_m2_x, torch.zeros((1, 4))))
        self.total_num_nodes_grown += 1
        return new_node

    def grow_connection(self: "Net", in_node: Node, out_node: Node) -> None:
        in_node.connect_to(out_node)
        self.nodes.receiving.append(out_node)
        self.nodes.emitting.append(in_node)

    def prune_node(self: "Net", node_being_pruned: Node | None = None) -> None:
        """Removes an existing hidden node."""
        if not node_being_pruned:
            if len(self.nodes.hidden) == 0:
                return
            node_being_pruned = random.choice(self.nodes.hidden)
        if node_being_pruned in self.nodes.being_pruned:
            return
        self.nodes.being_pruned.append(node_being_pruned)
        self.weights_list.remove(node_being_pruned.weights)
        self.n_mean_m2_x = torch.cat(
            (
                self.n_mean_m2_x[: node_being_pruned.mutable_uid],
                self.n_mean_m2_x[node_being_pruned.mutable_uid + 1 :],
            )
        )
        for node_being_pruned_out_node in node_being_pruned.out_nodes.copy():
            self.prune_connection(
                in_node=node_being_pruned,
                out_node=node_being_pruned_out_node,
                node_being_pruned=node_being_pruned,
            )
        for node_being_pruned_in_node in node_being_pruned.in_nodes.copy():
            self.prune_connection(
                in_node=node_being_pruned_in_node,
                out_node=node_being_pruned,
                node_being_pruned=node_being_pruned,
            )
        for node_list in self.nodes:
            while node_being_pruned in node_list:
                node_list.remove(node_being_pruned)
        for node in self.nodes.all:
            if node.mutable_uid > node_being_pruned.mutable_uid:
                node.mutable_uid -= 1

    def prune_connection(
        self: "Net", in_node: Node, out_node: Node, node_being_pruned: Node
    ) -> None:
        """Called by `prune_node` to remove the `node_being_pruned`'s
        connections.

        Any hidden node that becomes disconnected from the network as a result
        is also pruned."""
        if in_node not in out_node.in_nodes:
            return
        in_node.disconnect_from(out_node)
        self.nodes.receiving.remove(out_node)
        self.nodes.emitting.remove(in_node)
        if (
            in_node is not node_being_pruned
            and in_node in self.nodes.hidden
            and in_node not in self.nodes.emitting
        ):
            self.prune_node(in_node)
        if (
            out_node is not node_being_pruned
            and out_node in self.nodes.hidden
            and out_node not in self.nodes.receiving
        ):
            self.prune_node(out_node)

    def mutate(self: "Net") -> None:
        # PARAMETER PERTURBATION
        # `avg_num_grow_mutations`
        rand_val: float = 1.0 + 0.01 * torch.randn(1).item()
        self.avg_num_grow_mutations *= rand_val
        # `avg_num_prune_mutations`
        rand_val: float = 1.0 + 0.01 * torch.randn(1).item()
        self.avg_num_prune_mutations *= rand_val
        # `num_network_passes_per_input`
        rand_val: An[int, ge(1), le(100)] = torch.randint(1, 101, (1,)).item()
        if rand_val == 1 and self.num_network_passes_per_input != 1:
            self.num_network_passes_per_input -= 1
        if rand_val == 100:
            self.num_network_passes_per_input += 1
        # `local_connectivity_temperature`
        rand_val: float = 0.01 * torch.randn(1).item()
        self.local_connectivity_probability += rand_val
        if self.local_connectivity_probability < 0:
            self.local_connectivity_probability = 0
        if self.local_connectivity_probability > 1:
            self.local_connectivity_probability = 1

        # ARCHITECTURE PERTURBATION
        # `prune_node`
        rand_val: An[float, ge(0), le(1)] = float(torch.rand(1))
        if (self.avg_num_prune_mutations % 1) < rand_val:
            num_prune_mutations: An[int, ge(0)] = int(self.avg_num_prune_mutations)
        else:
            num_prune_mutations: An[int, ge(1)] = int(self.avg_num_prune_mutations) + 1
        for _ in range(num_prune_mutations):
            self.prune_node()
        # `grow_node`
        rand_val: An[float, ge(0), le(1)] = float(torch.rand(1))
        if (self.avg_num_grow_mutations % 1) < rand_val:
            num_grow_mutations: An[int, ge(0)] = int(self.avg_num_grow_mutations)
        else:
            num_grow_mutations: An[int, ge(1)] = int(self.avg_num_grow_mutations) + 1
        starting_node = None
        for _ in range(num_grow_mutations):
            # Chained `grow_node` mutations re-use the previously created
            # hidden node.
            starting_node = self.grow_node(in_node_1=starting_node)

        # NETWORK COMPUTATION COMPONENTS GENERATION
        mutable_nodes: list[Node] = self.nodes.output + self.nodes.hidden
        # A tensor that contains all nodes' in nodes' mutable ids. Used during
        # computation to fetch the correct values from the `outputs` attribute.
        self.in_nodes_indices: Int[Tensor, "NMN 3"] = -1 * torch.ones(
            (len(mutable_nodes), 3), dtype=torch.int32
        )
        for i, mutable_node in enumerate(mutable_nodes):
            for j, mutable_node_in_node in enumerate(mutable_node.in_nodes):
                self.in_nodes_indices[i][j] = mutable_node_in_node.mutable_uid
        self.weights: Float[Tensor, "NMN 3"] = torch.tensor(
            self.weights_list, dtype=torch.float32
        )
