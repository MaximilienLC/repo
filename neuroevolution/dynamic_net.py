"""Dynamically complexifying neural network.

Network passes occur through population-wide operations and are thus not
implemented here.

Acronyms:
    NHON: Number of hidden and output nodes.
    NON:  Number of output nodes.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import torch
from jaxtyping import Float
from ordered_set import OrderedSet
from torch import Tensor
from utils.beartype import ge, le, one_of


class Node:
    def __init__(
        self: "Node",
        role: An[str, one_of("input", "hidden", "output")],
        id: An[int, ge(0)],
    ) -> None:
        """`input nodes`:
        - There are as many input nodes as there are input signals.
        - Each input node is assigned an input value and forwards it to nodes
        that it connects to.
        - Input nodes are non-parametric and do not receive signal from other
        nodes.

        `hidden nodes`:
        - Hidden nodes are mutable parametric nodes that receive/emit signal
        from/to other nodes (including themselves).
        - Hidden nodes have at most 3 incoming connections.
        - Hidden nodes' weights are randomly set when another node connects to
        it and then kept frozen. They do not have biases.
        - During a network pass, a hidden node runs the operation
        `standardize(weights · in_nodes' outputs)`

        `output nodes`:
        - Output nodes inherit all hidden nodes' properties.
        - There are as many output nodes as there are expected output signal
        values.
        - The network implements a "bias output layer" whose values are added
        to the output nodes' computation results (effectively making output
        nodes actually have biases, unlike hidden nodes).
        """
        self.role: An[str, one_of("input", "hidden", "output")] = role
        self.id = id  # Each node has a separate mutable identifier
        self.out_nodes: list[Node] = []
        if self.role != "input":
            self.in_nodes: list[Node] = []
            self.weights: Float[Tensor, "3"] = torch.zeros(size=(3,))

    def __repr__(self: "Node") -> str:
        """Examples:
        Input node: ('x',) → 3 → (5, 7)
        Hidden node: (2, 4) → 5 → (7, 9)
        Output node: (3, 6) → 8 → ('y', 10)
        """
        node_inputs: tuple[Any, ...] = tuple(
            ("x" if self.role == "input" else (node.id for node in self.in_nodes)),
        )
        node_outputs: tuple[Any, ...] = tuple(node.id for node in self.out_nodes)
        if self.role == "output":
            node_outputs = ("y", *node_outputs)
        return str(node_inputs) + " → " + str(self.id) + " → " + str(node_outputs)

    def sample_nearby_node(
        self: "Node",
        nodes_considered: OrderedSet["Node"],
    ) -> "Node":
        # Start with nodes within distance of 1
        nodes_within_distance_i: OrderedSet["Node"] = OrderedSet(
            ([] if self.role == "input" else self.in_nodes) + self.out_nodes
        )
        # Iterate while no node has been found
        node_found: bool = False
        while not node_found:
            nodes_considered_at_distance_i: OrderedSet["Node"] = (
                nodes_within_distance_i & nodes_considered
            )
            if nodes_considered_at_distance_i:
                nearby_node: Node = random.choice(nodes_considered_at_distance_i)
                node_found: bool = True
            else:
                # Expand search to nodes within distance of i+1
                temp: OrderedSet["Node"] = nodes_within_distance_i.copy()
                for node in nodes_within_distance_i:
                    temp |= OrderedSet(
                        ([] if node.role == "input" else node.in_nodes)
                        + node.out_nodes,
                    )
                # If all nodes within distance i+1 have been considered,
                # increase the search range to all `nodes_considered`
                if nodes_within_distance_i == temp:
                    nodes_within_distance_i = OrderedSet(nodes_considered)
                else:
                    nodes_within_distance_i = temp
        return nearby_node

    def connect_to(self: "Node", node: "Node") -> None:
        # Random weight
        weight: Float[Tensor, "1"] = torch.randn(1)
        node.weights[len(node.in_nodes)] = weight
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:
        i = node.in_nodes.index(self)
        # Adjust the node's weights
        if i == 0:
            node.weights[0] = node.weights[1]
        if i in (0, 1):
            node.weights[1] = node.weights[2]
        node.weights[2] = 0
        self.out_nodes.remove(node)
        node.in_nodes.remove(self)


@dataclass
class NodeList:
    """Holds `Node` instances."""

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


class DynamicNet:

    def __init__(
        self: "DynamicNet",
        num_inputs: An[int, ge(1)],
        num_outputs: An[int, ge(1)],
    ) -> None:
        self.num_inputs: An[int, ge(1)] = num_inputs
        self.num_outputs: An[int, ge(1)] = num_outputs
        self.nodes: NodeList = NodeList()
        # A list that contains all mutable nodes' weights
        self.weights: list[Float[Tensor, "3"]] = []
        # A tensor that contains all nodes' in nodes' ids. Used during
        # computation to fetch the correct values from the `outputs` attribute
        self.in_nodes_ids: Float[Tensor, "NHON 3"] = torch.empty(size=(0, 3))
        # The network's "bias layer"
        self.biases: Float[Tensor, "NON"] = torch.zeros(size=(self.num_outputs,))
        self.initialize_architecture()
        # A mutable value that controls the average number of chained
        # `grow_node` mutations to perform per mutation call
        self.avg_num_grow_mutations: An[float, ge(0)] = 1.0
        # A mutable value that controls the average number of chained
        # `prune_node` mutations to perform per mutation call
        self.avg_num_prune_mutations: An[float, ge(0)] = 0.5
        # A mutable value that controls the number of passes through the network
        # per input
        self.num_network_passes_per_input: An[int, ge(1)] = 1

    def initialize_architecture(self: "DynamicNet") -> None:
        for _ in range(self.num_inputs):
            self.grow_node(role="input")
        for _ in range(self.num_outputs):
            self.grow_node(role="output")

    def grow_node(
        self: "DynamicNet",
        in_node_1: Node | None = None,
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> Node:
        # Node with ID 0 is a shadow-node for the sake of facilitating
        # computation.
        new_node = Node(role, id=len(self.nodes.all) + 1)
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
        elif role == "output":
            self.nodes.output.append(new_node)
        # Post-initialization, all `grow_node` calls create hidden nodes
        else:  # role == "hidden"
            self.nodes.hidden.append(new_node)
            receiving_nodes_set = OrderedSet(self.nodes.receiving)
            # `in_node_1' → `new_node`
            if not in_node_1:
                in_node_1 = random.choice(receiving_nodes_set)
            self.grow_connection(in_node=in_node_1, out_node=new_node)
            # `in_node_2' → `new_node`
            in_node_2: Node = in_node_1.sample_nearby_node(
                nodes_considered=receiving_nodes_set,
            )
            self.grow_connection(in_node=in_node_2, out_node=new_node)
            # `new_node' → `out_node_1`
            nodes_considered = OrderedSet()
            for node in self.nodes.hidden + self.nodes.output:
                if len(node.in_nodes) < 3:
                    nodes_considered.add(node)
            out_node_1: Node = new_node.sample_nearby_node(nodes_considered)
            self.grow_connection(in_node=new_node, out_node=out_node_1)
        if role in ["hidden", "output"]:
            self.weights.append(node.weights)
        return new_node

    def grow_connection(
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
    ) -> None:
        in_node.connect_to(out_node)
        self.nodes.receiving.append(out_node)
        self.nodes.emitting.append(in_node)

    def prune_node(self: "DynamicNet", node_being_pruned: Node | None = None) -> None:
        if not node_being_pruned:
            if len(self.nodes.hidden) == 0:
                return
            node_being_pruned = random.choice(self.nodes.hidden)
        if node_being_pruned in self.nodes.being_pruned:
            return
        self.nodes.being_pruned.append(node_being_pruned)
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
            if node.id > node.id:
                node.id -= 1
        self.in_nodes_ids -= self.in_nodes_ids > node.id

    def prune_connection(
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
        node_being_pruned: Node,
    ) -> None:
        if in_node not in out_node.in_nodes:
            return
        in_node.disconnect_from(out_node)
        self.nodes.receiving.remove(out_node)
        self.nodes.emitting.remove(in_node)
        # If the node that is not currently being pruned is now disconnected
        # from the network, prune it.
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

    def mutate(self: "DynamicNet") -> None:
        # PARAMETERS
        # `num_grow_mutations`
        rand_val: float = 1.0 + 0.01 * torch.randn(1).item()
        self.num_grow_mutations *= rand_val
        # `num_prune_mutations`
        rand_val: float = 1.0 + 0.01 * torch.randn(1).item()
        self.num_prune_mutations *= rand_val
        # `num_network_passes_per_input`
        rand_val: An[int, ge(1), le(100)] = torch.randint(1, 101, (1,)).item()
        if rand_val == 1 and self.num_network_passes_per_input != 1:
            self.num_network_passes_per_input -= 1
        if rand_val == 100:
            self.num_network_passes_per_input += 1
        # `biases`
        rand_vals: Float[Tensor, "NON"] = 0.01 * torch.randn(self.num_outputs)
        self.biases += rand_vals

        # ARCHITECTURE
        # `prune_node`
        rand_val: An[float, ge(0), le(1)] = float(torch.rand(1))
        if rand_val > self.avg_num_prune_mutations:
            num_prune_mutations: An[int, ge(0)] = int(self.avg_num_prune_mutations)
        else:
            num_prune_mutations: An[int, ge(1)] = int(self.avg_num_prune_mutations) + 1
        for _ in range(num_prune_mutations):
            self.prune_node()
        # `grow_node`
        rand_val: An[float, ge(0), le(1)] = float(torch.rand(1))
        if rand_val > self.avg_num_grow_mutations:
            num_grow_mutations: An[int, ge(0)] = int(self.avg_num_grow_mutations)
        else:
            num_grow_mutations: An[int, ge(1)] = int(self.avg_num_grow_mutations) + 1
        starting_node = None
        for _ in range(num_grow_mutations):
            # Chained `grow_node` mutations re-use the previously created hidde
            starting_node = self.grow_node(in_node_1=starting_node)
