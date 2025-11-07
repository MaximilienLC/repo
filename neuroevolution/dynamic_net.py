"""Dynamically complexifying neural network."""

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from ordered_set import OrderedSet
from torch import Tensor
from utils.beartype import ge, le, one_of

log = logging.getLogger(__name__)


class Node:
    def __init__(
        self: "Node",
        role: An[str, one_of("input", "hidden", "output")],
        id: An[int, ge(0)],
    ) -> None:
        """`input` nodes: There are as many input nodes as there are input
        signals. Each input node is assigned an input value and forwards it to
        nodes that it connects to. Input nodes are non-parametric and do not
        receive signal from other nodes.

        `output` nodes: There are as many output nodes as there are expected
        output signal values. Output nodes are parametric nodes that
        receive/emit signal from/to other nodes (or themselves). During
        a network pass, an output node runs the operation
        `standardize(weights · inputs) + bias`.

        `hidden` nodes: Hidden nodes are parametric nodes that receive/emit
        signal from/to other nodes (or themselves). Unlike output nodes, they
        do not have biases. During a network pass, a hidden node runs the
        operation `standardize(weights · inputs)`.

        Node outputs are standardized.
        Nodes do not have biases. All node outputs are standardized.
        """
        self.role: An[str, one_of("input", "hidden", "output")] = role
        """Each node has a separate identifier."""
        self.id = id
        self.in_nodes: list[Node] = []
        self.out_nodes: list[Node] = []
        if self.role != "input":
            self.weights: list[float] = [0, 0, 0]
            self.num_in_nodes = 0

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
        """Start with nodes within distance of 1."""
        nodes_within_distance_i: OrderedSet["Node"] = OrderedSet(
            self.in_nodes + self.out_nodes
        )
        """Iterate while no node has been found."""
        node_found: bool = False
        while not node_found:
            nodes_considered_at_distance_i: OrderedSet["Node"] = (
                nodes_within_distance_i & nodes_considered
            )
            if nodes_considered_at_distance_i:
                nearby_node: Node = random.choice(nodes_considered_at_distance_i)
                node_found: bool = True
            else:
                """Expand search to nodes within distance of i+1."""
                temp: OrderedSet["Node"] = nodes_within_distance_i.copy()
                for node in nodes_within_distance_i:
                    temp |= OrderedSet(
                        node.in_nodes + node.out_nodes,
                    )
                """If all nodes within distance i+1 have been considered,
                increase the search range to all `nodes_considered`."""
                if nodes_within_distance_i == temp:
                    nodes_within_distance_i = OrderedSet(nodes_considered)
                else:
                    nodes_within_distance_i = temp
        return nearby_node

    def connect_to(self: "Node", node: "Node") -> None:
        weight: Float[Tensor, "1"] = torch.randn(1)
        node.weights[node.num_in_nodes] = weight
        node.num_in_nodes += 1
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:
        i = node.in_nodes.index(self)
        if i == 0:
            node.weights[0] = node.weights[1]
        if i in (0, 1):
            node.weights[1] = node.weights[2]
        node.weights[2] = 0
        node.num_in_nodes -= 1
        self.out_nodes.remove(node)
        node.in_nodes.remove(self)


@dataclass
class NodeList:
    """Holds `Node` instances."""

    all: list["Node"] = field(default_factory=list)
    input: list["Node"] = field(default_factory=list)
    hidden: list["Node"] = field(default_factory=list)
    output: list["Node"] = field(default_factory=list)
    receiving: list["Node"] = field(default_factory=list)
    """List of nodes that are receiving information from a source. Nodes
    appear in this list once per source."""
    emitting: list["Node"] = field(default_factory=list)
    """List of nodes that are emitting information to a target. Nodes
    appear in this list once per target."""
    being_pruned: list["Node"] = field(default_factory=list)
    """List of nodes currently being pruned. As a pruning operation can
    kicksart a series of other pruning operations, this list is used to
    prevent infinite loops.
    """

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
        self.initialize_architecture()
        self.weights: list[Float[Tensor, "3"]] = []
        """A tensor containing the weights for each `output` and `hidden` node.
        i) Each node can have at most 3 incoming connections.
        ii) Given that all node outputs are standardized, there are no biases in
        this network."""
        self.outputs: Float[Tensor, "NHON"] = torch.empty(
            size=(self.config.num_outputs,), dtype=torch.float32
        )
        """A tensor containing the latest output values for each `output` and `hidden` node."""
        self.avg_num_grow_mutations: An[float, ge(0)] = 1.0
        """A mutable value that controls the average number of chained
        :meth:`grow_node` mutations to perform per mutation call."""
        self.avg_num_prune_mutations: An[float, ge(0)] = 0.5
        """A mutable value that controls the average number of chained
        :meth:`prune_node` mutations to perform per mutation call."""
        self.num_network_passes_per_input: An[int, ge(1)] = 1
        """
        A mutable value that controls the number of passes through the
        network per input."""

    def initialize_architecture(self: "DynamicNet") -> None:
        for _ in range(self.config.num_inputs):
            self.grow_node(role="input")
        for _ in range(self.config.num_outputs):
            self.grow_node(role="output")

    def mutate(self: "DynamicNet") -> None:
        self.mutate_parameters()
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
            starting_node = self.grow_node(in_node_1=starting_node)

    def mutate_parameters(self: "DynamicNet") -> None:
        # `num_grow_mutations`
        rand_val: float = 1.0 + float(torch.rand(1)) * 0.01
        self.num_grow_mutations *= rand_val
        # `num_prune_mutations`
        rand_val: float = 1.0 + float(torch.rand(1)) * 0.01
        self.num_prune_mutations *= rand_val
        # `num_network_passes_per_input`
        rand_val: An[int, ge(1), le(100)] = int(torch.randint(1, 101, (1,)))
        if rand_val == 1 and self.num_network_passes_per_input != 1:
            self.num_network_passes_per_input -= 1
        if rand_val == 100:
            self.num_network_passes_per_input += 1

    def grow_node(
        self: "DynamicNet",
        in_node_1: Node | None = None,
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> Node:
        new_node = Node(role, id=len(self.nodes.all))
        log.debug(f"New {role} node: {new_node} w/ id {new_node.id}.")
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
        elif role == "output":
            self.nodes.output.append(new_node)
        else:  # role == "hidden"
            self.nodes.hidden.append(new_node)
            receiving_nodes_set = OrderedSet(self.nodes.receiving)
            """`in_node_1' → `new_node`"""
            if not in_node_1:
                in_node_1 = random.choice(receiving_nodes_set)
            self.grow_connection(in_node=in_node_1, out_node=new_node)
            log.debug(
                f"Connected from {in_node_1} w/ id {in_node_1.id}",
            )
            """`in_node_2' → `new_node`"""
            in_node_2: Node = in_node_1.sample_nearby_node(
                nodes_considered=receiving_nodes_set,
            )
            self.grow_connection(in_node=in_node_2, out_node=new_node)
            log.debug(
                f"Connected from {in_node_2} w/ id {in_node_2.id}",
            )
            """`new_node' → `out_node_1`"""
            nodes_considered = OrderedSet()
            for node in self.nodes.hidden + self.nodes.output:
                if node.num_in_nodes < 3:
                    nodes_considered.add(node)
            out_node_1: Node = new_node.sample_nearby_node(
                nodes_considered=OrderedSet(self.nodes.hidden + self.nodes.output),
            )
            self.grow_connection(in_node=new_node, out_node=out_node_1)
            log.debug(f"Connected to {out_node_1} w/ id {out_node_1.id}")
        if role in ["hidden", "output"]:
            self.weights = torch.cat((self.weights, node.weights))
            self.outputs = torch.cat(self.outputs, torch.tensor([0.0]))
        return new_node

    def grow_connection(
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
    ) -> None:
        in_node.connect_to(out_node)
        self.nodes.receiving.append(out_node)
        self.nodes.emitting.append(in_node)

    def prune_node(
        self: "DynamicNet",
        node_to_prune: Node | None = None,
    ) -> None:
        node = node_to_prune
        if not node:
            if len(self.nodes.hidden) == 0:
                return
            node = random.choice(self.nodes.hidden)
        if node in self.nodes.being_pruned:
            return
        self.nodes.being_pruned.append(node)
        for out_node in node.out_nodes.copy():
            self.prune_connection(node, out_node, node)
        for in_node in node.in_nodes.copy():
            self.prune_connection(in_node, node, node)
        for node_list in self.nodes:
            while node in node_list:
                node_list.remove(node)

    def prune_connection(
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
        current_node_in_focus: Node,
    ) -> None:
        if in_node not in out_node.in_nodes:
            return
        in_node.disconnect_from(out_node)
        self.nodes.receiving.remove(out_node)
        self.nodes.emitting.remove(in_node)
        if (
            in_node is not current_node_in_focus
            and in_node not in self.nodes.emitting
            and in_node in self.nodes.hidden
        ):
            self.prune_node(in_node)
        if (
            out_node is not current_node_in_focus
            and out_node not in self.nodes.receiving
            and out_node in self.nodes.hidden
        ):
            self.prune_node(out_node)
