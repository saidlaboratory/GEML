"""Validated contracts for binary abstract syntax trees."""

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StringConstraints,
    model_validator,
)

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_BinaryArity = Annotated[StrictInt, Field(ge=0, le=2)]
_ChildSlot = Annotated[StrictInt, Field(ge=0, le=1)]


class _ASTContract(BaseModel):
    """Shared validation policy for binary-AST records."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ASTNode(_ASTContract):
    """One node in a binary AST."""

    node_id: _NonBlankStr
    node_kind: _NonBlankStr
    label: _NonBlankStr
    arity: _BinaryArity
    value: JsonValue = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ASTEdge(_ASTContract):
    """One explicit, ordered parent-to-child reference in a binary AST."""

    source_id: _NonBlankStr
    target_id: _NonBlankStr
    child_slot: _ChildSlot


class ASTStatistics(_ASTContract):
    """Precomputed size and depth statistics for an AST.

    Depth follows the project convention that every leaf has depth zero.
    """

    node_count: Annotated[StrictInt, Field(ge=1)]
    edge_count: _NonNegativeInt
    leaf_count: Annotated[StrictInt, Field(ge=1)]
    operator_count: _NonNegativeInt
    depth: _NonNegativeInt

    @model_validator(mode="after")
    def validate_counts_and_leaf_depth(self) -> Self:
        """Validate count partitioning and the leaf-depth base case."""
        if self.node_count != self.leaf_count + self.operator_count:
            raise ValueError("node_count must equal leaf_count + operator_count")
        if self.edge_count != self.node_count - 1:
            raise ValueError("edge_count must equal node_count - 1")
        if self.leaf_count > self.operator_count + 1:
            raise ValueError("leaf_count cannot exceed operator_count + 1 in a binary tree")
        if self.operator_count == 0 and self.depth != 0:
            raise ValueError("an AST containing only leaves must have depth 0")
        if self.operator_count > 0 and self.depth == 0:
            raise ValueError("an AST containing an operator must have depth at least 1")
        if self.depth > self.operator_count:
            raise ValueError("depth cannot exceed operator_count")
        minimum_depth = max(
            self.node_count.bit_length() - 1,
            self.operator_count.bit_length(),
        )
        if self.depth < minimum_depth:
            raise ValueError("depth is too small for the declared node and operator counts")
        return self


class ASTTree(_ASTContract):
    """A validated binary AST for one expression."""

    expression_id: _NonBlankStr
    root_id: _NonBlankStr
    nodes: tuple[ASTNode, ...] = Field(min_length=1)
    edges: tuple[ASTEdge, ...] = Field(default_factory=tuple)
    statistics: ASTStatistics

    @model_validator(mode="after")
    def validate_tree_records(self) -> Self:
        """Check local identity, endpoint, slot, and statistics consistency."""
        node_by_id: dict[str, ASTNode] = {}
        for node in self.nodes:
            if node.node_id in node_by_id:
                raise ValueError(f"duplicate node_id: {node.node_id}")
            node_by_id[node.node_id] = node

        if self.root_id not in node_by_id:
            raise ValueError("root_id must reference an existing node")

        children_by_id: dict[str, dict[int, str]] = {node_id: {} for node_id in node_by_id}
        incoming_counts = {node_id: 0 for node_id in node_by_id}

        for edge in self.edges:
            if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
                raise ValueError("every edge endpoint must reference an existing node")
            if edge.source_id == edge.target_id:
                raise ValueError("AST edges cannot be self-references")

            children = children_by_id[edge.source_id]
            if edge.child_slot in children:
                raise ValueError("a parent may reference only one child per child_slot")

            source_node = node_by_id[edge.source_id]
            if edge.child_slot >= source_node.arity:
                raise ValueError("child_slot must be less than the source node arity")

            children[edge.child_slot] = edge.target_id
            incoming_counts[edge.target_id] += 1
            if incoming_counts[edge.target_id] > 1:
                raise ValueError("a tree node may have at most one parent")

        if incoming_counts[self.root_id] != 0:
            raise ValueError("the root node cannot have a parent")

        for node_id, node in node_by_id.items():
            if set(children_by_id[node_id]) != set(range(node.arity)):
                raise ValueError("node arity must match its explicit ordered child slots")
            if node_id != self.root_id and incoming_counts[node_id] != 1:
                raise ValueError("every non-root node must have exactly one parent")

        depths = {self.root_id: 0}
        pending = [self.root_id]
        while pending:
            parent_id = pending.pop()
            child_depth = depths[parent_id] + 1
            for child_id in children_by_id[parent_id].values():
                depths[child_id] = child_depth
                pending.append(child_id)

        if len(depths) != len(self.nodes):
            raise ValueError("every AST node must be reachable from the root")

        expected_leaf_count = sum(node.arity == 0 for node in self.nodes)
        expected_operator_count = len(self.nodes) - expected_leaf_count
        if self.statistics.node_count != len(self.nodes):
            raise ValueError("statistics.node_count does not match nodes")
        if self.statistics.edge_count != len(self.edges):
            raise ValueError("statistics.edge_count does not match edges")
        if self.statistics.leaf_count != expected_leaf_count:
            raise ValueError("statistics.leaf_count does not match node arities")
        if self.statistics.operator_count != expected_operator_count:
            raise ValueError("statistics.operator_count does not match node arities")
        if self.statistics.depth != max(depths.values()):
            raise ValueError("statistics.depth does not match the tree structure")

        return self
