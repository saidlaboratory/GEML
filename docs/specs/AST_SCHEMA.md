# Binary AST schema

The binary-AST contract consists of `ASTNode`, `ASTEdge`, `ASTStatistics`, and `ASTTree` in
`geml.contracts.ast`. These models describe and validate records only; they do not expose tree
construction or traversal APIs.

## Authority and conventions

The ordered `nodes` and `edges` in `ASTTree`, together with `root_id`, are the authoritative AST
structure. Every parent-to-child reference has an explicit `child_slot`; slots are zero-based and
binary, so valid values are `0` and `1`. Repeated mathematical subexpressions remain distinct
nodes in an AST tree. Structural DAG sharing is a separate Goal 3 contract.

Depth uses the fixed convention that a leaf has depth `0`. A tree containing an operator must
therefore have depth at least `1`.

## Fields

### `ASTNode`

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `node_id` | yes | nonempty string | Identity unique within the tree. |
| `node_kind` | yes | nonempty string | Producer-defined node type; this contract does not freeze an operator registry. |
| `label` | yes | nonempty string | Structural node label. |
| `arity` | yes | integer from 0 through 2 | Number of explicit ordered children. |
| `value` | no | JSON value | Optional literal/source value data; defaults to null. |
| `metadata` | no | JSON object | Optional JSON-compatible node metadata; defaults to `{}`. |

### `ASTEdge`

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `source_id` | yes | nonempty string | Parent node ID. |
| `target_id` | yes | nonempty string | Child node ID. |
| `child_slot` | yes | integer 0 or 1 | Explicit ordered child position. |

### `ASTStatistics`

All fields are required integers: `node_count` (at least 1), `edge_count` (nonnegative),
`leaf_count` (at least 1), `operator_count` (nonnegative), and `depth` (nonnegative).
`node_count` must equal `leaf_count + operator_count`, and `edge_count` must equal
`node_count - 1`. For a realizable binary tree, `leaf_count` cannot exceed
`operator_count + 1`; depth is bounded above by `operator_count` and below by the binary-tree
capacity required for the declared node and operator counts.

### `ASTTree`

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `expression_id` | yes | nonempty string | Source expression identity. |
| `root_id` | yes | nonempty string | ID of the root in `nodes`. |
| `nodes` | yes | nonempty array of `ASTNode` | Immutable node collection. |
| `edges` | no | array of `ASTEdge` | Immutable edge collection; defaults to empty. |
| `statistics` | yes | `ASTStatistics` | Declared statistics checked against local records. |

The tree rejects duplicate node IDs, missing roots/endpoints, self-references, multiple parents,
duplicate parent slots, absent declared child slots, edges from leaves, unreachable/cyclic
components, and statistic/count/depth mismatches. Every non-root node must have exactly one
parent, and every node must be reachable from the root. Numeric fields require JSON integers, so
booleans are not accepted as integer values. These checks are private record validation; parsing,
construction, and reusable traversal APIs remain the responsibility of downstream issues.

## JSON-compatible example

```json
{
  "expression_id": "expr-000001",
  "root_id": "n0",
  "nodes": [
    {"node_id": "n0", "node_kind": "operator", "label": "Add", "arity": 2, "value": null, "metadata": {}},
    {"node_id": "n1", "node_kind": "variable", "label": "x", "arity": 0, "value": "x", "metadata": {}},
    {"node_id": "n2", "node_kind": "constant", "label": "Integer", "arity": 0, "value": 1, "metadata": {}}
  ],
  "edges": [
    {"source_id": "n0", "target_id": "n1", "child_slot": 0},
    {"source_id": "n0", "target_id": "n2", "child_slot": 1}
  ],
  "statistics": {"node_count": 3, "edge_count": 2, "leaf_count": 2, "operator_count": 1, "depth": 1}
}
```

## Scope and consumers

Issue 1-6 produces this contract, Goal 2 consumes it for official EML expansion, and Goal 3
adapts it into separately defined graph/DAG contracts. SymPy parsing, n-ary folding, AST building,
traversal, symbolic simplification, semantic equivalence, and graph/DAG sharing are out of scope.
