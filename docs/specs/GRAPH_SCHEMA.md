# Generic graph and DAG schema

Goal 3 uses one graph contract for AST, pure EML, macro, and motif
representations. The contract preserves ordered child slots, repeated child
references, ordered roots, and exact structural identity. It does not perform
tree-to-DAG conversion or semantic rewriting.

## Records

`ChildRef(slot, target_id)` is one explicit parent-to-child reference. Slots
are zero-based and contiguous. Two slots may reference the same target; both
records remain present and both contribute to the child-reference count.

`GraphRoot(root_id, target_id, representation_mode)` is one ordered root
reference. `root_id` identifies the occurrence (normally an expression ID),
`target_id` selects its graph node, and `representation_mode` preserves the
exact representation label such as `ast`, `pure_eml:official_v4`, or an
explicitly separate compiler mode. Root IDs are unique, while multiple roots
may deliberately reference the same target node.

`GraphNode(node_id, family, kind, label, value, children)` contains:

- a graph-local node identity;
- one representation family: `ast`, `eml`, `macro`, or `motif`;
- structural kind, label, and a strict JSON value field;
- an immutable tuple of ordered child references.

JSON objects and arrays are copied recursively into immutable snapshots while
retaining normal dictionary/list read behavior and canonical JSON encoding.
Tuples are not JSON arrays and are rejected rather than silently normalized.
This prevents caller-owned nested values from changing a node's structural
identity after construction.

`Graph(nodes, roots)` snapshots its input mapping and root records. Every
graph has at least one root, every node is reachable from a root, and all nodes
in one graph use the same representation family.

`GraphStatistics` reports unique nodes, explicit child references
(`edge_count` and its `child_reference_count` alias), leaves, roots, and
leaf-zero maximum depth. Statistics are computed only after validation, so
missing references, cycles, or unreachable components cannot be silently
excluded.

## Canonical structural signatures

`compute_signature(graph, node_id)` returns a deterministic 64-character
lowercase SHA-256 digest. Each node payload includes:

- signature format version;
- representation family;
- node kind, label, and typed JSON value;
- arity;
- every child slot and the corresponding child signature in slot order.

Node IDs are deliberately excluded. Isomorphic subtrees therefore have the
same signature even when their local IDs differ. Representation changes,
typed value changes, and ordered-child changes produce different signatures.
The implementation is iterative and memoizes shared descendants, so signing a
DAG does not re-expand it.

Signatures establish structural identity only. For example, `x + x` and
`2 * x` remain distinct even where a mathematical domain makes them
semantically equivalent.

## Validation

`validate_graph` retains all detected failures and checks:

1. nonempty records, actual `GraphNode` mapping values, supported
   representation families, mapping-key identity, nonblank fields, and strict
   finite JSON values;
2. root existence;
3. child target existence, nonnegative unique slots, and contiguous slot
   coverage;
4. reachability from the declared roots;
5. acyclicity across every component;
6. representation-specific purity.

The authoritative AST contract intentionally leaves `node_kind` and `label`
producer-defined. AST purity therefore enforces the frozen binary arity and
record rules without inventing an operator allowlist.

Pure EML graphs mirror the concrete Goal 2 IR exactly:

- an internal `eml` node is labeled `eml`, has no value, and has slots 0 and 1;
- a `variable` leaf uses the same valid source name as its label and value;
- a primitive `one` leaf is labeled `1` and has the exact integer value `1`.

No constant, macro, template, derived operation, or compound source syntax is
accepted inside a pure EML graph. Macro and motif families remain
representation-neutral because their vocabularies belong to later owning
issues.

Malformed mapping values are retained as validation errors. They do not abort
later root, reachability, cycle, or purity checks.

## Scope

This schema defines graph records, identity, statistics, and validation only.
AST conversion, EML conversion, direct compilation, experiment processing,
analysis, and semantic equivalence are owned by later Goal 3 issues.
