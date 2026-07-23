# Graph Schema (frozen by 3-1)

Generic graph/DAG shape that works the same way across AST, EML, macro,
and motif graphs. Everything in 3-2 through 3-8 builds on this - nobody
gets to redefine these shapes downstream.

## Core types (`src/geml/graph/schema.py`)

`ChildRef(slot, target_id)` - one edge from a parent to a kid. slot is
just position (0 = left/base, 1 = right/exponent, whatever). If two
slots point at the same target, both refs still exist separately - we
never collapse them into one, that's kind of the whole point of a DAG.

`GraphNode(node_id, family, kind, label, value, children)`
- family - "ast" | "eml" | "macro" | "motif", which representation this is
- kind - the actual operator/node type, e.g. "Add", "eml", "eml_add", "motif_17"
- label - optional extra descriptor, mostly useful for motifs (template name etc)
- value - only set for leaves (variable name, constant), None otherwise
- children - ordered tuple of ChildRef

`Graph(nodes, roots)` - nodes is just a dict keyed by id. roots is a
tuple not a single value on purpose - one shared DAG can back thousands
of expressions at once, so you need room for many entry points.

`GraphStatistics` - node_count, edge_count, leaf_count, root_count,
max_depth. Heads up: edge_count can end up bigger than node_count - 1
once sharing kicks in. That's not a bug - a plain tree always has
exactly node_count - 1 edges, but a DAG can have more since one node
gets pointed at from multiple places.

## Signatures (`src/geml/graph/signatures.py`)

compute_signature(graph, node_id) builds a string out of family, kind,
label, value, arity, and each child's own signature in order.

- Same shape → same signature.
- Swap the child order → different signature (slot number's baked into the string).
- This is structural only. x+x and 2*x mean the exact same thing
  mathematically but look different structurally, so they get
  different signatures - on purpose. Actually proving two things are
  mathematically equal is a goal 4 problem (e-graphs), not something
  this schema tries to solve.

## Validation (`src/geml/graph/validate.py`)

validate_graph(graph) runs five checks and collects every failure it
finds - doesn't just bail after the first one.

1. Roots actually exist - every id in roots needs to be a real node.
2. Child slots make sense - every ref has to point somewhere real, and no
   two children on one node can claim the same slot.
3. Reachability - nothing's allowed to just sit there disconnected
   from every root.
4. No cycles - a node can't end up being its own ancestor.
5. Purity per family - ast nodes have to use approved ast operators,
   eml nodes can only use eml/Var/Const. macro/motif aren't checked here
   since those are compiler-generated and don't have one fixed vocabulary.

Heads up: the AST/EML vocab lists in step 5 are placeholders for now,
waiting on 1-2 and 2-1 to actually merge their real contracts. Once
they do, swap AST_VOCAB/EML_VOCAB in validate.py for the real thing.

## What's NOT in here

No tree-to-DAG conversion, no compression - that's 3-2 and 3-3. This
issue is just the shape and the rules for checking it, nothing more.
