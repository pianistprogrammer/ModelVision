"""Post-processing passes applied to every :class:`ModelGraph`.

Runs after inspection, before validation/layout. Each pass is a pure
function ``pass(graph) -> ModelGraph`` — they compose linearly and never
raise (post-processing must never break rendering).
"""

from __future__ import annotations

from modelvision.core.exceptions import mv_warn
from modelvision.core.ir import Edge, LayerNode, ModelGraph
from modelvision.core.style import StyleSpec

# Threshold for auto-collapsing large models per PRD §6.4.
LARGE_MODEL_THRESHOLD = 500


def post_process(
    graph: ModelGraph,
    *,
    expand_groups: bool = False,
    insert_merge_nodes: bool = True,
    auto_collapse: bool = True,
) -> ModelGraph:
    """Run every enabled post-processing pass in the standard order."""
    if insert_merge_nodes:
        graph = _insert_merge_nodes(graph)
    if auto_collapse and not expand_groups:
        graph = _auto_collapse_large(graph)
    return graph


# ---------------------------------------------------------------------------
# Skip-connection ``+`` merge nodes (PRD §6.2)
# ---------------------------------------------------------------------------


def _insert_merge_nodes(graph: ModelGraph) -> ModelGraph:
    """Insert a small ``+`` merge node wherever a real node has in-degree > 1.

    We only insert when the extra edges are ``data`` edges — shared-weight
    and skip edges are decoration, not fan-in.
    """
    fanin: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        if edge.kind != "data":
            continue
        fanin.setdefault(edge.target_id, []).append(edge)

    to_merge = {tid: edges for tid, edges in fanin.items() if len(edges) > 1}
    if not to_merge:
        return graph

    new_nodes = list(graph.nodes)
    new_edges = [e for e in graph.edges if not (e.kind == "data" and e.target_id in to_merge)]
    # Preserve non-data edges to targets we're rewriting.
    kept_ids = {n.id for n in graph.nodes}
    for target_id, incoming in to_merge.items():
        if target_id not in kept_ids:
            new_edges.extend(incoming)
            continue
        merge_id = _unique(f"{target_id}._merge", kept_ids)
        kept_ids.add(merge_id)
        # Small diamond marker — the SVG renderer looks at ``layer_type``
        # to pick a shape hint.
        new_nodes.append(
            LayerNode(
                id=merge_id,
                name="+",
                layer_type="Merge",
                framework=graph.nodes[0].framework if graph.nodes else "unknown",
                style_override=StyleSpec(shape="diamond"),
                attributes={"merge_of": len(incoming)},
            )
        )
        # Redirect each incoming edge into the merge, then merge → original target.
        for e in incoming:
            new_edges.append(Edge(source_id=e.source_id, target_id=merge_id, label=e.label))
        new_edges.append(Edge(source_id=merge_id, target_id=target_id))

    # Preserve original node ordering; merges land at the end of the list
    # (position doesn't affect the layout — it re-ranks by topology).
    return ModelGraph(
        nodes=new_nodes,
        edges=new_edges,
        groups=graph.groups,
        metadata={**graph.metadata, "merge_nodes_inserted": len(to_merge)},
    )


# ---------------------------------------------------------------------------
# Large-model auto-collapse (PRD §6.4)
# ---------------------------------------------------------------------------


def _auto_collapse_large(graph: ModelGraph) -> ModelGraph:
    """If the graph exceeds :data:`LARGE_MODEL_THRESHOLD` nodes, fold groups.

    We replace every :class:`SegmentGroup` whose member count is ≥ 3 with a
    single collapsed node named after the group. Nodes not in any group are
    left untouched. Users get a console hint about ``expand_groups=True``.
    """
    if len(graph.nodes) <= LARGE_MODEL_THRESHOLD or not graph.groups:
        return graph

    mv_warn(
        f"Model has {len(graph.nodes)} nodes (> {LARGE_MODEL_THRESHOLD}) — "
        "collapsing groups. Pass expand_groups=True to render every node."
    )

    node_to_group: dict[str, str] = {}
    for group in graph.groups:
        if len(group.node_ids) < 3:
            continue
        for nid in group.node_ids:
            node_to_group[nid] = group.id

    if not node_to_group:
        return graph

    # Keep nodes not folded; add one collapsed placeholder per group.
    kept_nodes = [n for n in graph.nodes if n.id not in node_to_group]
    for group in graph.groups:
        if len(group.node_ids) < 3:
            continue
        member_types = [
            graph.get_node(nid).layer_type for nid in group.node_ids if graph.get_node(nid)
        ]
        common_type = _most_common(member_types) if member_types else "Group"
        kept_nodes.append(
            LayerNode(
                id=group.id,
                name=group.name,
                layer_type=f"Collapsed{common_type}",
                framework=graph.nodes[0].framework if graph.nodes else "unknown",
                params=sum(
                    (graph.get_node(nid).params or 0)
                    for nid in group.node_ids
                    if graph.get_node(nid)
                ),
                attributes={"collapsed_count": len(group.node_ids)},
                style_override=StyleSpec(shape="rounded_rect", dash="dashed"),
            )
        )

    # Rewire edges — anything crossing into or out of a folded group points
    # at the group's collapsed node instead.
    def resolve(nid: str) -> str:
        return node_to_group.get(nid, nid)

    seen: set[tuple[str, str, str]] = set()
    new_edges: list[Edge] = []
    for e in graph.edges:
        src, dst = resolve(e.source_id), resolve(e.target_id)
        if src == dst:
            continue  # collapsed intra-group edge
        key = (src, dst, e.kind)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(Edge(source_id=src, target_id=dst, label=e.label, kind=e.kind))

    return ModelGraph(
        nodes=kept_nodes,
        edges=new_edges,
        groups=[],  # groups are now nodes themselves
        metadata={**graph.metadata, "auto_collapsed": True},
    )


# ---------------------------------------------------------------------------
# Quantization + mixed-dtype badges (PRD §6.1)
# ---------------------------------------------------------------------------


def annotate_dtype_and_quantization(graph: ModelGraph, model: object) -> ModelGraph:
    """Add ``dtype`` and ``quantized`` badges to nodes when applicable.

    Inspectors call this after building the graph if they can supply the
    live module. It's a no-op when the framework doesn't expose the info.
    """
    # PyTorch: modules with ``weight_fake_quant`` are QAT layers; look up
    # module by graph node ID (matches the qualified dotted name from
    # ``_walk_modules``).
    named = {}
    try:
        for name, mod in getattr(model, "named_modules", lambda: [])():
            if name:
                named[name.replace(".", ".")] = mod
    except Exception:
        return graph

    dtypes: set[str] = set()
    for node in graph.nodes:
        module = named.get(node.id)
        if module is None:
            continue
        if hasattr(module, "weight_fake_quant") or hasattr(module, "activation_post_process"):
            node.attributes = {**node.attributes, "quantized": True}
        for param in getattr(module, "parameters", lambda recurse=False: [])(recurse=False):
            dtypes.add(str(getattr(param, "dtype", "")))
            break  # one is enough per module

    if len(dtypes) > 1:
        graph.metadata["mixed_precision"] = sorted(dtypes)
        # Annotate each node with its first-param dtype for the renderer.
        for node in graph.nodes:
            module = named.get(node.id)
            if module is None:
                continue
            for param in module.parameters(recurse=False):
                node.attributes = {**node.attributes, "dtype": str(param.dtype)}
                break
    return graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _most_common(items: list[str]) -> str:
    from collections import Counter

    return Counter(items).most_common(1)[0][0]


__all__ = [
    "LARGE_MODEL_THRESHOLD",
    "annotate_dtype_and_quantization",
    "post_process",
]
