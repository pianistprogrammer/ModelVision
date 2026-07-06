"""Vertical (top-to-bottom) layered layout.

Uses :mod:`networkx` for the topological rank assignment and a simple
per-rank slot allocator for the x-coordinate. The renderer downstream
doesn't care about edge routing — the SVG renderer draws straight lines
between box centers, which is fine for a layered DAG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from modelvision.layout import LaidOutGraph, NodeBox

if TYPE_CHECKING:
    from modelvision.core.ir import ModelGraph


# Constants tuned for readable diagrams; renderer can override.
NODE_WIDTH = 160.0
NODE_HEIGHT = 44.0
H_GAP = 40.0
V_GAP = 28.0
MARGIN = 40.0


def layout_layered(
    graph: ModelGraph,
    *,
    axis: Literal["vertical", "horizontal"] = "vertical",
    node_width: float = NODE_WIDTH,
    node_height: float = NODE_HEIGHT,
    h_gap: float = H_GAP,
    v_gap: float = V_GAP,
    margin: float = MARGIN,
) -> LaidOutGraph:
    """Return a :class:`LaidOutGraph` with layered coordinates.

    ``axis="vertical"`` places rank 0 at the top and successors below;
    ``axis="horizontal"`` places rank 0 on the left and successors to
    the right. The two layouts share the same rank-assignment code.
    """
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(n.id for n in graph.nodes)
    # Only data edges influence the layout; shared-weight edges are
    # decoration and would otherwise create false cycles.
    for e in graph.edges:
        if e.kind == "data":
            g.add_edge(e.source_id, e.target_id)

    # Assign ranks. If the graph has cycles (RNNs), fall back to a
    # BFS-depth heuristic — networkx.topological_generations would raise.
    try:
        rank_of: dict[str, int] = {}
        for depth, layer in enumerate(nx.topological_generations(g)):
            for node_id in layer:
                rank_of[node_id] = depth
    except nx.NetworkXUnfeasible:
        rank_of = _bfs_rank(g, [n.id for n in graph.nodes])

    # Preserve input order among ties.
    per_rank: dict[int, list[str]] = {}
    for n in graph.nodes:
        per_rank.setdefault(rank_of.get(n.id, 0), []).append(n.id)

    max_rank = max(per_rank) if per_rank else 0
    max_width = max((len(row) for row in per_rank.values()), default=1)

    boxes: dict[str, NodeBox] = {}
    if axis == "vertical":
        canvas_cross = margin * 2 + max_width * node_width + (max_width - 1) * h_gap
        for rank in range(max_rank + 1):
            row = per_rank.get(rank, [])
            row_extent = len(row) * node_width + (len(row) - 1) * h_gap if row else 0
            offset = (canvas_cross - row_extent) / 2
            for i, node_id in enumerate(row):
                x = offset + i * (node_width + h_gap)
                y = margin + rank * (node_height + v_gap)
                boxes[node_id] = NodeBox(
                    node_id=node_id, x=x, y=y, width=node_width, height=node_height
                )
        canvas_main = margin * 2 + (max_rank + 1) * node_height + max_rank * v_gap
        return LaidOutGraph(graph=graph, boxes=boxes, width=canvas_cross, height=canvas_main)

    # Horizontal.
    canvas_cross = margin * 2 + max_width * node_height + (max_width - 1) * v_gap
    for rank in range(max_rank + 1):
        col = per_rank.get(rank, [])
        col_extent = len(col) * node_height + (len(col) - 1) * v_gap if col else 0
        offset = (canvas_cross - col_extent) / 2
        for i, node_id in enumerate(col):
            x = margin + rank * (node_width + h_gap)
            y = offset + i * (node_height + v_gap)
            boxes[node_id] = NodeBox(
                node_id=node_id, x=x, y=y, width=node_width, height=node_height
            )
    canvas_main = margin * 2 + (max_rank + 1) * node_width + max_rank * h_gap
    return LaidOutGraph(graph=graph, boxes=boxes, width=canvas_main, height=canvas_cross)


def layout_vertical(graph: ModelGraph, **kwargs) -> LaidOutGraph:  # type: ignore[no-untyped-def]
    """Layered top-to-bottom layout — see :func:`layout_layered`."""
    return layout_layered(graph, axis="vertical", **kwargs)


def _bfs_rank(g, node_ids: list[str]) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """BFS-from-roots depth. Used when the graph has cycles."""
    roots = [n for n in node_ids if g.in_degree(n) == 0] or node_ids[:1]
    depth: dict[str, int] = {r: 0 for r in roots}
    frontier = list(roots)
    while frontier:
        node = frontier.pop(0)
        for succ in g.successors(node):
            if succ not in depth:
                depth[succ] = depth[node] + 1
                frontier.append(succ)
    # Any unreached node (disconnected subgraph) sits at rank 0.
    for n in node_ids:
        depth.setdefault(n, 0)
    return depth


__all__ = ["layout_layered", "layout_vertical"]
