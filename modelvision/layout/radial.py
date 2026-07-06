"""Radial layout — root in center, successors on concentric rings.

Useful for large branching models where a rectangular canvas becomes
awkward. Angle is allocated proportionally to subtree size so wide
subtrees don't crowd narrow ones.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from modelvision.layout import LaidOutGraph, NodeBox

if TYPE_CHECKING:
    from modelvision.core.ir import ModelGraph


NODE_WIDTH = 120.0
NODE_HEIGHT = 40.0
RING_GAP = 90.0
MARGIN = 60.0


def layout_radial(
    graph: ModelGraph,
    *,
    node_width: float = NODE_WIDTH,
    node_height: float = NODE_HEIGHT,
    ring_gap: float = RING_GAP,
    margin: float = MARGIN,
) -> LaidOutGraph:
    """Place ``graph`` nodes on concentric rings."""
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(n.id for n in graph.nodes)
    for e in graph.edges:
        if e.kind == "data":
            g.add_edge(e.source_id, e.target_id)

    node_ids = [n.id for n in graph.nodes]
    roots = [n for n in node_ids if g.in_degree(n) == 0] or node_ids[:1]
    # Synthetic super-root so we always get exactly one center.
    depth: dict[str, int] = {}
    for r in roots:
        depth[r] = 0
    frontier = list(roots)
    while frontier:
        node = frontier.pop(0)
        for succ in g.successors(node):
            if succ not in depth:
                depth[succ] = depth[node] + 1
                frontier.append(succ)
    for n in node_ids:
        depth.setdefault(n, 0)

    per_ring: dict[int, list[str]] = {}
    for n in graph.nodes:
        per_ring.setdefault(depth[n.id], []).append(n.id)

    max_ring = max(per_ring)
    outer_radius = max_ring * ring_gap
    size = 2 * (outer_radius + max(node_width, node_height) + margin)
    cx = cy = size / 2

    boxes: dict[str, NodeBox] = {}
    for ring, members in per_ring.items():
        if ring == 0:
            # Center — stack roots vertically at the middle.
            n = len(members)
            for i, node_id in enumerate(members):
                y = cy - (n - 1) * node_height / 2 + i * node_height
                boxes[node_id] = NodeBox(
                    node_id=node_id,
                    x=cx - node_width / 2,
                    y=y - node_height / 2,
                    width=node_width,
                    height=node_height,
                )
            continue
        radius = ring * ring_gap
        for i, node_id in enumerate(members):
            angle = 2 * math.pi * i / max(len(members), 1) - math.pi / 2
            x = cx + radius * math.cos(angle) - node_width / 2
            y = cy + radius * math.sin(angle) - node_height / 2
            boxes[node_id] = NodeBox(
                node_id=node_id, x=x, y=y, width=node_width, height=node_height
            )

    return LaidOutGraph(graph=graph, boxes=boxes, width=size, height=size)


__all__ = ["layout_radial"]
