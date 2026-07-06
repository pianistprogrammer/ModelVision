"""Tests for horizontal and radial layouts."""

from __future__ import annotations

from modelvision import Edge, LayerNode, ModelGraph
from modelvision.layout.horizontal import layout_horizontal
from modelvision.layout.radial import layout_radial


def _chain(n: int) -> ModelGraph:
    nodes = [LayerNode(id=str(i), name=str(i), layer_type="X", framework="test") for i in range(n)]
    edges = [Edge(source_id=str(i), target_id=str(i + 1)) for i in range(n - 1)]
    return ModelGraph(nodes=nodes, edges=edges)


def test_horizontal_places_ranks_left_to_right() -> None:
    laid = layout_horizontal(_chain(4))
    xs = [laid.boxes[str(i)].x for i in range(4)]
    assert xs == sorted(xs)
    # Y coordinates should be roughly the same for a linear chain.
    ys = [laid.boxes[str(i)].y for i in range(4)]
    assert max(ys) - min(ys) < 1e-6


def test_radial_places_root_at_center() -> None:
    laid = layout_radial(_chain(4))
    root = laid.boxes["0"]
    cx, cy = laid.width / 2, laid.height / 2
    # Root should sit near the center.
    assert abs(root.cx - cx) < 1
    assert abs(root.cy - cy) < 5
