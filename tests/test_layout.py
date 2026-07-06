"""Tests for the vertical layout — coordinates, cycles, disconnection."""

from __future__ import annotations

from modelvision import Edge, LayerNode, ModelGraph
from modelvision.layout.vertical import layout_vertical


def _linear_graph(n: int) -> ModelGraph:
    nodes = [LayerNode(id=str(i), name=str(i), layer_type="X", framework="test") for i in range(n)]
    edges = [Edge(source_id=str(i), target_id=str(i + 1)) for i in range(n - 1)]
    return ModelGraph(nodes=nodes, edges=edges)


def test_linear_chain_ranks_increase() -> None:
    laid = layout_vertical(_linear_graph(5))
    ys = [laid.boxes[str(i)].y for i in range(5)]
    assert ys == sorted(ys)  # strictly increasing top-to-bottom
    assert laid.width > 0
    assert laid.height > 0


def test_cyclic_graph_does_not_hang() -> None:
    """RNN-style self-loop must fall back to the BFS-depth heuristic, not spin."""
    g = ModelGraph(
        nodes=[LayerNode(id=n, name=n, layer_type="X", framework="test") for n in ("a", "b", "c")],
        edges=[
            Edge(source_id="a", target_id="b"),
            Edge(source_id="b", target_id="c"),
            Edge(source_id="c", target_id="a"),  # cycle
        ],
    )
    laid = layout_vertical(g)
    # All three nodes are placed somewhere; the test is that it terminates.
    assert set(laid.boxes) == {"a", "b", "c"}


def test_disconnected_subgraph_still_places_every_node() -> None:
    g = ModelGraph(
        nodes=[LayerNode(id=n, name=n, layer_type="X", framework="test") for n in ("a", "b", "c")],
        edges=[Edge(source_id="a", target_id="b")],  # 'c' is disconnected
    )
    laid = layout_vertical(g)
    assert set(laid.boxes) == {"a", "b", "c"}


def test_shared_edges_do_not_break_topo_sort() -> None:
    """A shared-weight back-edge must not turn a valid DAG into a cyclic graph."""
    g = ModelGraph(
        nodes=[LayerNode(id=n, name=n, layer_type="X", framework="test") for n in ("a", "b", "c")],
        edges=[
            Edge(source_id="a", target_id="b"),
            Edge(source_id="b", target_id="c"),
            Edge(source_id="c", target_id="a", kind="shared"),  # would cycle if counted
        ],
    )
    laid = layout_vertical(g)
    assert laid.boxes["a"].y < laid.boxes["b"].y < laid.boxes["c"].y
