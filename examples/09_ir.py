"""Example 09 — Working with the IR directly.

For advanced use cases — building a graph by hand, transforming an
existing one, or writing a custom renderer that consumes ModelVision's
IR without going through the standard pipeline.

Run::

    python examples/09_ir.py
"""

from __future__ import annotations

from modelvision.core.ir import Edge, LayerNode, ModelGraph, SegmentGroup
from modelvision.core.postprocess import post_process
from modelvision.core.style import StyleSpec
from modelvision.layout.vertical import layout_vertical
from modelvision.renderers.svg_renderer import render_svg
from modelvision.themes import get_theme


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Build a ModelGraph by hand — no framework at all.
    # ------------------------------------------------------------------
    graph = ModelGraph(
        nodes=[
            LayerNode(
                id="input",
                name="Input",
                layer_type="Input",
                framework="custom",
                output_shape=("B", 3, 224, 224),
            ),
            LayerNode(
                id="stem.conv",
                name="Conv7×7/2",
                layer_type="Conv2d",
                framework="custom",
                params=9408,
                output_shape=("B", 64, 112, 112),
            ),
            LayerNode(
                id="stem.bn",
                name="BatchNorm",
                layer_type="BatchNorm2d",
                framework="custom",
                params=128,
                output_shape=("B", 64, 112, 112),
            ),
            LayerNode(
                id="stem.act",
                name="ReLU",
                layer_type="ReLU",
                framework="custom",
                output_shape=("B", 64, 112, 112),
            ),
            LayerNode(
                id="stem.pool",
                name="MaxPool3×3/2",
                layer_type="MaxPool2d",
                framework="custom",
                output_shape=("B", 64, 56, 56),
            ),
            LayerNode(
                id="head",
                name="FC 1000",
                layer_type="Linear",
                framework="custom",
                params=64_000,
                output_shape=("B", 1000),
            ),
        ],
        edges=[
            Edge(source_id="input", target_id="stem.conv"),
            Edge(source_id="stem.conv", target_id="stem.bn"),
            Edge(source_id="stem.bn", target_id="stem.act"),
            Edge(source_id="stem.act", target_id="stem.pool"),
            Edge(source_id="stem.pool", target_id="head"),
        ],
        groups=[
            SegmentGroup(
                id="stem", name="Stem", node_ids=["stem.conv", "stem.bn", "stem.act", "stem.pool"]
            )
        ],
        metadata={"model_class": "CustomBackbone", "framework": "custom"},
    )

    # ------------------------------------------------------------------
    # 2. Transform the graph — attach a highlight style to specific nodes.
    # ------------------------------------------------------------------
    for node in graph.nodes:
        if node.layer_type == "Conv2d":
            node.style_override = StyleSpec(stroke="#dc2626", stroke_width=2.5, glow=True)

    # ------------------------------------------------------------------
    # 3. Run the same post-processing the public API uses.
    # ------------------------------------------------------------------
    graph = post_process(graph)

    # ------------------------------------------------------------------
    # 4. Serialize to JSON — the whole IR is a dict of primitives.
    # ------------------------------------------------------------------
    import json

    with open("09_graph.json", "w") as fh:
        json.dump(graph.to_dict(), fh, indent=2, default=str)
    print(f"wrote 09_graph.json: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # ------------------------------------------------------------------
    # 5. Render manually — skip the auto-detect and pipeline.
    # ------------------------------------------------------------------
    laid_out = layout_vertical(graph)
    svg = render_svg(laid_out, theme=get_theme("dark"))
    with open("09_from_ir.svg", "w") as fh:
        fh.write(svg)
    print(f"wrote 09_from_ir.svg: {len(svg):,} chars")

    # ------------------------------------------------------------------
    # 6. Round-trip: hand-built graphs work with the public API too.
    #    ``inspect`` accepts anything, but you can also feed the IR
    #    straight into the renderer for full control.
    # ------------------------------------------------------------------
    print(f"Graph metadata: {graph.metadata}")


if __name__ == "__main__":
    main()
