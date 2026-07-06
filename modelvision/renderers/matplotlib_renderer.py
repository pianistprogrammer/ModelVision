"""Matplotlib renderer — draws the diagram into an :class:`matplotlib.axes.Axes`.

Useful for users who want to embed a ModelVision diagram directly in
an existing matplotlib figure (e.g. next to loss curves, or in a
saved PDF report).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelvision.core._optional import require
from modelvision.core.style import Group, NodeStyle, Theme, resolve_style

if TYPE_CHECKING:
    from modelvision.layout import LaidOutGraph


def render_matplotlib(
    laid_out: LaidOutGraph,
    *,
    theme: Theme,
    layer_palette: dict[str, str] | None = None,
    groups: list[Group] | None = None,
    node_styles: dict[str, NodeStyle] | None = None,
    ax: Any = None,
) -> Any:
    """Draw ``laid_out`` onto ``ax``. Returns the axes."""
    plt = require("matplotlib.pyplot")
    patches = require("matplotlib.patches")

    if ax is None:
        _, ax = plt.subplots(figsize=(laid_out.width / 72, laid_out.height / 72))

    ax.set_xlim(0, laid_out.width)
    ax.set_ylim(laid_out.height, 0)  # invert y so top is 0
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor(theme.background)

    # Edges first.
    for e in laid_out.graph.edges:
        if e.source_id not in laid_out.boxes or e.target_id not in laid_out.boxes:
            continue
        src, dst = laid_out.boxes[e.source_id], laid_out.boxes[e.target_id]
        ls = "--" if e.kind in {"shared", "skip"} else "-"
        ax.annotate(
            "",
            xy=(dst.cx, dst.y),
            xytext=(src.cx, src.y + src.height),
            arrowprops={"arrowstyle": "->" if e.kind == "data" else "-", "color": theme.edge_color, "linestyle": ls},
        )

    # Nodes.
    for node in laid_out.graph.nodes:
        style = resolve_style(node, theme=theme, layer_palette=layer_palette, groups=groups, node_styles=node_styles)
        box = laid_out.boxes[node.id]
        rect = patches.FancyBboxPatch(
            (box.x, box.y),
            box.width,
            box.height,
            boxstyle="round,pad=2,rounding_size=4",
            linewidth=style.stroke_width or 1,
            facecolor=style.fill or theme.default_fill,
            edgecolor=style.stroke or theme.default_stroke,
        )
        ax.add_patch(rect)
        ax.text(
            box.cx,
            box.cy,
            style.label or f"{node.name}: {node.layer_type}",
            color=style.font_color or theme.font_color,
            ha="center",
            va="center",
            fontsize=max(6, (style.font_size or theme.font_size) - 2),
        )
    return ax


__all__ = ["render_matplotlib"]
