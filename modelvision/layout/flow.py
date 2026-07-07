"""Flow layout — visualtorch's signature funneling ribbon.

Adjacent 3D isometric blocks placed edge-to-edge along a horizontal
centerline. Sizing follows visualtorch's exact algorithm (see
`visualtorch.flow._box_factory`), reproduced verbatim from
https://github.com/willyfh/visualtorch/blob/main/visualtorch/flow.py

Given a ``(B, C, H, W)`` tensor::

    x = clamp(shape[1] * scale_xy, min_xy, max_xy)   # H → face height
    y = clamp(shape[2] * scale_xy, min_xy, max_xy)   # W → alt. face dim
    z = clamp(shape[0] * shape[3+] * scale_z, min_z, max_z)  # C → face width along flow
    de = int(x / 3)                                   # isometric depth = H/3

Front-of-model blocks (large H) get the dramatic 3D pyramid look
because ``de`` is proportional to H — the visualtorch signature.

Blocks are vertically **centered on a horizontal axis** so the diagram
reads as a tensor being extruded forward through space rather than a
bar chart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelvision.layout import LaidOutGraph, NodeBox

if TYPE_CHECKING:
    from modelvision.core.ir import ModelGraph


# Defaults mirror visualtorch's ``flow_view`` signature, but we cap
# ``max_z`` more aggressively — visualtorch's default of 400 produces
# very long tails for a Linear(4096) FC head.
MIN_XY = 10.0
MAX_XY = 2000.0
MIN_Z = 10.0
MAX_Z = 400.0
MAX_Z_LOW_DIM = 120.0  # separate, tighter cap for 1D/2D "tail" tensors so
# the FC head doesn't dominate the canvas.
SCALE_XY = 1.0
SCALE_Z = 0.1
SIDE_MARGIN = 40.0
V_MARGIN = 40.0
SPACING = 10.0  # gap between adjacent blocks (visualtorch default)


def layout_flow(
    graph: ModelGraph,
    *,
    min_xy: float = MIN_XY,
    max_xy: float = MAX_XY,
    min_z: float = MIN_Z,
    max_z: float = MAX_Z,
    max_z_low_dim: float = MAX_Z_LOW_DIM,
    scale_xy: float = SCALE_XY,
    scale_z: float = SCALE_Z,
    v_margin: float = V_MARGIN,
    side_margin: float = SIDE_MARGIN,
    spacing: float = SPACING,
    low_dim_orientation: str = "z",
) -> LaidOutGraph:
    """Return a :class:`LaidOutGraph` in flow layout.

    Per-node isometric extrusion depth is stashed under
    ``graph.metadata["flow_depths"]`` so the SVG renderer can pick it up.

    ``max_z_low_dim`` caps the face-width for 1D/2D tail tensors
    (Flatten, Linear, Dropout, ReLU-on-vector) more tightly than the
    ``max_z`` used for conv-style ``(C, H, W)`` blocks. This keeps FC
    heads from producing a ribbon that runs off the canvas — the
    visualtorch default lets Linear(4096) render as a 400-px-wide block,
    which is longer than most models' entire conv stack.
    """
    ordered = list(graph.nodes)

    heights: dict[str, float] = {}
    widths: dict[str, float] = {}
    depths: dict[str, float] = {}

    for node in ordered:
        h, w, d = _size_from_shape(
            node.output_shape,
            scale_xy=scale_xy,
            min_xy=min_xy,
            max_xy=max_xy,
            scale_z=scale_z,
            min_z=min_z,
            max_z=max_z,
            max_z_low_dim=max_z_low_dim,
            low_dim_orientation=low_dim_orientation,
        )
        heights[node.id] = h
        widths[node.id] = w
        depths[node.id] = d

    actual_max_height = max(heights.values(), default=min_xy)
    actual_max_depth = max(depths.values(), default=0.0)

    # Centered on the horizontal axis, with extrusion room above.
    canvas_height = actual_max_height + actual_max_depth + 2 * v_margin
    center_y = v_margin + actual_max_depth + actual_max_height / 2

    cursor_x = side_margin + actual_max_depth
    boxes: dict[str, NodeBox] = {}
    for node in ordered:
        h = heights[node.id]
        w = widths[node.id]
        boxes[node.id] = NodeBox(
            node_id=node.id,
            x=cursor_x,
            y=center_y - h / 2,  # vertically centered on the axis
            width=w,
            height=h,
        )
        cursor_x += w + spacing  # visualtorch-style gap between blocks

    canvas_width = cursor_x + side_margin + actual_max_depth

    graph.metadata["flow_depths"] = dict(depths)
    graph.metadata["flow_max_depth"] = actual_max_depth
    graph.metadata["flow_center_y"] = center_y
    return LaidOutGraph(graph=graph, boxes=boxes, width=canvas_width, height=canvas_height)


# ---------------------------------------------------------------------------
# Sizing rule — replicated from visualtorch.flow._box_factory
# ---------------------------------------------------------------------------


def _size_from_shape(
    shape: tuple | None,
    *,
    scale_xy: float,
    min_xy: float,
    max_xy: float,
    scale_z: float,
    min_z: float,
    max_z: float,
    max_z_low_dim: float,
    low_dim_orientation: str,
) -> tuple[float, float, float]:
    """Return ``(face_height, face_width, extrusion_depth)`` for one layer.

    Drops the batch dim, expands 1D/2D shapes onto the chosen orientation
    axis, then pads to 4D and applies visualtorch's clamps.
    """
    if not shape or len(shape) < 2:
        return min_xy, min_z, 0.0

    rest = list(shape[1:])
    dims: list[int] = [d if isinstance(d, int) and d > 0 else 1 for d in rest]

    # Track whether this was originally a 1D/2D "tail" tensor. Those get
    # the tighter ``max_z_low_dim`` cap.
    is_low_dim = len(dims) in (1, 2)
    if is_low_dim:
        # visualtorch: shape = (1,)*cxyz.index(orientation) + (value,).
        # ``cxyz`` indices: c=0, x=1, y=2, z=3.
        idx = "cxyz".index(low_dim_orientation) if low_dim_orientation in "cxyz" else 3
        value = dims[-1]
        dims = [1] * idx + [value]

    while len(dims) < 4:
        dims.append(1)

    c, h, w, extra = dims[0], dims[1], dims[2], dims[3]

    # visualtorch's exact rule from _box_factory:
    x_source = h * scale_xy  # used only for extrusion depth
    y_source = w * scale_xy  # becomes face height (y2)
    z_source = c * extra * scale_z  # becomes face width along flow (x2)

    effective_max_z = max_z_low_dim if is_low_dim else max_z
    face_height = _clamp(y_source, min_xy, max_xy)
    face_width = _clamp(z_source, min_z, effective_max_z)
    extrusion = _clamp(x_source / 3.0, 0.0, max_xy / 3.0)
    return face_height, face_width, extrusion


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


__all__ = ["layout_flow"]
