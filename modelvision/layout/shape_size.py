"""Visualtorch-inspired shape-based box sizing.

When ``size_by_shape=True`` is passed to :func:`~modelvision.render`, each
node's box dimensions scale with the layer's ``output_shape``:

- **Depth (channels/features)** — the largest tensor dim (typically the
  channel count for conv layers or ``out_features`` for linear layers)
  scales the box width.
- **Spatial extent (H × W)** — the remaining spatial dims scale height.

This is the same intuition visualtorch uses to make a Conv layer's box
volume proportional to its tensor volume. It gives you an at-a-glance
feel for where a model spends most of its parameters vs. spatial resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelvision.layout import LaidOutGraph


# Bounds — never make a box smaller than this or larger than that. Prevents
# scale-invariance issues where a 1-node graph produces a 4000px-wide box.
MIN_WIDTH = 60.0
MAX_WIDTH = 320.0
MIN_HEIGHT = 32.0
MAX_HEIGHT = 120.0


def resize_by_shape(
    laid_out: LaidOutGraph,
    *,
    min_width: float = MIN_WIDTH,
    max_width: float = MAX_WIDTH,
    min_height: float = MIN_HEIGHT,
    max_height: float = MAX_HEIGHT,
) -> LaidOutGraph:
    """Return a new :class:`LaidOutGraph` with boxes sized proportionally to shape.

    The size mapping is per-graph — we scan all output_shapes, find the
    biggest channel dim and biggest spatial dim, and interpolate every
    other box's dimensions relative to those two extremes.
    """
    from modelvision.layout import LaidOutGraph, NodeBox

    channels = []
    spatials = []
    for node in laid_out.graph.nodes:
        c, s = _extract_dims(node.output_shape)
        if c is not None:
            channels.append(c)
        if s is not None:
            spatials.append(s)

    max_c = max(channels) if channels else 1
    max_s = max(spatials) if spatials else 1

    new_boxes: dict[str, NodeBox] = {}
    for node in laid_out.graph.nodes:
        old = laid_out.boxes.get(node.id)
        if old is None:
            continue
        c, s = _extract_dims(node.output_shape)
        # Width is driven by channel/feature depth; height by spatial extent.
        w = _lerp(min_width, max_width, (c or 0) / max_c)
        h = _lerp(min_height, max_height, (s or 0) / max_s)
        # Center the new box on the old center so the layout still looks right.
        new_boxes[node.id] = NodeBox(
            node_id=node.id,
            x=old.cx - w / 2,
            y=old.cy - h / 2,
            width=w,
            height=h,
        )

    return LaidOutGraph(
        graph=laid_out.graph,
        boxes=new_boxes,
        width=laid_out.width,
        height=laid_out.height,
    )


def _extract_dims(shape: tuple | None) -> tuple[int | None, int | None]:
    """Return ``(channel_dim, spatial_extent)`` from an output shape tuple.

    Assumes ``(B, C, H, W)`` for 4D, ``(B, C, L)`` for 3D, ``(B, F)`` for 2D,
    where ``B`` may be a symbolic string like ``"B"``.

    Symbolic dims are treated as unknown. Numeric batch dim is detected by
    length (the leftmost dim in a shape whose length is ≥ 2) and dropped.
    """
    if not shape:
        return None, None
    if len(shape) < 2:
        return None, None
    # Drop the leftmost dim as batch — regardless of whether it's symbolic
    # or a concrete int. This matches PyTorch/TF's ``(B, C, ...)``
    # convention.
    remaining = list(shape[1:])
    nums = [d for d in remaining if isinstance(d, int) and d > 0]
    if not nums:
        return None, None
    channels = nums[0]
    spatial: int | None = None
    if len(nums) >= 3:
        spatial = nums[-1] * nums[-2]
    elif len(nums) == 2:
        spatial = nums[-1]
    return channels, spatial


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * max(0.0, min(1.0, t))


__all__ = ["MAX_HEIGHT", "MAX_WIDTH", "MIN_HEIGHT", "MIN_WIDTH", "resize_by_shape"]
