"""SVG renderer.

Consumes a :class:`~modelvision.layout.LaidOutGraph` plus resolved
:class:`~modelvision.core.style.StyleSpec`\\s per node and emits an SVG
string. Deterministic output (fixed float precision, sorted iteration)
so golden-file regression tests are stable across CI runs.

The output includes ``data-node-id`` attributes on each node group so
the M4 HTML renderer can wire click-to-inspect handlers without a
separate serialization pass.
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

from modelvision.core.ir import Edge, LayerNode, ModelGraph
from modelvision.core.style import Group, StyleSpec, Theme, resolve_style
from modelvision.layout import LaidOutGraph, NodeBox

# Two decimal places is plenty for pixel-space rendering and dodges the
# cross-platform float stringification issue flagged in the plan.
_FMT = "{:.2f}"


def render_svg(
    laid_out: LaidOutGraph,
    *,
    theme: Theme,
    layer_palette: dict[str, Any] | None = None,
    groups: list[Group] | None = None,
    node_styles: dict[str, StyleSpec] | None = None,
    show_params: bool = True,
    show_shapes: bool = True,
    show_dtypes: bool = False,
    embed_fonts: bool = True,
    title: str | None = None,
    legend: bool = False,
    default_shape: str | None = None,
    flow_style: bool = False,
    opacity: float | None = None,
    shade_step: float | None = None,
) -> str:
    """Return the full SVG document for ``laid_out``.

    ``default_shape`` overrides the shape used for any node whose
    :class:`StyleSpec` doesn't set one — this is how the ``volumetric``
    render mode makes every node render as an isometric cuboid.

    ``flow_style=True`` switches to visualtorch's "flow" aesthetic:
    isometric blocks placed edge-to-edge with per-node extrusion depth
    (read from ``graph.metadata['flow_depths']``), no in-block labels,
    no segment containers, no arrows. Best paired with ``layout='flow'``.

    ``opacity`` (0.0-1.0 or 0-255) fades every rendered block. ``shade_step``
    (0.0-0.5) controls how much darker top/right faces are on 3D shapes —
    higher means more contrast. Defaults produce visualtorch-style shading.
    """
    graph = laid_out.graph
    width, height = laid_out.width, laid_out.height

    # Normalize color-map entries — accept either str hex OR
    # {"fill": ..., "outline": ...} dicts (visualtorch-compat).
    fill_palette, outline_palette = _split_palette(layer_palette)
    layer_palette = fill_palette  # only fill drives resolve_style
    # Normalize opacity to a 0.0-1.0 fraction; if the user gave 0-255 (visualtorch
    # convention) or 0-100, coerce sensibly.
    fill_opacity = _normalize_opacity(opacity)
    # shade_step defaults align with visualtorch's default (~10 out of 255).
    shade = shade_step if shade_step is not None else 0.15

    # Flow style needs extrusion headroom on top of the tallest block.
    if flow_style:
        max_depth = float(graph.metadata.get("flow_max_depth", 60))
        depths = graph.metadata.get("flow_depths", {})
        # Pre-scan for any captions so we can reserve top space for them.
        # Sources: (1) per-node NodeStyle.caption via resolve_style,
        # (2) SegmentGroup.name (each group gets one banner spanning its members).
        node_captions: dict[str, str] = {}
        for node in graph.nodes:
            style = resolve_style(
                node,
                theme=theme,
                layer_palette=layer_palette,
                groups=groups,
                node_styles=node_styles,
            )
            if style.caption:
                node_captions[node.id] = style.caption
        group_captions: list[tuple[str, list[str]]] = [
            (g.name, list(g.node_ids)) for g in graph.groups if g.name and g.node_ids
        ]

        caption_pad = 0.0
        if node_captions or group_captions:
            # Reserve a strip above the extrusion: two rows of ~14px text
            # plus a short tick below each caption.
            caption_pad = 40.0 if group_captions else 24.0

        # ``show_params/show_shapes`` translate into below-block labels
        # here (Conv2d / (1, 16, 32, 32) style, following the user's HTML
        # reference). We reserve enough room for rotated labels since
        # dense flow diagrams need -45° rotation to avoid overlap; the
        # extra bottom room is cheap and empty diagrams don't waste it
        # visibly.
        show_labels = show_params or show_shapes
        label_pad = 120.0 if show_labels else 0.0

        iso_pad = max_depth + 20.0 + caption_pad
        width += iso_pad + label_pad  # rotated labels at the last block overhang rightward
        height += iso_pad + label_pad
        parts: list[str] = []
        parts.append(_svg_header(width, height, title, theme, embed_fonts, iso_pad=iso_pad))
        parts.append(_defs(theme))
        parts.append(_background(width, height, theme.background, iso_pad=iso_pad))

        # Draw left-to-right. For each block, first draw the funnel from
        # the previous block to this one (so its far end lands under the
        # block's fill), then draw this block. That interleaving is what
        # visualtorch calls "the vanishing-point trick" — see
        # `visualtorch.flow._draw_funnels_and_boxes`.
        prev_box: NodeBox | None = None
        prev_depth: float = 0.0
        for node in graph.nodes:
            style = resolve_style(
                node,
                theme=theme,
                layer_palette=layer_palette,
                groups=groups,
                node_styles=node_styles,
            )
            fill = style.fill or theme.default_fill
            # Per-layer-type outline overrides theme's default stroke.
            stroke = outline_palette.get(node.layer_type, style.stroke or theme.default_stroke)
            stroke_w = style.stroke_width or theme.default_stroke_width
            box = laid_out.boxes[node.id]
            this_depth = depths.get(node.id, 0.0)

            if prev_box is not None:
                parts.append(_flow_funnel(prev_box, prev_depth, box, this_depth, stroke, stroke_w))
            parts.append(
                _flow_block(
                    box,
                    this_depth,
                    fill,
                    stroke,
                    stroke_w,
                    node,
                    variant=default_shape,
                    opacity=fill_opacity,
                    shade=shade,
                )
            )

            prev_box = box
            prev_depth = this_depth

        # Below-block labels — layer type + dimension, styled like the
        # user's HTML reference. Skipped when show_labels is False.
        if show_labels:
            # If blocks are dense (narrow face-widths), rotate labels
            # -45° so many labels can fit without overlapping.
            widths = [laid_out.boxes[n.id].width for n in graph.nodes if n.id in laid_out.boxes]
            median_w = sorted(widths)[len(widths) // 2] if widths else 60.0
            rotate_labels = median_w < 60.0
            # Rotated labels extend further down + to the right, so
            # bump the baseline up a bit so they don't clip off-canvas.
            baseline_y = laid_out.height + (-8 if rotate_labels else -4)
            for node in graph.nodes:
                if node.id not in laid_out.boxes:
                    continue
                box = laid_out.boxes[node.id]
                parts.append(
                    _flow_label_below(
                        box,
                        node,
                        theme,
                        baseline_y,
                        show_type=show_params,
                        show_shape=show_shapes,
                        rotate=rotate_labels,
                    )
                )

        # Captions render AFTER blocks so they sit on top and never clip.
        for node_id, text in node_captions.items():
            if node_id in laid_out.boxes:
                parts.append(
                    _flow_caption_node(
                        laid_out.boxes[node_id],
                        depths.get(node_id, 0.0),
                        max_depth,
                        text,
                        theme,
                    )
                )
        for name, node_ids in group_captions:
            placed = [laid_out.boxes[nid] for nid in node_ids if nid in laid_out.boxes]
            if placed:
                parts.append(_flow_caption_group(placed, depths, max_depth, name, theme))

        if legend:
            parts.append(
                _render_legend(
                    graph,
                    layer_palette,
                    theme,
                    canvas_width=laid_out.width,
                    canvas_height=laid_out.height,
                )
            )
        parts.append("</svg>\n")
        return "".join(parts)

    # Isometric extrusion pokes ~25% of the box's shorter side up-and-right
    # past its layout footprint. Expand the viewBox so those faces don't
    # clip against the canvas edge.
    iso_pad = 0.0
    if default_shape == "isometric":
        iso_pad = 30.0
        width += iso_pad
        height += iso_pad

    parts = []
    parts.append(_svg_header(width, height, title, theme, embed_fonts, iso_pad=iso_pad))
    parts.append(_defs(theme))
    parts.append(_background(width, height, theme.background, iso_pad=iso_pad))

    # Groups first (behind everything).
    parts.extend(_render_segments(graph, laid_out, theme))
    # Edges next (behind nodes).
    parts.extend(_render_edges(graph, laid_out, theme))
    # Nodes on top.
    for node in graph.nodes:
        style = resolve_style(
            node,
            theme=theme,
            layer_palette=layer_palette,
            groups=groups,
            node_styles=node_styles,
        )
        if default_shape and not style.shape:
            style = style.merge(StyleSpec(shape=default_shape))
        parts.append(
            _render_node(node, laid_out, style, theme, show_params, show_shapes, show_dtypes)
        )

    if legend:
        # Legend anchors to the layout footprint's bottom-right, not the
        # (possibly extended) viewBox — otherwise it drifts outside the
        # rendered canvas in ``volumetric`` mode.
        parts.append(
            _render_legend(
                graph,
                layer_palette,
                theme,
                canvas_width=laid_out.width,
                canvas_height=laid_out.height,
            )
        )

    parts.append("</svg>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Structural pieces
# ---------------------------------------------------------------------------


def _svg_header(
    width: float,
    height: float,
    title: str | None,
    theme: Theme,
    embed_fonts: bool,
    *,
    iso_pad: float = 0.0,
) -> str:
    t = f"<title>{escape(title)}</title>" if title else ""
    # ``embed_fonts=True`` uses a font-stack with common system fallbacks so
    # the SVG renders reasonably wherever it's opened. Base64-embedding the
    # actual TTF would multiply the file size 10× — we don't do that by
    # default; the font-family fallback chain covers the same intent.
    font = theme.font_family if embed_fonts else "system-ui, sans-serif"
    # In volumetric mode the viewBox is shifted up so extruded top faces
    # have room to render without clipping against the canvas edge.
    view_x = -iso_pad if iso_pad else 0
    view_y = -iso_pad if iso_pad else 0
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="{_FMT.format(view_x)} {_FMT.format(view_y)} '
        f'{_FMT.format(width)} {_FMT.format(height)}" '
        f'width="{_FMT.format(width)}" height="{_FMT.format(height)}" '
        f'font-family="{escape(font)}">'
        f"{t}"
    )


def _defs(theme: Theme) -> str:
    # Reusable arrowhead marker for edges. Deliberately minimal for M1;
    # per-shape ``<symbol>``s land in M2 when we add ``StyleSpec.shape``.
    return (
        "<defs>"
        f'<marker id="mv-arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 Z" fill="{theme.edge_color}"/>'
        f"</marker>"
        "</defs>"
    )


def _background(width: float, height: float, fill: str, iso_pad: float = 0.0) -> str:
    # Cover the full viewBox including the negative region used for isometric extrusion.
    x = -iso_pad
    y = -iso_pad
    w = width + iso_pad
    h = height + iso_pad
    return (
        f'<rect x="{_FMT.format(x)}" y="{_FMT.format(y)}" '
        f'width="{_FMT.format(w)}" height="{_FMT.format(h)}" '
        f'fill="{fill}"/>'
    )


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def _render_segments(graph: ModelGraph, laid_out: LaidOutGraph, theme: Theme) -> list[str]:
    out: list[str] = []
    # Reserve a taller top strip so the group label sits above the members
    # rather than overlapping the topmost node. 26px accommodates both the
    # label and any isometric-extrusion top faces that stick up past the
    # layout footprint of the first row.
    top_pad = 26.0
    side_pad = 10.0
    for seg in graph.groups:
        placed = [laid_out.boxes[nid] for nid in seg.node_ids if nid in laid_out.boxes]
        if not placed:
            continue
        min_x = min(b.x for b in placed) - side_pad
        min_y = min(b.y for b in placed) - top_pad
        max_x = max(b.x + b.width for b in placed) + side_pad
        max_y = max(b.y + b.height for b in placed) + side_pad
        out.append(
            f'<g data-group-id="{escape(seg.id)}">'
            f'<rect x="{_FMT.format(min_x)}" y="{_FMT.format(min_y)}" '
            f'width="{_FMT.format(max_x - min_x)}" '
            f'height="{_FMT.format(max_y - min_y)}" '
            f'rx="6" ry="6" '
            f'fill="{theme.group_fill}" stroke="{theme.group_stroke}" '
            f'stroke-width="1" stroke-dasharray="4 3" fill-opacity="0.6"/>'
            f'<text x="{_FMT.format(min_x + 6)}" y="{_FMT.format(min_y + 12)}" '
            f'fill="{theme.font_color}" font-size="10" font-weight="600">'
            f"{escape(seg.name)}</text>"
            "</g>"
        )
    return out


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def _render_edges(graph: ModelGraph, laid_out: LaidOutGraph, theme: Theme) -> list[str]:
    out: list[str] = []
    for e in graph.edges:
        if e.source_id not in laid_out.boxes or e.target_id not in laid_out.boxes:
            continue
        src = laid_out.boxes[e.source_id]
        dst = laid_out.boxes[e.target_id]
        out.append(_edge_path(src, dst, e, theme))
    return out


def _edge_path(src, dst, edge: Edge, theme: Theme) -> str:  # type: ignore[no-untyped-def]
    # Straight line from bottom of source to top of destination.
    x1, y1 = src.cx, src.y + src.height
    x2, y2 = dst.cx, dst.y
    dash = ' stroke-dasharray="5 4"' if edge.kind in {"shared", "skip"} else ""
    marker = ' marker-end="url(#mv-arrow)"' if edge.kind == "data" else ""
    label = ""
    if edge.label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        label = (
            f'<text x="{_FMT.format(mx)}" y="{_FMT.format(my)}" '
            f'fill="{theme.font_color}" font-size="9" '
            f'text-anchor="middle" dominant-baseline="middle">'
            f"{escape(edge.label)}</text>"
        )
    return (
        f'<g class="mv-edge" data-kind="{edge.kind}">'
        f'<line x1="{_FMT.format(x1)}" y1="{_FMT.format(y1)}" '
        f'x2="{_FMT.format(x2)}" y2="{_FMT.format(y2)}" '
        f'stroke="{theme.edge_color}" stroke-width="{theme.edge_width}"'
        f"{dash}{marker}/>"
        f"{label}</g>"
    )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def _render_legend(
    graph: ModelGraph,
    layer_palette: dict[str, str] | None,
    theme: Theme,
    canvas_width: float,
    canvas_height: float,
) -> str:
    """Draw a color-swatch legend anchored to the bottom-right corner.

    On very tall / very narrow layouts the legend can overlap the last
    node in the column. Users who don't want a legend can just pass
    ``legend=False`` (the default); when they do want one, we prefer
    a consistent corner placement over auto-placement magic that
    could still miss on unusual layouts.
    """
    palette = {**theme.layer_palette, **(layer_palette or {})}
    types_present = _ordered_unique(n.layer_type for n in graph.nodes)
    entries: list[tuple[str, str]] = []
    for lt in types_present:
        fill = palette.get(lt) or palette.get("*")
        if fill:
            entries.append((lt, fill))
    if not entries:
        return ""

    row_h = 18.0
    swatch = 12.0
    pad = 10.0
    label_gutter = 6.0
    text_w = max((len(lt) for lt, _ in entries), default=0) * 7.0
    box_w = pad + swatch + label_gutter + text_w + pad
    box_h = pad + row_h * len(entries) + pad / 2
    x = canvas_width - box_w - 12
    y = canvas_height - box_h - 12

    parts: list[str] = [
        '<g class="mv-legend" data-legend="true">',
        f'<rect x="{_FMT.format(x)}" y="{_FMT.format(y)}" '
        f'width="{_FMT.format(box_w)}" height="{_FMT.format(box_h)}" '
        f'rx="4" ry="4" '
        f'fill="{theme.background}" stroke="{theme.default_stroke}" '
        f'stroke-width="1" opacity="0.95"/>',
    ]
    for i, (label, fill) in enumerate(entries):
        row_y = y + pad + i * row_h
        parts.append(
            f'<rect x="{_FMT.format(x + pad)}" y="{_FMT.format(row_y)}" '
            f'width="{_FMT.format(swatch)}" height="{_FMT.format(swatch)}" '
            f'rx="2" ry="2" fill="{fill}" '
            f'stroke="{theme.default_stroke}" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{_FMT.format(x + pad + swatch + label_gutter)}" '
            f'y="{_FMT.format(row_y + swatch - 1)}" '
            f'fill="{theme.font_color}" font-size="11" '
            f'dominant-baseline="alphabetic">'
            f"{escape(label)}</text>"
        )
    parts.append("</g>")
    return "".join(parts)


def _ordered_unique(items) -> list:  # type: ignore[no-untyped-def]
    """Preserve first-seen order while dropping duplicates."""
    seen: set = set()
    out: list = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _render_node(
    node: LayerNode,
    laid_out: LaidOutGraph,
    style: StyleSpec,
    theme: Theme,
    show_params: bool,
    show_shapes: bool,
    show_dtypes: bool = False,
) -> str:
    box = laid_out.boxes[node.id]
    fill = style.fill or theme.default_fill
    stroke = style.stroke or theme.default_stroke
    stroke_width = style.stroke_width or theme.default_stroke_width
    label = style.label or f"{node.name}: {node.layer_type}"
    label_short, tooltip = _truncate(label, 32)

    subtitle_bits: list[str] = []
    if show_params and node.params:
        subtitle_bits.append(_human_params(node.params))
    if show_shapes and node.output_shape:
        subtitle_bits.append(f"→ {_fmt_shape(node.output_shape)}")
    if show_dtypes and node.attributes.get("dtype"):
        subtitle_bits.append(str(node.attributes["dtype"]).replace("torch.", ""))
    subtitle = " · ".join(subtitle_bits)

    # Badges — small text overlays for quantization and repeat counts.
    badge_str = ""
    if node.attributes.get("quantized"):
        badge_str += _badge(box, "Q", theme, offset=0)
    if node.attributes.get("repeat"):
        badge_str += _badge(box, f"× {node.attributes['repeat']}", theme, offset=20)

    shape_svg = _shape_element(box, style, fill, stroke, stroke_width)

    body: list[str] = [
        f'<g class="mv-node" data-node-id="{escape(node.id)}" '
        f'data-layer-type="{escape(node.layer_type)}">',
        f"<title>{escape(tooltip)}</title>",
        shape_svg,
    ]
    body.append(_node_label(box, label_short, style, theme, subtitle))
    body.append(badge_str)
    body.append("</g>")
    return "".join(body)


def _shape_element(
    box: NodeBox, style: StyleSpec, fill: str, stroke: str, stroke_width: float
) -> str:
    """Draw the node's outer shape.

    Supported shapes:

    - ``rect``, ``rounded_rect`` — flat rectangles (default).
    - ``diamond``, ``parallelogram`` — flat polygons.
    - ``cylinder`` — capped tube.
    - ``isometric`` — an extruded cuboid with front / top / right faces
      shaded darker, visualtorch-inspired ("volumetric" mode).
    - ``stacked`` — many offset flat slices, visualizing channel depth
      the way visualtorch's ``StackedBox`` does.

    Anything else falls back to a rounded rectangle.
    """
    shape = style.shape or "rounded_rect"
    dash = ' stroke-dasharray="5 4"' if style.dash == "dashed" else ""
    common = f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}'
    if shape == "isometric":
        return _isometric_element(box, fill, stroke, stroke_width, dash)
    if shape == "stacked":
        return _stacked_element(box, fill, stroke, stroke_width, dash)
    if shape == "diamond":
        cx, cy = box.cx, box.cy
        hx, hy = box.width / 2, box.height / 2
        pts = (
            f"{_FMT.format(cx)},{_FMT.format(cy - hy)} "
            f"{_FMT.format(cx + hx)},{_FMT.format(cy)} "
            f"{_FMT.format(cx)},{_FMT.format(cy + hy)} "
            f"{_FMT.format(cx - hx)},{_FMT.format(cy)}"
        )
        return f'<polygon points="{pts}" {common}/>'
    if shape == "cylinder":
        x, y, w, h = box.x, box.y, box.width, box.height
        rx = w / 2
        ry = h * 0.15
        return (
            f'<path d="M {_FMT.format(x)} {_FMT.format(y + ry)} '
            f"A {_FMT.format(rx)} {_FMT.format(ry)} 0 0 0 {_FMT.format(x + w)} {_FMT.format(y + ry)} "
            f"L {_FMT.format(x + w)} {_FMT.format(y + h - ry)} "
            f"A {_FMT.format(rx)} {_FMT.format(ry)} 0 0 1 {_FMT.format(x)} {_FMT.format(y + h - ry)} "
            f'Z" {common}/>'
        )
    if shape == "parallelogram":
        skew = box.height * 0.35
        pts = (
            f"{_FMT.format(box.x + skew)},{_FMT.format(box.y)} "
            f"{_FMT.format(box.x + box.width)},{_FMT.format(box.y)} "
            f"{_FMT.format(box.x + box.width - skew)},{_FMT.format(box.y + box.height)} "
            f"{_FMT.format(box.x)},{_FMT.format(box.y + box.height)}"
        )
        return f'<polygon points="{pts}" {common}/>'
    if shape == "rect":
        return (
            f'<rect x="{_FMT.format(box.x)}" y="{_FMT.format(box.y)}" '
            f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
            f"{common}/>"
        )
    # Default: rounded_rect. Honor ``border_radius`` if the user set it.
    radius = style.border_radius if style.border_radius is not None else 4
    return (
        f'<rect x="{_FMT.format(box.x)}" y="{_FMT.format(box.y)}" '
        f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
        f'rx="{_FMT.format(radius)}" ry="{_FMT.format(radius)}" {common}/>'
    )


# ---------------------------------------------------------------------------
# Volumetric shapes (visualtorch-inspired)
# ---------------------------------------------------------------------------


def _isometric_element(box: NodeBox, fill: str, stroke: str, stroke_width: float, dash: str) -> str:
    """Extruded 3D cuboid — front face + top parallelogram + right parallelogram.

    Depth is a fraction of the shorter side; the top and right faces are
    filled with faded shades of ``fill`` to imply lighting from the top-left
    (matches visualtorch's ``Box.draw`` two-shade rule).
    """
    depth = min(box.width, box.height) * 0.25
    x1, y1, x2, y2 = box.x, box.y, box.x + box.width, box.y + box.height
    # Top face — brighter fade.
    top_pts = (
        f"{_FMT.format(x1)},{_FMT.format(y1)} "
        f"{_FMT.format(x1 + depth)},{_FMT.format(y1 - depth)} "
        f"{_FMT.format(x2 + depth)},{_FMT.format(y1 - depth)} "
        f"{_FMT.format(x2)},{_FMT.format(y1)}"
    )
    # Right face — darker fade.
    right_pts = (
        f"{_FMT.format(x2)},{_FMT.format(y1)} "
        f"{_FMT.format(x2 + depth)},{_FMT.format(y1 - depth)} "
        f"{_FMT.format(x2 + depth)},{_FMT.format(y2 - depth)} "
        f"{_FMT.format(x2)},{_FMT.format(y2)}"
    )
    top_fill = _fade_hex(fill, -0.15)
    right_fill = _fade_hex(fill, -0.30)
    stroke_attr = f'stroke="{stroke}" stroke-width="{stroke_width}"{dash}'
    return (
        f'<polygon points="{top_pts}" fill="{top_fill}" {stroke_attr}/>'
        f'<polygon points="{right_pts}" fill="{right_fill}" {stroke_attr}/>'
        f'<rect x="{_FMT.format(x1)}" y="{_FMT.format(y1)}" '
        f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
        f'fill="{fill}" {stroke_attr}/>'
    )


def _stacked_element(box: NodeBox, fill: str, stroke: str, stroke_width: float, dash: str) -> str:
    """Multiple offset copies of the box, imitating channel depth.

    Matches visualtorch's ``StackedBox``: N thin rectangles offset by a
    small diagonal ``offset_z``, with alternating slightly-darker fill so
    the stack looks solid rather than a smear.
    """
    slices = 6
    offset = min(box.width, box.height) * 0.06
    total_offset = offset * (slices - 1)
    x1, y1 = box.x - total_offset / 2, box.y - total_offset / 2
    stroke_attr = f'stroke="{stroke}" stroke-width="{stroke_width}"{dash}'
    parts: list[str] = []
    for i in range(slices):
        dx = i * offset
        this_fill = fill if i % 2 == 0 else _fade_hex(fill, -0.15)
        parts.append(
            f'<rect x="{_FMT.format(x1 + dx)}" y="{_FMT.format(y1 + dx)}" '
            f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
            f'fill="{this_fill}" {stroke_attr}/>'
        )
    return "".join(parts)


def _flow_label_below(
    box: NodeBox,
    node: LayerNode,
    theme: Theme,
    baseline_y: float,
    *,
    show_type: bool,
    show_shape: bool,
    rotate: bool = False,
) -> str:
    """Render a single-line ``Conv2d (32, 8, 8)`` label below a flow block.

    Layer name and tensor shape share one line — name in bold, shape in a
    lighter gray following it. When ``rotate=True`` (dense timelines
    where labels would otherwise overlap), the whole label rotates 45°
    down-and-right from the block's bottom-center so each block gets
    its own readable caption without stepping on neighbours.
    """
    cx = box.x + box.width / 2
    display_name = _short_label_for(node) if show_type else ""
    shape_str = _short_shape_for(node) if show_shape else ""

    if not display_name and not shape_str:
        return ""

    short, _tooltip = _truncate(display_name, 20) if display_name else ("", "")

    if rotate:
        # Anchor at the block's own bottom-center + a small gap so the
        # label always starts BELOW the block and rotates away from it.
        anchor_x = cx
        anchor_y = box.y + box.height + 6
        transform = f"translate({_FMT.format(anchor_x)}, {_FMT.format(anchor_y)}) rotate(45)"
        return (
            f'<g class="mv-flow-label" data-node-id="{escape(node.id)}" '
            f'transform="{transform}">'
            f'<text x="4" y="0" text-anchor="start" dominant-baseline="middle" '
            f'font-size="10">'
            f'<tspan fill="{theme.font_color}" font-weight="700">'
            f"{escape(short)}</tspan>"
            f"{_shape_tspan(shape_str, theme)}"
            f"</text>"
            f"</g>"
        )

    # Non-rotated: one line centered under the block.
    return (
        f'<g class="mv-flow-label" data-node-id="{escape(node.id)}">'
        f'<text x="{_FMT.format(cx)}" y="{_FMT.format(baseline_y)}" '
        f'text-anchor="middle" font-size="11">'
        f'<tspan fill="{theme.font_color}" font-weight="700">'
        f"{escape(short)}</tspan>"
        f"{_shape_tspan(shape_str, theme)}"
        f"</text>"
        f"</g>"
    )


def _shape_tspan(shape_str: str, theme: Theme) -> str:
    """The trailing ``  (32, 8, 8)`` piece of a flow label, if we have one."""
    if not shape_str:
        return ""
    return (
        f'<tspan dx="6" fill="{theme.font_color}" opacity="0.65" '
        f'font-weight="400">'
        f"{escape(shape_str)}</tspan>"
    )


def _short_label_for(node: LayerNode) -> str:
    """Pick the most informative one-line label for a block.

    Prefers ``StyleSpec.label`` (user override) then a compact form of
    the layer type — e.g. ``Conv2d`` alone, without the framework-name
    prefix that would produce ``features.0: Conv2d``.
    """
    override = node.style_override
    if override and override.label:
        return override.label
    return node.layer_type


def _short_shape_for(node: LayerNode) -> str:
    """Render the output shape as a compact ``(C, H, W)`` tuple string.

    Batch dim is dropped so the number reads as "one tensor's shape."
    Prints as ``(32, 8, 8)`` — parenthesized, comma-separated — which
    matches how PyTorch/NumPy print shapes and reads naturally inline
    after the layer name.
    """
    shape = node.output_shape
    if not shape:
        return ""
    rest = list(shape[1:])
    if not rest:
        return ""
    parts = [str(d) if isinstance(d, int) else "?" for d in rest]
    return "(" + ", ".join(parts) + ")"


def _flow_caption_node(
    box: NodeBox,
    depth: float,
    max_depth: float,
    text: str,
    theme: Theme,
) -> str:
    """Render a single caption line above one block's back edge.

    Draws a tiny vertical tick from the caption baseline down to the top
    of the block's back-face (extrusion-shifted), so the reader knows
    which block the label points at.
    """
    top_back_x = box.x + box.width / 2 + depth / 2
    top_back_y = box.y - depth
    # Caption sits above the *maximum* extrusion so all captions line up.
    caption_y = -max_depth + 6
    return (
        f'<g class="mv-flow-caption" data-node-id="{escape(box.node_id)}">'
        f'<line x1="{_FMT.format(top_back_x)}" y1="{_FMT.format(caption_y + 4)}" '
        f'x2="{_FMT.format(top_back_x)}" y2="{_FMT.format(top_back_y)}" '
        f'stroke="{theme.font_color}" stroke-width="0.8" opacity="0.6"/>'
        f'<text x="{_FMT.format(top_back_x)}" y="{_FMT.format(caption_y)}" '
        f'fill="{theme.font_color}" font-size="11" font-weight="600" '
        f'text-anchor="middle">'
        f"{escape(text)}</text>"
        f"</g>"
    )


def _flow_caption_group(
    placed: list[NodeBox],
    depths: dict[str, float],
    max_depth: float,
    text: str,
    theme: Theme,
) -> str:
    """Render one caption spanning the leftmost-to-rightmost members of a group.

    Draws a horizontal bracket across the top-back edges of the members,
    with the caption centered above it.
    """
    left_box = min(placed, key=lambda b: b.x)
    right_box = max(placed, key=lambda b: b.x + b.width)
    left_depth = depths.get(left_box.node_id, 0.0)
    right_depth = depths.get(right_box.node_id, 0.0)

    x1 = left_box.x + left_depth
    x2 = right_box.x + right_box.width + right_depth
    bracket_y = -max_depth + 22
    caption_y = -max_depth + 12
    tick_len = 6.0

    return (
        f'<g class="mv-flow-caption mv-flow-group-caption">'
        # Horizontal bracket across the top of the group members.
        f'<line x1="{_FMT.format(x1)}" y1="{_FMT.format(bracket_y)}" '
        f'x2="{_FMT.format(x2)}" y2="{_FMT.format(bracket_y)}" '
        f'stroke="{theme.font_color}" stroke-width="0.8" opacity="0.7"/>'
        # Left tick.
        f'<line x1="{_FMT.format(x1)}" y1="{_FMT.format(bracket_y)}" '
        f'x2="{_FMT.format(x1)}" y2="{_FMT.format(bracket_y + tick_len)}" '
        f'stroke="{theme.font_color}" stroke-width="0.8" opacity="0.7"/>'
        # Right tick.
        f'<line x1="{_FMT.format(x2)}" y1="{_FMT.format(bracket_y)}" '
        f'x2="{_FMT.format(x2)}" y2="{_FMT.format(bracket_y + tick_len)}" '
        f'stroke="{theme.font_color}" stroke-width="0.8" opacity="0.7"/>'
        # Caption text centered over the bracket.
        f'<text x="{_FMT.format((x1 + x2) / 2)}" y="{_FMT.format(caption_y)}" '
        f'fill="{theme.font_color}" font-size="12" font-weight="700" '
        f'text-anchor="middle">'
        f"{escape(text)}</text>"
        f"</g>"
    )


def _flow_funnel(
    src: NodeBox,
    src_depth: float,
    dst: NodeBox,
    dst_depth: float,
    stroke: str,
    stroke_width: float,
) -> str:
    """Draw the tapered connector between two adjacent flow-view blocks.

    Ports visualtorch's ``flow._draw_funnel``: four straight lines joining
    the corresponding corners of the two blocks' front and back faces:

    - src top-back  → dst top-back
    - src bot-back  → dst bot-back
    - src top-front → dst top-front
    - src bot-front → dst bot-front

    When the two blocks have different sizes, these lines diverge and
    produce the characteristic vanishing-point taper.
    """
    x1a, y1a = src.x + src.width, src.y
    y2a = src.y + src.height
    x1b, y1b = dst.x, dst.y
    y2b = dst.y + dst.height

    pen = f'stroke="{stroke}" stroke-width="{stroke_width}"'
    lines = [
        # Back edges (offset up-right by each block's own de).
        (x1a + src_depth, y1a - src_depth, x1b + dst_depth, y1b - dst_depth),
        (x1a + src_depth, y2a - src_depth, x1b + dst_depth, y2b - dst_depth),
        # Front edges.
        (x1a, y1a, x1b, y1b),
        (x1a, y2a, x1b, y2b),
    ]
    return "".join(
        f'<line x1="{_FMT.format(x1)}" y1="{_FMT.format(y1)}" '
        f'x2="{_FMT.format(x2)}" y2="{_FMT.format(y2)}" {pen} fill="none"/>'
        for x1, y1, x2, y2 in lines
    )


def _flow_block(
    box: NodeBox,
    depth: float,
    fill: str,
    stroke: str,
    stroke_width: float,
    node: LayerNode,
    *,
    variant: str | None = None,
    opacity: float | None = None,
    shade: float = 0.15,
) -> str:
    """One flow-view block. Style depends on ``variant``:

    - ``None`` or ``"isometric"`` — a single 3D isometric cuboid whose
      extrusion depth comes from the layer (visualtorch's flow style).
    - ``"stacked"`` — multiple offset flat slices, imitating a stack of
      channels — the look from the user's HTML reference and
      visualtorch's ``StackedBox`` primitive.
    - ``"rect"`` / anything else — a flat rectangle.

    ``opacity`` (0.0-1.0) fades the block; ``shade`` (0.0-0.5) controls
    how much darker the top/right faces are relative to the front fill.
    A ``<title>`` tooltip preserves the layer identity on hover.
    """
    if variant == "stacked":
        return _flow_block_stacked(
            box,
            depth,
            fill,
            stroke,
            stroke_width,
            node,
            opacity=opacity,
            shade=shade,
        )

    opacity_attr = f' opacity="{_FMT.format(opacity)}"' if opacity is not None else ""

    if depth <= 0:
        # Linear/Flatten with no spatial info — render as a flat rect.
        return (
            f'<g class="mv-flow-node" data-node-id="{escape(node.id)}" '
            f'data-layer-type="{escape(node.layer_type)}"{opacity_attr}>'
            f"<title>{escape(node.name)}: {escape(node.layer_type)}</title>"
            f'<rect x="{_FMT.format(box.x)}" y="{_FMT.format(box.y)}" '
            f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            f"</g>"
        )

    x1, y1 = box.x, box.y
    x2, y2 = box.x + box.width, box.y + box.height
    # Isometric offset goes up-and-right by ``depth``.
    top_pts = (
        f"{_FMT.format(x1)},{_FMT.format(y1)} "
        f"{_FMT.format(x1 + depth)},{_FMT.format(y1 - depth)} "
        f"{_FMT.format(x2 + depth)},{_FMT.format(y1 - depth)} "
        f"{_FMT.format(x2)},{_FMT.format(y1)}"
    )
    right_pts = (
        f"{_FMT.format(x2)},{_FMT.format(y1)} "
        f"{_FMT.format(x2 + depth)},{_FMT.format(y1 - depth)} "
        f"{_FMT.format(x2 + depth)},{_FMT.format(y2 - depth)} "
        f"{_FMT.format(x2)},{_FMT.format(y2)}"
    )
    # ``shade`` — 0.15 by default (visualtorch's shade_step≈10/255 default).
    # 0.15 → top face 15% darker, right face 30% darker.
    top_fill = _fade_hex(fill, -shade)
    right_fill = _fade_hex(fill, -2 * shade)
    stroke_attr = f'stroke="{stroke}" stroke-width="{stroke_width}"'
    return (
        f'<g class="mv-flow-node" data-node-id="{escape(node.id)}" '
        f'data-layer-type="{escape(node.layer_type)}"{opacity_attr}>'
        f"<title>{escape(node.name)}: {escape(node.layer_type)}</title>"
        f'<polygon points="{top_pts}" fill="{top_fill}" {stroke_attr}/>'
        f'<polygon points="{right_pts}" fill="{right_fill}" {stroke_attr}/>'
        f'<rect x="{_FMT.format(x1)}" y="{_FMT.format(y1)}" '
        f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
        f'fill="{fill}" {stroke_attr}/>'
        f"</g>"
    )


def _flow_block_stacked(
    box: NodeBox,
    depth: float,
    fill: str,
    stroke: str,
    stroke_width: float,
    node: LayerNode,
    *,
    opacity: float | None = None,
    shade: float = 0.18,
) -> str:
    """Draw a flow block as a stack of offset slices.

    Slice count scales with the isometric ``depth`` param (which itself
    tracks the tensor's spatial size), so front-of-model layers get
    thicker-looking stacks and the FC tail collapses to a single slice.
    Alternating slices use a slightly darker fill so the stack looks
    solid rather than a smear.
    """
    opacity_attr = f' opacity="{_FMT.format(opacity)}"' if opacity is not None else ""

    # Slice count — anywhere from 1 (small layers, no room to stack) to
    # ~7 (biggest layer). Depth is at most ~max_xy/3 = ~666, so this maps
    # roughly to "one slice per 30px of extrusion".
    slice_count = 1 + max(0, int(depth / 30))
    slice_count = min(slice_count, 8)

    if slice_count == 1:
        # Small block: just a flat rect. Skip the stack effect entirely
        # so the FC tail doesn't get a fake "depth" it doesn't have.
        return (
            f'<g class="mv-flow-node" data-node-id="{escape(node.id)}" '
            f'data-layer-type="{escape(node.layer_type)}"{opacity_attr}>'
            f"<title>{escape(node.name)}: {escape(node.layer_type)}</title>"
            f'<rect x="{_FMT.format(box.x)}" y="{_FMT.format(box.y)}" '
            f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            f"</g>"
        )

    per_slice_off = min(box.height, box.width) * 0.14
    per_slice_off = min(per_slice_off, 12.0)
    darker = _fade_hex(fill, -shade)

    stroke_attr = f'stroke="{stroke}" stroke-width="{stroke_width}"'
    parts: list[str] = [
        f'<g class="mv-flow-node" data-node-id="{escape(node.id)}" '
        f'data-layer-type="{escape(node.layer_type)}"{opacity_attr}>',
        f"<title>{escape(node.name)}: {escape(node.layer_type)}</title>",
    ]
    for i in range(slice_count - 1, -1, -1):
        off = i * per_slice_off
        slice_fill = darker if i % 2 == 1 else fill
        parts.append(
            f'<rect x="{_FMT.format(box.x + off)}" y="{_FMT.format(box.y - off)}" '
            f'width="{_FMT.format(box.width)}" height="{_FMT.format(box.height)}" '
            f'fill="{slice_fill}" {stroke_attr}/>'
        )
    parts.append("</g>")
    return "".join(parts)


def _split_palette(
    layer_palette: dict[str, Any] | None,
) -> tuple[dict[str, str] | None, dict[str, str]]:
    """Split a mixed ``layer_palette`` into separate fill / outline dicts.

    Accepts entries in either form:
    - ``"Conv2d": "#e69f00"`` — legacy, sets fill only.
    - ``"Conv2d": {"fill": "#e69f00", "outline": "#333"}`` — visualtorch-style.

    Returns ``(fill_palette, outline_palette)``. ``outline_palette`` may
    be empty if no outlines were specified.
    """
    if not layer_palette:
        return None, {}
    fills: dict[str, str] = {}
    outlines: dict[str, str] = {}
    for layer_type, value in layer_palette.items():
        if isinstance(value, str):
            fills[layer_type] = value
        elif isinstance(value, dict):
            if "fill" in value:
                fills[layer_type] = value["fill"]
            if "outline" in value:
                outlines[layer_type] = value["outline"]
        else:
            # Unknown value type — skip silently rather than break rendering.
            continue
    return fills, outlines


def _normalize_opacity(opacity: float | None) -> float | None:
    """Accept opacity in 0-1, 0-100, or 0-255 form; return the 0-1 fraction.

    ``None`` means "no explicit opacity" — keep whatever fill was set.
    Values > 1 are interpreted as 0-255 to match visualtorch's convention,
    then 0-100 if that's implausibly small.
    """
    if opacity is None:
        return None
    if opacity <= 1.0:
        return max(0.0, min(1.0, float(opacity)))
    if opacity <= 100.0:
        return max(0.0, min(1.0, float(opacity) / 100.0))
    return max(0.0, min(1.0, float(opacity) / 255.0))


def _fade_hex(color: str, amount: float) -> str:
    """Lighten (positive) or darken (negative) a hex color by ``amount`` in [-1, 1]."""
    from modelvision.core.color import parse_hex

    r, g, b, a = parse_hex(color)
    if amount < 0:
        r = int(r * (1 + amount))
        g = int(g * (1 + amount))
        b = int(b * (1 + amount))
    else:
        r = int(r + (255 - r) * amount)
        g = int(g + (255 - g) * amount)
        b = int(b + (255 - b) * amount)
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _badge(box: NodeBox, text: str, theme: Theme, *, offset: float) -> str:
    """Small pill in the top-right corner of a node."""
    x = box.x + box.width - 10 - offset
    y = box.y + 4
    return (
        f'<g class="mv-badge">'
        f'<rect x="{_FMT.format(x - 8)}" y="{_FMT.format(y)}" '
        f'width="18" height="12" rx="6" ry="6" '
        f'fill="{theme.default_stroke}" opacity="0.85"/>'
        f'<text x="{_FMT.format(x + 1)}" y="{_FMT.format(y + 9)}" '
        f'fill="{theme.background}" font-size="8" font-weight="700" '
        f'text-anchor="middle">'
        f"{escape(text)}</text>"
        f"</g>"
    )


def _fmt_shape(shape: tuple[Any, ...]) -> str:
    """Render a shape tuple: numeric dims as-is, symbolic dims (str) verbatim."""
    return "(" + ", ".join(str(d) if d is not None else "?" for d in shape) + ")"


def _node_label(box, label: str, style: StyleSpec, theme: Theme, subtitle: str) -> str:  # type: ignore[no-untyped-def]
    fc = style.font_color or theme.font_color
    fs = style.font_size or theme.font_size
    fw = style.font_weight or "600"
    if subtitle:
        return (
            f'<text x="{_FMT.format(box.cx)}" y="{_FMT.format(box.cy - 4)}" '
            f'fill="{fc}" font-size="{fs}" font-weight="{fw}" '
            f'text-anchor="middle" dominant-baseline="middle">'
            f"{escape(label)}</text>"
            f'<text x="{_FMT.format(box.cx)}" y="{_FMT.format(box.cy + 10)}" '
            f'fill="{fc}" font-size="{max(9, fs - 3)}" '
            f'text-anchor="middle" dominant-baseline="middle" opacity="0.75">'
            f"{escape(subtitle)}</text>"
        )
    return (
        f'<text x="{_FMT.format(box.cx)}" y="{_FMT.format(box.cy)}" '
        f'fill="{fc}" font-size="{fs}" font-weight="{fw}" '
        f'text-anchor="middle" dominant-baseline="middle">'
        f"{escape(label)}</text>"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _truncate(s: str, limit: int) -> tuple[str, str]:
    """Return ``(short, full)`` where ``short`` is ellipsized if needed."""
    if len(s) <= limit:
        return s, s
    return s[: limit - 1] + "…", s


def _human_params(n: int) -> str:
    """Render a param count in human-friendly units."""
    if n < 1_000:
        return f"{n} params"
    if n < 1_000_000:
        return f"{n / 1_000:.1f}K params"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M params"
    return f"{n / 1_000_000_000:.2f}B params"


__all__ = ["render_svg"]
