"""Internal implementations of the public :mod:`modelvision` entry points.

Kept out of ``__init__.py`` so top-level imports stay fast: this module
and its transitive imports (renderers, layouts, inspectors) are only
loaded the first time a user actually calls :func:`render` or
:func:`inspect`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from modelvision.core.exceptions import InspectionError, RenderError, mv_warn
from modelvision.core.ir import ModelGraph
from modelvision.core.postprocess import post_process
from modelvision.core.style import Group, NodeStyle, Theme
from modelvision.core.validation import (
    AccessibilityMode,
    apply_accessibility,
    validate_groups,
    validate_node_styles,
)
from modelvision.inspectors.auto import detect_framework
from modelvision.themes import get_theme

# 10 MB — beyond this we emit a size warning per PRD §6.4.
_LARGE_SVG_BYTES = 10 * 1024 * 1024


def inspect(model: Any, *, framework: str | None = None, **kwargs: Any) -> ModelGraph:
    """Dispatch to the matching inspector and return a :class:`ModelGraph`."""
    from typing import cast

    fw = framework or detect_framework(model)
    inspector = _get_inspector(fw)
    return cast(ModelGraph, inspector.inspect(model, **kwargs))


def render(
    model: Any,
    *,
    output: str | os.PathLike[str] | None = None,
    framework: str | None = None,
    theme: str | Theme = "light",
    layer_palette: dict[str, str | dict[str, str]] | None = None,
    palette: str | list[str] | None = None,
    node_styles: dict[str, NodeStyle] | None = None,
    groups: list[Group] | None = None,
    layout: str = "vertical",
    show_params: bool = True,
    show_shapes: bool = True,
    show_dtypes: bool = False,
    overwrite: bool = True,
    title: str | None = None,
    symbolic_shapes: bool = False,
    show_shared_weights: bool = True,
    input_shape: tuple[int, ...] | None = None,
    strict: bool = True,
    accessibility_check: AccessibilityMode = False,
    expand_groups: bool = False,
    inline: bool = False,
    embed_fonts: bool = True,
    legend: bool = False,
    volumetric: bool = False,
    style_variant: str | None = None,
    size_by_shape: bool = False,
    type_ignore: list[str] | None = None,
    show_input: bool = True,
    opacity: float | None = None,
    shade_step: float | None = None,
    node_size: int | float | tuple[float, float] | None = None,
    layer_spacing: float | None = None,
    dpi: int = 300,
    width: int | None = None,
    height: int | None = None,
    **_: Any,
) -> Any:
    """Full render pipeline: detect → inspect → post-process → validate → style → layout → render.

    Returns ``None`` when writing to disk. Returns the SVG/HTML string
    when ``output`` is ``None`` and we're not in a notebook. Returns a
    :class:`PIL.Image.Image` when ``inline=True`` (Jupyter display).

    ``model`` may also be a pre-built :class:`~modelvision.core.ir.ModelGraph`
    — useful for custom / non-ML diagrams or when you want full control
    over nodes and edges without a framework model.

    - ``dpi`` — resolution for raster outputs (PNG, PDF). Default 300 —
      the standard for publications and print. Increase to 600 for A3
      posters, lower to 150 for quick previews.
    - ``width`` / ``height`` — explicit pixel dimensions for raster
      output. When set, override the SVG's natural viewBox size. The
      aspect ratio is preserved if only one is provided.

    New in v0.2 — visualtorch-inspired options:

    - ``palette`` — a named palette (``"okabe_ito"``, ``"tol_bright"``,
      ``"vivid"``, ``"pastel"``, ``"high_contrast"``, ``"grayscale"``) or
      an explicit color list. Builds a per-layer-type mapping automatically.
    - ``legend`` — draw a small color-swatch legend in the bottom corner.
    - ``volumetric`` — render every node as an isometric extruded cuboid.
      Equivalent to ``style_variant="volumetric"``.
    - ``style_variant`` — ``"flat"`` (default), ``"volumetric"``, or
      ``"stacked"`` (feature-map slices).
    - ``size_by_shape`` — scale each node's box proportional to its
      output tensor shape (channels drive width, spatial dims drive height).
    """
    # 1. Inspect — or accept a pre-built ModelGraph directly.
    if isinstance(model, ModelGraph):
        graph = model
    else:
        graph = inspect(
            model,
            framework=framework,
            symbolic_shapes=symbolic_shapes,
            show_shared_weights=show_shared_weights,
            input_shape=input_shape,
            expand_groups=expand_groups,
        )

    # 2. Post-process — merge nodes, auto-collapse, badges.
    graph = post_process(graph, expand_groups=expand_groups)

    # 2a. type_ignore — drop nodes by layer_type. Edges spanning a
    # dropped node get rewired so the ribbon stays connected.
    if type_ignore:
        graph = _filter_by_type(graph, set(type_ignore))

    # 2b. Shape propagation — the flow layout needs concrete output_shapes.
    # For PyTorch we always propagate (the inspector leaves them empty),
    # for ONNX / Keras / JAX we only propagate if the inspector didn't
    # already fill them in.
    if input_shape and layout == "flow":
        from modelvision.core.shape_prop import propagate_shapes

        needs_prop = any(n.output_shape is None for n in graph.nodes)
        if needs_prop:
            graph = propagate_shapes(graph, input_shape)

    # 3. Validation (per PRD §6.3).
    validate_node_styles(graph, node_styles)
    validate_groups(graph, groups, strict=strict)

    # 4. Resolve theme + accessibility adjustment.
    theme_obj = get_theme(theme)

    # If the user passed ``palette=``, build a layer_palette from it and
    # merge with any explicit ``layer_palette`` (explicit entries win).
    if palette is not None:
        from modelvision.core.palettes import build_layer_palette

        auto = build_layer_palette(palette)
        layer_palette = {**auto, **(layer_palette or {})}

    node_styles = apply_accessibility(
        graph,
        mode=accessibility_check,
        theme=theme_obj,
        layer_palette=layer_palette,
        groups=groups,
        node_styles=node_styles,
    )

    # 5. Layout.
    laid_out = _apply_layout(graph, layout, node_size=node_size, layer_spacing=layer_spacing)
    if size_by_shape:
        from modelvision.layout.shape_size import resize_by_shape

        laid_out = resize_by_shape(laid_out)

    # ``volumetric=True`` is shorthand for style_variant="volumetric".
    if volumetric and style_variant is None:
        style_variant = "volumetric"
    default_shape = _default_shape_for_variant(style_variant)
    # ``layout='flow'`` sets ``flow_depths`` metadata on the graph — that's
    # our signal to switch the renderer into flow_style mode.
    flow_style = bool(graph.metadata.get("flow_depths"))

    # 6. Render — dispatch by output extension (or ``inline``).
    if inline and output is None:
        # Notebook display — return a PIL Image.
        from modelvision.renderers.raster import svg_to_pil
        from modelvision.renderers.svg_renderer import render_svg

        svg = render_svg(
            laid_out,
            theme=theme_obj,
            layer_palette=layer_palette,
            groups=groups,
            node_styles=node_styles,
            show_params=show_params,
            show_shapes=show_shapes,
            show_dtypes=show_dtypes,
            embed_fonts=embed_fonts,
            title=title,
            legend=legend,
            default_shape=default_shape,
            flow_style=flow_style,
            opacity=opacity,
            shade_step=shade_step,
        )
        _warn_if_large(svg)
        return svg_to_pil(svg)

    fmt = _output_format(output)
    if fmt in {"svg", "png", "pdf"}:
        from modelvision.renderers.svg_renderer import render_svg

        svg = render_svg(
            laid_out,
            theme=theme_obj,
            layer_palette=layer_palette,
            groups=groups,
            node_styles=node_styles,
            show_params=show_params,
            show_shapes=show_shapes,
            show_dtypes=show_dtypes,
            embed_fonts=embed_fonts,
            title=title,
            legend=legend,
            default_shape=default_shape,
            flow_style=flow_style,
            opacity=opacity,
            shade_step=shade_step,
        )
        _warn_if_large(svg)
        if output is None:
            return svg
        if fmt == "svg":
            _write_text(output, svg, overwrite=overwrite)
        elif fmt == "png":
            from modelvision.renderers.raster import svg_to_png

            _prepare_output(output, overwrite=overwrite)
            svg_to_png(svg, output, dpi=dpi, width=width, height=height)
        elif fmt == "pdf":
            from modelvision.renderers.raster import svg_to_pdf

            _prepare_output(output, overwrite=overwrite)
            svg_to_pdf(svg, output)
        return None

    if fmt == "html":
        from modelvision.renderers.html_renderer import render_html

        html = render_html(
            laid_out,
            theme=theme_obj,
            layer_palette=layer_palette,
            groups=groups,
            node_styles=node_styles,
            show_params=show_params,
            show_shapes=show_shapes,
            show_dtypes=show_dtypes,
            embed_fonts=embed_fonts,
            title=title,
            legend=legend,
            default_shape=default_shape,
            flow_style=flow_style,
        )
        _warn_if_large(html)
        if output is None:
            return html
        _write_text(output, html, overwrite=overwrite)
        return None

    raise RenderError(f"Unsupported output format {fmt!r}. Supported: svg, png, pdf, html.")


def _filter_by_type(graph: ModelGraph, ignore: set[str]) -> ModelGraph:
    """Drop nodes whose ``layer_type`` is in ``ignore``, rewiring edges through them.

    For each dropped node D with predecessors ``{a, b}`` and successors
    ``{y, z}``, we synthesize edges ``a→y, a→z, b→y, b→z`` so the flow
    ribbon stays connected. Segment groups are updated to remove the
    dropped IDs; groups that end up empty are dropped too.
    """
    from modelvision.core.ir import Edge, ModelGraph, SegmentGroup

    dropped_ids = {n.id for n in graph.nodes if n.layer_type in ignore}
    if not dropped_ids:
        return graph

    kept_nodes = [n for n in graph.nodes if n.id not in dropped_ids]

    # Adjacency maps over data edges only — shared-weight edges are
    # decoration and don't participate in rewiring.
    preds: dict[str, list[str]] = {}
    succs: dict[str, list[str]] = {}
    for e in graph.edges:
        if e.kind != "data":
            continue
        succs.setdefault(e.source_id, []).append(e.target_id)
        preds.setdefault(e.target_id, []).append(e.source_id)

    # For each kept edge, decide if it needs rewiring past dropped nodes.
    def _resolve_targets(node_id: str) -> list[str]:
        """Walk forward past dropped nodes until we hit a kept target."""
        result: list[str] = []
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n not in dropped_ids:
                result.append(n)
                continue
            for succ in succs.get(n, []):
                stack.append(succ)
        return result

    new_edges: list[Edge] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for e in graph.edges:
        if e.source_id in dropped_ids:
            # This edge starts at a dropped node — we'll pick it up from
            # its predecessors' side.
            continue
        for target in _resolve_targets(e.target_id):
            key = (e.source_id, target, e.kind)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            new_edges.append(
                Edge(
                    source_id=e.source_id,
                    target_id=target,
                    label=e.label,
                    kind=e.kind,
                )
            )

    new_groups: list[SegmentGroup] = []
    for g in graph.groups:
        remaining = [nid for nid in g.node_ids if nid not in dropped_ids]
        if remaining:
            new_groups.append(
                SegmentGroup(
                    id=g.id,
                    name=g.name,
                    node_ids=remaining,
                    style_override=g.style_override,
                )
            )

    return ModelGraph(
        nodes=kept_nodes,
        edges=new_edges,
        groups=new_groups,
        metadata={**graph.metadata, "type_ignore": sorted(ignore)},
    )


def _default_shape_for_variant(variant: str | None) -> str | None:
    """Map ``style_variant`` to a default node shape."""
    if variant is None or variant == "flat":
        return None
    if variant == "volumetric":
        return "isometric"
    if variant == "stacked":
        return "stacked"
    raise RenderError(
        f"Unknown style_variant={variant!r}. Expected one of: flat, volumetric, stacked."
    )


def _warn_if_large(payload: str) -> None:
    size = len(payload.encode("utf-8"))
    if size > _LARGE_SVG_BYTES:
        mv_warn(
            f"Output is {size / 1024 / 1024:.1f} MB — large diagrams may be slow to open. "
            "Consider layout='horizontal' or expand_groups=False."
        )


def from_torch(model: Any, **kwargs: Any) -> Any:
    return render(model, framework="torch", **kwargs)


def from_keras(model: Any, **kwargs: Any) -> Any:
    return render(model, framework="keras", **kwargs)


def from_jax(module: Any, **kwargs: Any) -> Any:
    return render(module, framework="jax", **kwargs)


def from_huggingface(model_or_config: Any, **kwargs: Any) -> Any:
    return render(model_or_config, framework="huggingface", **kwargs)


def from_sklearn(pipeline: Any, **kwargs: Any) -> Any:
    return render(pipeline, framework="sklearn", **kwargs)


def from_onnx(path: str, **kwargs: Any) -> Any:
    return render(path, framework="onnx", **kwargs)


def from_gguf(path: str, **kwargs: Any) -> Any:
    """Visualize a ``.gguf`` file (llama.cpp / Ollama). Header-only, no weights loaded."""
    return render(path, framework="gguf", **kwargs)


# ---------------------------------------------------------------------------
# Layout dispatch
# ---------------------------------------------------------------------------


def _apply_layout(
    graph: ModelGraph,
    layout: str,
    *,
    node_size: int | float | tuple[float, float] | None = None,
    layer_spacing: float | None = None,
) -> Any:
    """Dispatch to the requested layout module.

    ``node_size`` overrides the per-node box dimension: pass a single
    number to set both width and height, or a ``(width, height)`` tuple
    for explicit control. ``layer_spacing`` sets the gap between layers
    (v_gap for vertical, h_gap for horizontal).
    """
    node_w, node_h = _resolve_node_size(node_size)
    kwargs: dict[str, Any] = {}
    if node_w is not None:
        kwargs["node_width"] = node_w
    if node_h is not None:
        kwargs["node_height"] = node_h

    if layout == "vertical":
        from modelvision.layout.vertical import layout_vertical

        if layer_spacing is not None:
            kwargs["v_gap"] = layer_spacing
        return layout_vertical(graph, **kwargs)
    if layout == "horizontal":
        from modelvision.layout.horizontal import layout_horizontal

        if layer_spacing is not None:
            kwargs["h_gap"] = layer_spacing
        return layout_horizontal(graph, **kwargs)
    if layout == "radial":
        from modelvision.layout.radial import layout_radial

        # Radial has its own node dimensions; only forward if given.
        radial_kwargs = {}
        if node_w is not None:
            radial_kwargs["node_width"] = node_w
        if node_h is not None:
            radial_kwargs["node_height"] = node_h
        return layout_radial(graph, **radial_kwargs)
    if layout == "hierarchical":
        from modelvision.layout.hierarchical import layout_hierarchical

        if layer_spacing is not None:
            kwargs["v_gap"] = layer_spacing
        return layout_hierarchical(graph, **kwargs)
    if layout == "flow":
        from modelvision.layout.flow import layout_flow

        flow_kwargs: dict[str, Any] = {}
        if layer_spacing is not None:
            flow_kwargs["spacing"] = layer_spacing
        return layout_flow(graph, **flow_kwargs)
    raise RenderError(f"Unknown layout {layout!r}.")


def _resolve_node_size(
    node_size: int | float | tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    """Split a ``node_size`` scalar or ``(w, h)`` pair into (width, height)."""
    if node_size is None:
        return None, None
    if isinstance(node_size, (int, float)):
        # visualtorch's ``node_size=100`` sets a per-neuron dot diameter in
        # graph mode; for our layered layouts we treat it as the box height
        # and scale width proportionally so labels still fit. If callers
        # want independent w/h they pass a tuple.
        h = float(node_size)
        w = max(120.0, h * 3.5)
        return w, h
    if isinstance(node_size, (tuple, list)) and len(node_size) == 2:
        return float(node_size[0]), float(node_size[1])
    raise RenderError(f"node_size must be a number or (width, height) tuple, got {node_size!r}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _output_format(output: str | os.PathLike[str] | None) -> str:
    if output is None:
        return "svg"
    ext = Path(os.fspath(output)).suffix.lower().lstrip(".")
    if not ext:
        raise RenderError(f"Cannot infer output format from path {output!r} (no extension).")
    return ext


def _prepare_output(path: str | os.PathLike[str], *, overwrite: bool) -> None:
    p = Path(os.fspath(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if not overwrite:
            raise FileExistsError(f"{p} exists and overwrite=False.")
        mv_warn(f"Overwriting existing file {p}.")


def _write_text(path: str | os.PathLike[str], content: str, *, overwrite: bool) -> None:
    _prepare_output(path, overwrite=overwrite)
    Path(os.fspath(path)).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Inspector dispatch
# ---------------------------------------------------------------------------


_INSPECTORS: dict[str, str] = {
    "torch": "modelvision.inspectors.torch_inspector:PyTorchInspector",
    "keras": "modelvision.inspectors.keras_inspector:KerasInspector",
    "jax": "modelvision.inspectors.jax_inspector:JAXInspector",
    "huggingface": "modelvision.inspectors.huggingface_inspector:HuggingFaceInspector",
    "sklearn": "modelvision.inspectors.sklearn_inspector:SklearnInspector",
    "onnx": "modelvision.inspectors.onnx_inspector:ONNXInspector",
    "gguf": "modelvision.inspectors.gguf_inspector:GGUFInspector",
}


def _get_inspector(framework: str) -> Any:
    spec = _INSPECTORS.get(framework)
    if spec is None:
        raise InspectionError(f"No inspector registered for framework={framework!r}.")
    module_path, cls_name = spec.split(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, cls_name)()
