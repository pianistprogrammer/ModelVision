"""ModelVision CLI — the ``mvision`` command.

Structured as a click group with three subcommands so LLM tool-users can
discover the surface area with ``mvision --help`` and drill into each
verb::

    mvision render   MODEL_SOURCE [CLASS] [OPTIONS]   # produce a diagram
    mvision inspect  MODEL_SOURCE [CLASS] [OPTIONS]   # summarize / dump the IR
    mvision list     {palettes,themes,layouts}        # enumerate enum values

Bare-word invocation ``mvision model.py MyNet -o out.svg`` still works —
it's aliased to ``render`` for humans who don't want to type the verb.
Every flag corresponds 1:1 to a keyword on :func:`modelvision.render`, so
an LLM that can shell out has parity with an LLM that can code Python.

Every subcommand can also emit machine-readable output:

- ``--json`` on ``inspect`` prints a JSON dump of the :class:`ModelGraph`
  IR to stdout (or ``--output`` if given).
- ``--stdout`` on ``render`` writes the SVG/HTML to stdout instead of a
  file, so pipelines can consume it without touching the filesystem.
- ``--dry-run`` on either command reports node counts, param totals, and
  the resolved kwarg set as JSON — useful for sanity-checking before
  spending a big cairosvg PDF render.

Exit codes:
- ``0`` — success.
- ``1`` — a :class:`ModelVisionError` (bad model source, unknown class,
  unsupported framework, etc.) — the human-readable message goes to
  stderr.
- ``2`` — argparse/click usage error (unknown flag, bad enum value).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from modelvision._api import inspect as _inspect
from modelvision._api import render as _render
from modelvision.core.exceptions import InspectionError, ModelVisionError
from modelvision.core.palettes import PALETTES

# Canonical enum choices, exposed via ``mvision list`` for LLMs that
# want to enumerate options before calling.
_THEMES = ["light", "dark", "pastel", "grayscale", "high_contrast"]
_LAYOUTS = ["vertical", "horizontal", "radial", "hierarchical", "flow"]
_FRAMEWORKS = ["auto", "torch", "keras", "jax", "huggingface", "sklearn", "onnx", "gguf"]
_STYLE_VARIANTS = ["flat", "volumetric", "stacked"]


# ---------------------------------------------------------------------------
# Root group — provides the ``mvision <verb>`` verbs.
# ---------------------------------------------------------------------------


class _MvisionGroup(click.Group):
    """Group that falls back to the ``render`` subcommand when the first
    positional arg is a file path rather than a known verb.

    So ``mvision model.py MyNet -o out.svg`` still works even though it
    skips the explicit ``render`` verb — helpful for humans, orthogonal
    for LLMs.
    """

    def resolve_command(self, ctx, args):  # type: ignore[no-untyped-def]
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["render", *args]
        return super().resolve_command(ctx, args)


@click.group(cls=_MvisionGroup, name="mvision")
@click.version_option(package_name="modelvision")
def main() -> None:
    """Render neural network architecture diagrams from the command line.

    Try ``mvision --help``, ``mvision render --help``, or
    ``mvision list palettes`` to explore.
    """


# ---------------------------------------------------------------------------
# Shared option decorators. Keeping them in one place means ``render`` and
# ``inspect`` stay in sync when we add a new flag.
# ---------------------------------------------------------------------------


def _model_source_arguments(f):  # type: ignore[no-untyped-def]
    """Add ``MODEL_SOURCE`` and optional ``CLASS_NAME`` positionals."""
    f = click.argument("class_name", required=False)(f)
    f = click.argument(
        "model_source",
        type=click.Path(exists=True, dir_okay=False, path_type=str),
    )(f)
    return f


def _framework_options(f):  # type: ignore[no-untyped-def]
    f = click.option(
        "--framework",
        "-f",
        type=click.Choice(_FRAMEWORKS),
        default="auto",
        help="Framework hint. Auto-detects from module prefix or file extension.",
    )(f)
    f = click.option(
        "--init-args",
        "init_args_json",
        default=None,
        help="JSON dict passed as constructor kwargs, e.g. '{\"num_classes\": 100}'.",
    )(f)
    f = click.option(
        "--input-shape",
        default=None,
        help='Symbolic input shape, e.g. "1x3x224x224". Required for flow layout '
        "and for lazy-built Keras / JAX models.",
    )(f)
    return f


def _style_options(f):  # type: ignore[no-untyped-def]
    """Every flag that affects the visual output. Mirrors ``render()``."""
    f = click.option("--theme", "-t", default="light", type=click.Choice(_THEMES))(f)
    f = click.option(
        "--layout",
        type=click.Choice(_LAYOUTS),
        default="vertical",
        help="Diagram layout. 'flow' is visualtorch-style — pair with --input-shape.",
    )(f)
    f = click.option(
        "--palette",
        "palette_name",
        default=None,
        type=click.Choice(list(PALETTES)),
        help="Named color palette. Overridden by --layer-palette entries.",
    )(f)
    f = click.option(
        "--layer-palette",
        "layer_palette_raw",
        default=None,
        help='Comma-separated "LayerType=#hex" pairs, e.g. "Conv2d=#4a90d9,Linear=#9b59b6".',
    )(f)
    f = click.option(
        "--style-variant",
        type=click.Choice(_STYLE_VARIANTS),
        default=None,
        help="'flat' (default), 'volumetric' (3D extruded), or 'stacked' (slice ribbon).",
    )(f)
    f = click.option(
        "--volumetric", is_flag=True, help="Shorthand for --style-variant volumetric."
    )(f)
    f = click.option(
        "--legend", is_flag=True, help="Draw a per-layer-type color legend in the corner."
    )(f)
    f = click.option(
        "--size-by-shape",
        is_flag=True,
        help="Scale each node's box proportional to its output tensor shape.",
    )(f)
    f = click.option("--no-params", is_flag=True, help="Hide parameter counts.")(f)
    f = click.option("--no-shapes", is_flag=True, help="Hide tensor shapes.")(f)
    f = click.option("--show-dtypes", is_flag=True, help="Show per-layer dtype badges.")(f)
    f = click.option(
        "--expand-groups",
        is_flag=True,
        help="Never fold repeated blocks (HuggingFace, large models).",
    )(f)
    f = click.option(
        "--accessibility",
        type=click.Choice(["off", "warn", "enforce"]),
        default="off",
        help="WCAG contrast check: 'warn' emits warnings, 'enforce' auto-adjusts colors.",
    )(f)
    f = click.option(
        "--symbolic-shapes",
        is_flag=True,
        help="Attempt torch.fx.symbolic_trace for cross-scope edges (PyTorch only).",
    )(f)
    f = click.option("--no-shared-weights", is_flag=True, help="Hide dashed shared-weight edges.")(
        f
    )
    f = click.option(
        "--strict/--no-strict", default=True, help="Raise on group overlaps + unknown node IDs."
    )(f)
    return f


def _output_options(f):  # type: ignore[no-untyped-def]
    f = click.option(
        "--output",
        "-o",
        type=click.Path(path_type=str),
        default=None,
        help="Output path. Extension picks format (.svg/.png/.pdf/.html). "
        "Omit + use --stdout to print to stdout.",
    )(f)
    f = click.option(
        "--stdout",
        "to_stdout",
        is_flag=True,
        help="Write SVG/HTML to stdout instead of a file (no --output needed).",
    )(f)
    f = click.option(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs (PNG/PDF). Default 300 (publication).",
    )(f)
    f = click.option(
        "--width",
        "img_width",
        type=int,
        default=None,
        help="Output pixel width. Preserves aspect ratio if only --width given.",
    )(f)
    f = click.option(
        "--height",
        "img_height",
        type=int,
        default=None,
        help="Output pixel height. Preserves aspect ratio if only --height given.",
    )(f)
    f = click.option(
        "--node-size",
        type=float,
        default=None,
        help="Per-node box size (visualtorch-compat). Number = height, "
        "width auto-scaled to fit labels.",
    )(f)
    f = click.option(
        "--layer-spacing",
        type=float,
        default=None,
        help="Gap between adjacent layers in the active layout.",
    )(f)
    f = click.option(
        "--overwrite/--no-overwrite", default=True, help="Overwrite the output file if it exists."
    )(f)
    f = click.option("--title", default=None, help="Optional title embedded in the diagram.")(f)
    return f


# ---------------------------------------------------------------------------
# ``mvision render`` — the main verb.
# ---------------------------------------------------------------------------


@main.command()
@_model_source_arguments
@_framework_options
@_style_options
@_output_options
@click.option(
    "--dry-run",
    is_flag=True,
    help="Skip rendering — print the resolved kwargs + graph summary as JSON.",
)
def render(
    model_source: str,
    class_name: str | None,
    framework: str,
    init_args_json: str | None,
    input_shape: str | None,
    theme: str,
    layout: str,
    palette_name: str | None,
    layer_palette_raw: str | None,
    style_variant: str | None,
    volumetric: bool,
    legend: bool,
    size_by_shape: bool,
    no_params: bool,
    no_shapes: bool,
    show_dtypes: bool,
    expand_groups: bool,
    accessibility: str,
    symbolic_shapes: bool,
    no_shared_weights: bool,
    strict: bool,
    output: str | None,
    to_stdout: bool,
    dpi: int,
    img_width: int | None,
    img_height: int | None,
    node_size: float | None,
    layer_spacing: float | None,
    overwrite: bool,
    title: str | None,
    dry_run: bool,
) -> None:
    """Render a neural network to a diagram file.

    Examples::

        mvision render model.py MyNet -o diagram.svg
        mvision render model.py MyNet -o diagram.html --theme dark --volumetric --legend
        mvision render vgg.py VGG --layout flow --input-shape "1x3x224x224" \\
                       --palette okabe_ito -o vgg.svg
        mvision render model.py MyNet --stdout > diagram.svg
    """
    kwargs = _build_render_kwargs(
        framework=framework,
        input_shape=input_shape,
        theme=theme,
        layout=layout,
        palette_name=palette_name,
        layer_palette_raw=layer_palette_raw,
        style_variant=style_variant,
        volumetric=volumetric,
        legend=legend,
        size_by_shape=size_by_shape,
        no_params=no_params,
        no_shapes=no_shapes,
        show_dtypes=show_dtypes,
        expand_groups=expand_groups,
        accessibility=accessibility,
        symbolic_shapes=symbolic_shapes,
        no_shared_weights=no_shared_weights,
        strict=strict,
        dpi=dpi,
        img_width=img_width,
        img_height=img_height,
        node_size=node_size,
        layer_spacing=layer_spacing,
        title=title,
    )

    try:
        model = _load_model(model_source, class_name, framework, init_args_json)
    except ModelVisionError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if dry_run:
        _emit_dry_run(model, kwargs, model_source=model_source, class_name=class_name)
        return

    try:
        result = _render(model, output=None if to_stdout else output, **kwargs)
    except ModelVisionError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if to_stdout and result is not None:
        click.echo(result, nl=False)
    elif output:
        _write_or_overwrite(output, overwrite=overwrite, content=result)
        click.echo(f"Wrote {output}", err=True)
    else:
        # No --output and no --stdout — user probably wants a preview.
        click.echo(
            "Nothing written. Pass --output/-o FILE or --stdout to emit the diagram.",
            err=True,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# ``mvision inspect`` — print the IR without rendering.
# ---------------------------------------------------------------------------


@main.command()
@_model_source_arguments
@_framework_options
@click.option(
    "--json", "as_json", is_flag=True, help="Emit the ModelGraph as JSON on stdout (LLM-friendly)."
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=str),
    default=None,
    help="Write the JSON to a file instead of stdout.",
)
@click.option("--expand-groups", is_flag=True)
@click.option("--symbolic-shapes", is_flag=True)
def inspect(
    model_source: str,
    class_name: str | None,
    framework: str,
    init_args_json: str | None,
    input_shape: str | None,
    as_json: bool,
    output: str | None,
    expand_groups: bool,
    symbolic_shapes: bool,
) -> None:
    """Summarize a model — layer table by default, JSON via --json.

    Examples::

        mvision inspect model.py MyNet
        mvision inspect model.py MyNet --json > graph.json
        mvision inspect model.onnx --json | jq '.nodes[] | .layer_type' | sort -u
    """
    try:
        model = _load_model(model_source, class_name, framework, init_args_json)
    except ModelVisionError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    shape = _parse_shape(input_shape) if input_shape else None
    fw = None if framework == "auto" else framework
    try:
        graph = _inspect(
            model,
            framework=fw,
            input_shape=shape,
            expand_groups=expand_groups,
            symbolic_shapes=symbolic_shapes,
        )
    except ModelVisionError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        payload = json.dumps(graph.to_dict(), indent=2, default=str)
        if output:
            Path(output).write_text(payload, encoding="utf-8")
            click.echo(f"Wrote {output}", err=True)
        else:
            click.echo(payload)
        return

    _print_summary_table(graph)


# ---------------------------------------------------------------------------
# ``mvision list`` — enumerate valid enum values. LLMs love these.
# ---------------------------------------------------------------------------


@main.group(name="list")
def list_() -> None:
    """Enumerate valid choices for other flags."""


@list_.command("palettes")
@click.option("--json", "as_json", is_flag=True)
def list_palettes(as_json: bool) -> None:
    """List built-in color palettes."""
    if as_json:
        click.echo(json.dumps({name: colors for name, colors in PALETTES.items()}, indent=2))
        return
    console = Console()
    table = Table(title="Built-in palettes")
    table.add_column("name", style="cyan")
    table.add_column("colors", style="dim")
    for name, colors in PALETTES.items():
        table.add_row(name, "  ".join(colors))
    console.print(table)


@list_.command("themes")
@click.option("--json", "as_json", is_flag=True)
def list_themes(as_json: bool) -> None:
    """List built-in themes."""
    if as_json:
        click.echo(json.dumps(_THEMES))
        return
    for name in _THEMES:
        click.echo(name)


@list_.command("layouts")
@click.option("--json", "as_json", is_flag=True)
def list_layouts(as_json: bool) -> None:
    """List available diagram layouts."""
    if as_json:
        click.echo(json.dumps(_LAYOUTS))
        return
    for name in _LAYOUTS:
        click.echo(name)


@list_.command("frameworks")
@click.option("--json", "as_json", is_flag=True)
def list_frameworks(as_json: bool) -> None:
    """List frameworks the inspector can handle."""
    if as_json:
        click.echo(json.dumps(_FRAMEWORKS))
        return
    for name in _FRAMEWORKS:
        click.echo(name)


# ---------------------------------------------------------------------------
# Helpers — model loading + argument massaging.
# ---------------------------------------------------------------------------


def _build_render_kwargs(**raw: Any) -> dict[str, Any]:
    """Translate flag names to their :func:`modelvision.render` kwarg equivalents."""
    fw = None if raw["framework"] == "auto" else raw["framework"]
    shape = _parse_shape(raw["input_shape"]) if raw["input_shape"] else None
    accessibility_mode: Any = False
    if raw["accessibility"] == "warn":
        accessibility_mode = True
    elif raw["accessibility"] == "enforce":
        accessibility_mode = "enforce"
    return {
        "framework": fw,
        "theme": raw["theme"],
        "layout": raw["layout"],
        "palette": raw["palette_name"],
        "layer_palette": _parse_palette(raw["layer_palette_raw"]),
        "style_variant": raw["style_variant"],
        "volumetric": raw["volumetric"],
        "legend": raw["legend"],
        "size_by_shape": raw["size_by_shape"],
        "show_params": not raw["no_params"],
        "show_shapes": not raw["no_shapes"],
        "show_dtypes": raw["show_dtypes"],
        "expand_groups": raw["expand_groups"],
        "accessibility_check": accessibility_mode,
        "symbolic_shapes": raw["symbolic_shapes"],
        "show_shared_weights": not raw["no_shared_weights"],
        "strict": raw["strict"],
        "input_shape": shape,
        "dpi": raw["dpi"],
        "width": raw.get("img_width"),
        "height": raw.get("img_height"),
        "node_size": raw.get("node_size"),
        "layer_spacing": raw.get("layer_spacing"),
        "title": raw["title"],
    }


def _load_model(
    source: str, class_name: str | None, framework: str, init_args_json: str | None
) -> Any:
    """Load a model from a Python file, ONNX path, or Keras save file."""
    source_path = Path(source)
    suffix = source_path.suffix.lower()

    if suffix == ".onnx":
        return str(source_path)

    if suffix == ".gguf":
        return str(source_path)

    if suffix in {".keras", ".h5"}:
        try:
            import keras  # type: ignore
        except ImportError:
            from tensorflow import keras  # type: ignore
        return keras.models.load_model(source_path)

    if suffix == ".py":
        if class_name is None:
            raise InspectionError(
                "Loading from a .py source requires the model class name "
                "as the second positional argument."
            )
        module = _import_from_path(source_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise InspectionError(
                f"Class {class_name!r} not found in {source_path}. "
                f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
            )
        kwargs = json.loads(init_args_json) if init_args_json else {}
        return cls(**kwargs)

    if suffix in {".pt", ".pth"}:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise InspectionError("Loading .pt requires torch installed.") from exc
        obj = torch.load(source_path, map_location="cpu", weights_only=False)
        if not hasattr(obj, "named_modules"):
            raise InspectionError(
                f"{source_path} did not deserialize to an nn.Module. "
                "Use a Python file with the class definition instead."
            )
        return obj

    raise InspectionError(
        f"Unrecognized model source extension {suffix!r}. "
        "Supported: .py, .onnx, .gguf, .keras, .h5, .pt, .pth"
    )


def _import_from_path(path: Path) -> Any:
    """Import ``path`` as a module without adding it to ``sys.path`` permanently."""
    spec = importlib.util.spec_from_file_location(f"_mvision_user_{path.stem}", str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise InspectionError(f"Could not import {path} as a module.")
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent.resolve())
    added = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(parent)
    return module


def _parse_shape(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.replace(",", "x").split("x"))


def _parse_palette(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    pairs = [item.split("=") for item in raw.split(",") if item.strip()]
    return {k.strip(): v.strip() for k, v in pairs}


def _emit_dry_run(
    model: Any, kwargs: dict[str, Any], *, model_source: str, class_name: str | None
) -> None:
    """Print the resolved kwargs and a graph summary as JSON. No files written."""
    fw = kwargs.get("framework")
    shape = kwargs.get("input_shape")
    graph = _inspect(
        model,
        framework=fw,
        input_shape=shape,
        expand_groups=kwargs.get("expand_groups", False),
        symbolic_shapes=kwargs.get("symbolic_shapes", False),
    )
    report = {
        "model_source": model_source,
        "class_name": class_name,
        "resolved_kwargs": {k: _jsonable(v) for k, v in kwargs.items()},
        "graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "groups": len(graph.groups),
            "total_params": graph.metadata.get("total_params"),
            "framework": graph.metadata.get("framework"),
            "layer_types": _histogram(n.layer_type for n in graph.nodes),
        },
    }
    click.echo(json.dumps(report, indent=2, default=str))


def _jsonable(v: Any) -> Any:
    """Coerce values into something ``json.dumps`` won't choke on."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(v) for k, v in v.items()}
    return repr(v)


def _histogram(items) -> dict[str, int]:  # type: ignore[no-untyped-def]
    from collections import Counter

    return dict(Counter(items))


def _write_or_overwrite(path: str, *, overwrite: bool, content: Any) -> None:
    """When ``_render`` returned a string (SVG/HTML), it wasn't written for us."""
    if isinstance(content, str):
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} exists and overwrite=False.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _print_summary_table(graph: Any) -> None:
    console = Console()
    table = Table(
        title=f"{graph.metadata.get('model_class', 'model')} "
        f"({graph.metadata.get('framework', '?')})"
    )
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("type", style="magenta")
    table.add_column("params", justify="right")
    table.add_column("shape", style="dim")
    for n in graph.nodes:
        table.add_row(
            n.id,
            n.layer_type,
            _fmt_params(n.params),
            str(n.output_shape or ""),
        )
    console.print(table)
    total = graph.metadata.get("total_params")
    if total:
        console.print(f"[bold]Total params:[/bold] {total:,}")
    console.print(
        f"[bold]Nodes:[/bold] {len(graph.nodes)}  "
        f"[bold]Edges:[/bold] {len(graph.edges)}  "
        f"[bold]Groups:[/bold] {len(graph.groups)}"
    )


def _fmt_params(n: int | None) -> str:
    if not n:
        return ""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}K"
    return f"{n / 1_000_000:.2f}M"


if __name__ == "__main__":  # pragma: no cover
    main()
