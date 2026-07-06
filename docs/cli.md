# CLI reference — `mvision`

The `mvision` command exposes ModelVision's full render pipeline through the shell,
structured so that both humans and LLM tool-callers can drive it without needing
to write Python. Every render option available to `modelvision.render()` has a
matching CLI flag.

Install via any framework extra (`torch` is the smallest that lets `.py` sources
work):

```bash
uv add "modelvision[torch]"
```

Then run `mvision --help` or `python -m modelvision --help`.

## Structure

```
mvision                                   # equivalent to `mvision render`
mvision render   MODEL_SOURCE [CLASS]     # produce a diagram
mvision inspect  MODEL_SOURCE [CLASS]     # summarize / dump IR
mvision list     {palettes|themes|        # enumerate enum values
                  layouts|frameworks}
```

The bareword form `mvision model.py MyNet -o out.svg` (no `render` verb) is kept
as a convenience for humans — the `render` command is invoked automatically when
the first positional arg is a file path.

## Model sources

| Extension | What it loads |
|---|---|
| `.py` | Python file — must specify a class name; instantiated with no args unless `--init-args` provides JSON kwargs. |
| `.onnx` | ONNX model — no class name needed. |
| `.keras`, `.h5` | Saved Keras model. |
| `.pt`, `.pth` | PyTorch pickle — must deserialize to an `nn.Module`. |

## `mvision render`

Every render option is a flag; the resolved kwargs are printed verbatim by `--dry-run`
so an LLM can sanity-check its arguments before committing.

### Model discovery

- `--framework auto|torch|keras|jax|huggingface|sklearn|onnx` (default `auto`).
- `--init-args '{"num_classes": 100}'` — JSON dict passed to the model constructor.
- `--input-shape 1x3x224x224` — required for `flow` layout and lazy-built Keras/JAX models.

### Layout & style

- `--layout vertical|horizontal|radial|hierarchical|flow` (default `vertical`).
- `--theme light|dark|pastel|grayscale|high_contrast` (default `light`).
- `--palette PALETTE_NAME` — pick a named palette (`mvision list palettes`).
- `--layer-palette "Conv2d=#4a90d9,Linear=#9b59b6"` — override per layer type.
- `--style-variant flat|volumetric|stacked` — flat (default), 3D extruded, or slice ribbon.
- `--volumetric` — shorthand for `--style-variant volumetric`.
- `--legend` — draw a per-layer-type color legend.
- `--size-by-shape` — scale each node's box proportional to its output tensor shape.

### Content

- `--no-params`, `--no-shapes` — hide the corresponding subtitle.
- `--show-dtypes` — draw dtype badges for mixed-precision models.
- `--expand-groups` — never fold repeated blocks (HuggingFace, `>500`-node models).
- `--symbolic-shapes` — attempt `torch.fx.symbolic_trace` for cross-scope edges.
- `--no-shared-weights` — hide dashed tied-weight edges.
- `--accessibility off|warn|enforce` — WCAG contrast check; `enforce` auto-adjusts font colors.

### Output

- `-o path.{svg,png,pdf,html}` — write to a file, format inferred from extension.
- `--stdout` — write SVG/HTML to stdout (mutually useful with `-o` absent).
- `--dry-run` — skip rendering, print resolved kwargs + graph summary as JSON.
- `--dpi N` — PNG resolution.
- `--overwrite/--no-overwrite` — file-exists behavior.
- `--title "My Model"` — embed a title in the diagram.

### Examples

```bash
# Simplest — auto-detect framework, default theme.
mvision render model.py MyNet -o diagram.svg

# Visualtorch-style 3D flow diagram of a VGG.
mvision render vgg.py VGG16 \
    -o vgg.svg \
    --layout flow \
    --input-shape 1x3x224x224 \
    --palette okabe_ito \
    --legend

# Interactive HTML with click-to-inspect + pan/zoom.
mvision render model.py MyNet -o diagram.html --theme dark --volumetric

# Dry-run to see what the render *would* do without spending time on it.
mvision render model.py MyNet --dry-run --theme dark --volumetric

# Pipe SVG through anything expecting stdin.
mvision render model.py MyNet --stdout | inkscape --pipe --export-type=pdf -o out.pdf
```

## `mvision inspect`

Structured access to the `ModelGraph` IR without rendering.

- Default output is a rich terminal table.
- `--json` dumps the graph as JSON to stdout or `-o FILE`.
- Accepts all the same model-loading flags as `render` (framework, init-args, input-shape).

```bash
# Rich table.
mvision inspect model.py MyNet

# JSON to stdout — pipe to jq for programmatic filtering.
mvision inspect model.onnx --json | jq '.nodes[] | .layer_type' | sort -u

# JSON to a file.
mvision inspect model.py MyNet --json -o graph.json
```

## `mvision list`

Enumerates the valid string values for enum flags. Every subcommand supports
`--json` so an LLM can consume the list programmatically.

```bash
mvision list palettes           # rich table with color swatches
mvision list palettes --json    # {"okabe_ito": ["#E69F00", ...], ...}
mvision list themes --json      # ["light", "dark", "pastel", ...]
mvision list layouts --json     # ["vertical", "horizontal", "radial", ...]
mvision list frameworks --json  # ["auto", "torch", "keras", ...]
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | `ModelVisionError` — bad model source, unknown class, unsupported framework. Human-readable message on stderr. |
| `2` | click usage error — unknown flag, bad enum value, or `render` invoked without `-o`/`--stdout`. |

## Notes for LLM tool-callers

- **Discover before invoking.** Call `mvision list palettes --json`, `mvision list layouts --json`, etc. once and cache the results. Invalid enum values fail loudly with exit code 2 before any expensive render.
- **Preview with `--dry-run`.** Cheaper than PDF, prints the same graph metadata as `inspect` plus every resolved kwarg. Use this to validate assumptions about node count / param total.
- **Use `--stdout` for pipelines.** Avoids allocating a scratch directory and lets you compose with other CLIs.
- **Don't guess `input_shape`.** For the `flow` layout, wrong input shapes silently produce degenerate layouts (min-height everywhere). Prefer `mvision inspect --json` first if you're unsure of the model's expected input.
