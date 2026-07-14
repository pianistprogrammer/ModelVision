# CLI

```bash
mvision render model.py MyNet -o diagram.svg --theme dark
mvision inspect model.onnx --json | jq '.nodes[].layer_type' | sort -u
mvision list palettes --json
```

The CLI is a click group with three verbs — `render`, `inspect`, `list` —
plus a bareword alias (`mvision model.py MyNet -o diagram.svg` still works).
Every render option is a flag; the resolved kwargs are printed by `--dry-run`
so you (or an LLM tool-caller) can sanity-check arguments before spending
render time.

See the full CLI reference at [../cli.md](../cli.md) for every flag,
subcommand, and exit code.

## Common recipes

```bash
# Visualtorch-style 3D flow diagram.
mvision render vgg.py VGG16 -o vgg.svg \
    --layout flow --input-shape 1x3x224x224 \
    --palette pastel --legend

# Interactive HTML with pan/zoom + click-to-inspect.
mvision render model.py MyNet -o diagram.html --theme dark --volumetric

# Preview shape/params/kwargs without rendering.
mvision render model.py MyNet --dry-run --volumetric

# Emit SVG to stdout for pipelines.
mvision render model.py MyNet --stdout > diagram.svg
```

