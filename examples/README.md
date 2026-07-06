# Runnable examples

Every file in this folder is a self-contained script that produces one
or more output files in the current directory. Each one starts with a
docstring explaining what it does and which ``modelvision`` extras it
needs.

Run any example from the repo root with::

    uv run python examples/01_pytorch_basic.py

or, if you have `modelvision` installed globally::

    python examples/01_pytorch_basic.py

## Index

| # | Script | What it shows | Extras needed |
|---|---|---|---|
| 01 | `01_pytorch_basic.py` | The one-liner. Render a PyTorch CNN to SVG. | `torch` |
| 02 | `02_styling.py` | Every level of the 5-tier style resolver. Themes, palettes, groups, per-node overrides, shape variants, accessibility. | `torch` |
| 03 | `03_outputs.py` | SVG, HTML, PNG, PDF, inline PIL, matplotlib embed. | `torch`, `pdf`, `plot` |
| 04 | `04_layouts.py` | Vertical, horizontal, radial, hierarchical layouts side by side. | `torch` |
| 05 | `05_huggingface.py` | HuggingFace: config-only rendering and full model instances. | `huggingface`, `torch` |
| 06 | `06_sklearn.py` | Pipeline, ColumnTransformer, GridSearchCV. | `sklearn` |
| 07 | `07_onnx.py` | Universal ONNX path — no framework required at inspection time. | `onnx` |
| 08 | `08_tricky.py` | Weight tying, ModuleList/Dict, DataParallel, torch.compile, skip connections. | `torch` |
| 09 | `09_ir.py` | Build a `ModelGraph` by hand, transform it, render it directly. | (none) |
| 10 | `10_notebook.py` | `inline=True` for Jupyter/Colab. | `torch`, `pdf` |
| 11 | `11_cli.py` | Reference for the `mvision` command-line tool. | (docs only) |

## Setup

The recommended way to run these is with [uv](https://github.com/astral-sh/uv)::

    uv add "modelvision[all]"

or, for a single extra::

    uv add "modelvision[torch]"

If you're developing on the repo itself, from the checkout::

    uv sync --all-extras --extra dev
    uv run python examples/01_pytorch_basic.py
