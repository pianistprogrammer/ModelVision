"""Reproduces visualtorch's flow-example gallery using ModelVision.

Each snippet below corresponds to a page under
https://visualtorch.readthedocs.io/en/latest/usage_examples/flow/ —
we run the ModelVision equivalent and save a PNG so you can compare
side-by-side.

Run::

    python examples/14_visualtorch_gallery.py

Outputs land in ./outputs/vt_*.svg + .png.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch.nn as nn

import modelvision as mv

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def _cnn():  # type: ignore[no-untyped-def]
    """The same Sequential CNN visualtorch uses on most flow example pages."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


COMMON = dict(layout="flow", input_shape=(1, 3, 224, 224), overwrite=True)


# ---------------------------------------------------------------------------
# 1. 2D View — flat mode (visualtorch's draw_volume=False).
# ---------------------------------------------------------------------------

mv.render(_cnn(), OUT / "vt_01_2d_view.svg", style_variant="flat", **COMMON)


# ---------------------------------------------------------------------------
# 2. Basic Sequential — legend=True.
# ---------------------------------------------------------------------------

mv.render(_cnn(), OUT / "vt_02_basic_sequential.svg", legend=True, **COMMON)


# ---------------------------------------------------------------------------
# 3. Color Palettes — swap through named palettes.
# ---------------------------------------------------------------------------

for pal in ("okabe_ito", "tol_bright", "vivid", "pastel", "high_contrast"):
    mv.render(_cnn(), OUT / f"vt_03_palette_{pal}.svg", palette=pal, **COMMON)


# ---------------------------------------------------------------------------
# 4. Custom Color — dict of hex fills keyed by layer type.
# ---------------------------------------------------------------------------

color_map: dict = defaultdict(dict)
color_map["Conv2d"] = "#E69F00"
color_map["ReLU"] = "#56B4E9"
color_map["MaxPool2d"] = "#CC79A7"
color_map["Flatten"] = "#009E73"
color_map["Linear"] = "#0072B2"
mv.render(_cnn(), OUT / "vt_04_custom_color.svg", layer_palette=color_map, **COMMON)


# ---------------------------------------------------------------------------
# 5. Custom Opacity — visualtorch uses 0-255; ModelVision accepts either.
# ---------------------------------------------------------------------------

mv.render(_cnn(), OUT / "vt_05_opacity.svg", opacity=100, **COMMON)


# ---------------------------------------------------------------------------
# 6. Custom Shading — controls how much darker top/right faces are.
# ---------------------------------------------------------------------------

mv.render(_cnn(), OUT / "vt_06_shading.svg", shade_step=0.35, **COMMON)


# ---------------------------------------------------------------------------
# 7. Dark Background — theme swap + per-type outlines.
# ---------------------------------------------------------------------------

fluoro = {
    "Conv2d": {"fill": "#00F5FF", "outline": "#E0FFFF"},
    "ReLU": {"fill": "#FCEE09", "outline": "#FFFACD"},
    "MaxPool2d": {"fill": "#FF10F0", "outline": "#FFD1FA"},
    "Linear": {"fill": "#00F5FF", "outline": "#E0FFFF"},
    "Flatten": {"fill": "#FCEE09", "outline": "#FFFACD"},
}
mv.render(_cnn(), OUT / "vt_07_dark_background.svg", layer_palette=fluoro, theme="dark", **COMMON)


# ---------------------------------------------------------------------------
# 8. Ignore Layers — drop nn.ReLU + nn.Flatten from the diagram.
# ---------------------------------------------------------------------------

mv.render(_cnn(), OUT / "vt_08_ignore_layers.svg", type_ignore=["ReLU", "Flatten"], **COMMON)


# ---------------------------------------------------------------------------
# 9. Stacked variant — visualtorch's classic slice-stack look, plus flow.
# ---------------------------------------------------------------------------

mv.render(_cnn(), OUT / "vt_09_stacked.svg", style_variant="stacked", **COMMON)


# ---------------------------------------------------------------------------
# Everything together — the combo the user's HTML reference showed.
# ---------------------------------------------------------------------------

mv.render(
    _cnn(),
    OUT / "vt_10_everything.svg",
    style_variant="stacked",
    layer_palette={
        "Conv2d": {"fill": "#fff59d", "outline": "#fbc02d"},
        "ReLU": {"fill": "#fff59d", "outline": "#fbc02d"},
        "MaxPool2d": {"fill": "#ce93d8", "outline": "#8e24aa"},
        "Linear": {"fill": "#bbdefb", "outline": "#1e88e5"},
        "Flatten": {"fill": "#bbdefb", "outline": "#1e88e5"},
    },
    shade_step=0.20,
    opacity=0.95,
    **COMMON,
)


print("Wrote", len(list(OUT.glob("vt_*.svg"))), "SVGs to", OUT)
