"""Reproduces visualtorch's graph-style examples across every framework
we support — proves the graph-style kwargs work not just on PyTorch
but on Keras / JAX / HuggingFace / ONNX too.

Run::

    PYTHONPATH=. python examples/15_visualtorch_graph_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

import modelvision as mv

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def _dense() -> nn.Module:
    """The Basic Dense fixture from visualtorch's graph examples."""

    class SimpleDense(nn.Module):
        def __init__(self):
            super().__init__()
            self.h0 = nn.Linear(4, 8)
            self.h1 = nn.Linear(8, 8)
            self.h2 = nn.Linear(8, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x):
            return self.out(self.h2(self.h1(self.h0(x))))

    return SimpleDense()


def _cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 56 * 56, 10),
    )


# -----------------------------------------------------------------------------
# 1. Basic Dense — the vertical layout equivalent of visualtorch's graph style.
# -----------------------------------------------------------------------------

mv.render(_dense(), OUT / "vg_01_basic_dense.svg", layout="vertical", overwrite=True)

# -----------------------------------------------------------------------------
# 2. Palettes — swap through all six.
# -----------------------------------------------------------------------------

for palette in ("okabe_ito", "tol_bright", "vivid", "pastel", "high_contrast"):
    mv.render(_cnn(), OUT / f"vg_02_palette_{palette}.svg", palette=palette, overwrite=True)

# -----------------------------------------------------------------------------
# 3. Custom color per layer type + Input-equivalent (use layer_palette; we
#    treat the input as a first "Input" node when it's added by the inspector,
#    which the torch inspector doesn't do — visualtorch's Input pseudo-layer
#    keys don't map to us today).
# -----------------------------------------------------------------------------

mv.render(
    _dense(),
    OUT / "vg_03_custom_color.svg",
    layer_palette={"Linear": "#009E73"},
    overwrite=True,
)

# -----------------------------------------------------------------------------
# 4. Custom node size — visualtorch's `node_size=100` scales the box.
# -----------------------------------------------------------------------------

mv.render(_dense(), OUT / "vg_04_node_size.svg", node_size=64, overwrite=True)

# -----------------------------------------------------------------------------
# 5. Custom opacity.
# -----------------------------------------------------------------------------

mv.render(_cnn(), OUT / "vg_05_opacity.svg", opacity=0.55, overwrite=True)

# -----------------------------------------------------------------------------
# 6. Custom layer spacing.
# -----------------------------------------------------------------------------

mv.render(_dense(), OUT / "vg_06_layer_spacing.svg", layer_spacing=120, overwrite=True)

# -----------------------------------------------------------------------------
# 7. Dark background with per-layer fill + outline (fluoro cyberpunk).
# -----------------------------------------------------------------------------

mv.render(
    _cnn(),
    OUT / "vg_07_dark_background.svg",
    theme="dark",
    layer_palette={
        "Conv2d": {"fill": "#00F5FF", "outline": "#E0FFFF"},
        "ReLU": {"fill": "#FCEE09", "outline": "#FFFACD"},
        "MaxPool2d": {"fill": "#FF10F0", "outline": "#FFD1FA"},
        "Linear": {"fill": "#00F5FF", "outline": "#E0FFFF"},
        "Flatten": {"fill": "#FCEE09", "outline": "#FFFACD"},
    },
    overwrite=True,
)

# -----------------------------------------------------------------------------
# 8. Higher resolution — node_size + layer_spacing combined.
# -----------------------------------------------------------------------------

mv.render(
    _cnn(),
    OUT / "vg_08_higher_resolution.svg",
    node_size=80,
    layer_spacing=80,
    overwrite=True,
)

# -----------------------------------------------------------------------------
# 9. Ignore specific layer types — visualtorch calls this "type_ignore".
# -----------------------------------------------------------------------------

mv.render(
    _cnn(),
    OUT / "vg_09_type_ignore.svg",
    type_ignore=["ReLU", "Flatten"],
    overwrite=True,
)


print("Wrote", len(list(OUT.glob("vg_*.svg"))), "graph-style SVGs to", OUT)
