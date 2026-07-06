"""Example 12 — Visualtorch-inspired volumetric rendering.

Demonstrates the new options that mirror `visualtorch <https://visualtorch.readthedocs.io>`_
aesthetics: 3D isometric extruded cuboids, stacked-slice channel
visualization, the Okabe-Ito colorblind-safe palette, size-by-shape,
and legend rendering.

Run::

    python examples/12_volumetric.py
"""

from __future__ import annotations

import torch.nn as nn

import modelvision as mv


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main() -> None:
    model = TinyCNN()

    # -----------------------------------------------------------------
    # 1. Flat + legend + Okabe-Ito colorblind-safe palette.
    #    Recommended default for scientific papers.
    # -----------------------------------------------------------------
    mv.render(
        model,
        "12_flat_okabe.svg",
        palette="okabe_ito",
        legend=True,
        theme="light",
    )

    # -----------------------------------------------------------------
    # 2. Volumetric — the visualtorch signature look. Every node becomes
    #    an isometric extruded cuboid with light-from-top-left shading.
    # -----------------------------------------------------------------
    mv.render(
        model,
        "12_volumetric.svg",
        palette="okabe_ito",
        volumetric=True,
        legend=True,
        theme="light",
    )

    # -----------------------------------------------------------------
    # 3. Stacked-slice mode — imitates visualtorch's StackedBox,
    #    great for showing "this layer has many channels" at a glance.
    # -----------------------------------------------------------------
    mv.render(
        model,
        "12_stacked.svg",
        palette="okabe_ito",
        style_variant="stacked",
        legend=True,
        theme="light",
    )

    # -----------------------------------------------------------------
    # 4. Every built-in palette side-by-side.
    # -----------------------------------------------------------------
    for name in mv.PALETTES:
        mv.render(
            model,
            f"12_palette_{name}.svg",
            palette=name,
            legend=True,
            theme="light",
        )

    # -----------------------------------------------------------------
    # 5. Build a custom layer_palette from a palette + wildcard.
    # -----------------------------------------------------------------
    palette = mv.build_layer_palette("tol_bright", wildcard="#dddddd")
    mv.render(
        model,
        "12_custom_from_palette.svg",
        layer_palette=palette,
        legend=True,
        theme="dark",
        volumetric=True,
    )

    # -----------------------------------------------------------------
    # 6. Volumetric on horizontal layout (the closest match to
    #    visualtorch's "flow" style).
    # -----------------------------------------------------------------
    mv.render(
        model,
        "12_volumetric_horizontal.svg",
        palette="okabe_ito",
        volumetric=True,
        legend=True,
        layout="horizontal",
        theme="light",
    )

    print("wrote every 12_*.svg file")
    print(f"Available palettes: {list(mv.PALETTES)}")


if __name__ == "__main__":
    main()
