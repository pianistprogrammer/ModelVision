"""VGG-16 rendered with a dark theme + per-layer palette.

Requires the ``torch`` extra (which pulls in ``torchvision``)::

    uv add "modelvision[torch]"

Run::

    python examples/torch_vgg.py
"""

from __future__ import annotations

import modelvision as mv


def main() -> None:
    # ``torchvision`` ships with the ``torch`` extra of modelvision.
    from torchvision import models  # type: ignore[import-not-found]

    mv.render(
        models.vgg16(),
        output="vgg16.svg",
        theme="dark",
        layer_palette={
            "Conv2d": "#4a90d9",
            "ReLU": "#27ae60",
            "MaxPool2d": "#e67e22",
            "Linear": "#9b59b6",
            "Dropout": "#7f8c8d",
        },
        show_params=True,
    )
    print("wrote vgg16.svg")


if __name__ == "__main__":
    main()
