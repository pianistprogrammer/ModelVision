"""Tests that ``import modelvision`` is cheap and doesn't drag frameworks in."""

from __future__ import annotations

import subprocess
import sys


def test_import_is_lazy() -> None:
    """A fresh interpreter importing ``modelvision`` must not import torch/tf/jax."""
    code = (
        "import sys, modelvision;"
        "loaded = set(m for m in sys.modules"
        " if m.split('.')[0] in {'torch','tensorflow','jax','flax','haiku','transformers','sklearn','onnx'});"
        "print(sorted(loaded))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "[]", f"framework leaked at import time: {out}"


def test_public_api_surface() -> None:
    import modelvision

    expected = {
        "AmbiguousFrameworkError",
        "Edge",
        "Group",
        "InspectionError",
        "LayerNode",
        "ModelGraph",
        "ModelVisionError",
        "ModelVisionWarning",
        "NodeStyle",
        "PALETTES",
        "RenderError",
        "SegmentGroup",
        "StyleSpec",
        "Theme",
        "__version__",
        "build_layer_palette",
        "from_torch",
        "inspect",
        "render",
    }
    assert expected.issubset(set(modelvision.__all__))
