"""ModelVision — framework-agnostic neural network architecture visualizer.

Public surface: :func:`render`, :func:`inspect`, :func:`from_torch`,
:func:`from_keras`, :func:`from_jax`, :func:`from_huggingface`,
:func:`from_sklearn`, :func:`from_onnx`, plus the style/theme dataclasses.

Framework imports are lazy — importing this package does not import
torch/tensorflow/jax/etc. Each :func:`from_*` entry point pulls in its
framework at call time via :mod:`modelvision.core._optional`.
"""

from __future__ import annotations

from typing import Any

from modelvision.core.exceptions import (
    AmbiguousFrameworkError,
    InspectionError,
    ModelVisionError,
    ModelVisionWarning,
    RenderError,
)
from modelvision.core.ir import Edge, LayerNode, ModelGraph, SegmentGroup
from modelvision.core.palettes import PALETTES, build_layer_palette
from modelvision.core.style import Group, NodeStyle, StyleSpec, Theme

try:
    from modelvision._version import __version__
except ImportError:  # pragma: no cover - only during editable installs without hatch-vcs
    __version__ = "0.0.0"


# ---------------------------------------------------------------------------
# Public entry points. All lazily import ``_api`` so that ``import modelvision``
# doesn't drag in the render pipeline (and its dependencies) until needed.
# ---------------------------------------------------------------------------


def render(model: Any, output: Any = None, /, **kwargs: Any) -> Any:
    """Render a model to a diagram. See PRD §7.2 for the full signature."""
    from modelvision._api import render as _render

    if output is not None:
        kwargs.setdefault("output", output)
    return _render(model, **kwargs)


def inspect(model: Any, /, framework: str | None = None, **kwargs: Any) -> ModelGraph:
    """Return the normalized :class:`ModelGraph` IR for a model."""
    from modelvision._api import inspect as _inspect

    return _inspect(model, framework=framework, **kwargs)


def from_torch(model: Any, /, **kwargs: Any) -> Any:
    from modelvision._api import from_torch as _from_torch

    return _from_torch(model, **kwargs)


def from_keras(model: Any, /, **kwargs: Any) -> Any:
    from modelvision._api import from_keras as _from_keras

    return _from_keras(model, **kwargs)


def from_jax(module: Any, /, **kwargs: Any) -> Any:
    from modelvision._api import from_jax as _from_jax

    return _from_jax(module, **kwargs)


def from_huggingface(model_or_config: Any, /, **kwargs: Any) -> Any:
    from modelvision._api import from_huggingface as _from_hf

    return _from_hf(model_or_config, **kwargs)


def from_sklearn(pipeline: Any, /, **kwargs: Any) -> Any:
    from modelvision._api import from_sklearn as _from_sk

    return _from_sk(pipeline, **kwargs)


def from_onnx(path: str, /, **kwargs: Any) -> Any:
    from modelvision._api import from_onnx as _from_onnx

    return _from_onnx(path, **kwargs)


def from_gguf(path: str, /, **kwargs: Any) -> Any:
    """Visualize a ``.gguf`` file (llama.cpp / Ollama). Header-only, no weights loaded."""
    from modelvision._api import from_gguf as _from_gguf

    return _from_gguf(path, **kwargs)


__all__ = [
    "PALETTES",
    "AmbiguousFrameworkError",
    "Edge",
    "Group",
    "InspectionError",
    "LayerNode",
    "ModelGraph",
    "ModelVisionError",
    "ModelVisionWarning",
    "NodeStyle",
    "RenderError",
    "SegmentGroup",
    "StyleSpec",
    "Theme",
    "__version__",
    "build_layer_palette",
    "from_huggingface",
    "from_jax",
    "from_keras",
    "from_onnx",
    "from_gguf",
    "from_sklearn",
    "from_torch",
    "inspect",
    "render",
]
