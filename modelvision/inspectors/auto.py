"""Framework auto-detection.

Given an arbitrary Python object, decide which inspector to use. The
detection is purely by module-prefix on ``type(model).__mro__`` — no
framework imports happen here, so users without a framework installed
never pay for its absence.
"""

from __future__ import annotations

import os
from typing import Any

from modelvision.core.exceptions import AmbiguousFrameworkError, InspectionError

# Each entry: (framework name, module-prefix predicate). Order matters
# only for ties — a HuggingFace model IS an nn.Module, so we check the
# more specific ``transformers`` prefix first.
_PREFIXES: list[tuple[str, tuple[str, ...]]] = [
    ("huggingface", ("transformers.",)),
    ("torch", ("torch.", "torchvision.")),
    ("keras", ("keras.", "tensorflow.", "tf_keras.")),
    ("jax", ("flax.", "haiku.", "jax.")),
    ("sklearn", ("sklearn.",)),
]


def detect_framework(model: Any) -> str:
    """Return the framework name for ``model``.

    - Strings ending in ``.onnx`` → ``"onnx"``.
    - Strings ending in ``.gguf`` → ``"gguf"`` (llama.cpp / Ollama).
    - Instances → walk ``type(model).__mro__`` and match module prefix.
    - Raises :class:`AmbiguousFrameworkError` if no prefix matches.
    """
    if isinstance(model, (str, os.PathLike)):
        lower = str(model).lower()
        if lower.endswith(".onnx"):
            return "onnx"
        if lower.endswith(".gguf"):
            return "gguf"
        raise InspectionError(
            f"Cannot infer framework from path {model!r}. "
            "Pass framework= explicitly or use a supported model file (.onnx, .gguf)."
        )

    for cls in type(model).__mro__:
        module = getattr(cls, "__module__", "") or ""
        for name, prefixes in _PREFIXES:
            if any(module.startswith(p) for p in prefixes):
                return name

    raise AmbiguousFrameworkError(
        f"Could not auto-detect framework for object of type "
        f"{type(model).__module__}.{type(model).__qualname__!r}. "
        "Pass framework= explicitly."
    )
