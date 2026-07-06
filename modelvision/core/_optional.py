"""Lazy-import shim for optional framework dependencies.

Every inspector imports its framework via :func:`require` inside its
:meth:`inspect` method — never at module top — so ``import modelvision``
stays cheap and works without any framework installed.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# Module → extras group name mapping used to build the install hint.
_EXTRAS: dict[str, str] = {
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "jax": "jax",
    "flax": "jax",
    "haiku": "jax",
    "transformers": "huggingface",
    "sklearn": "sklearn",
    "onnx": "onnx",
    "cairosvg": "pdf",
    "matplotlib": "plot",
}


def require(module: str, *, extra: str | None = None) -> ModuleType:
    """Import ``module`` or raise :class:`ImportError` with an ``uv add`` hint.

    Parameters
    ----------
    module:
        Import path, e.g. ``"torch"`` or ``"flax.linen"``.
    extra:
        Override the guessed extras group. Otherwise inferred from the
        top-level package name via :data:`_EXTRAS`.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        top = module.split(".", 1)[0]
        group = extra or _EXTRAS.get(top, top)
        raise ImportError(
            f"{module!r} is required for this operation. "
            f'Install it with: uv add "modelvision[{group}]"'
        ) from exc
