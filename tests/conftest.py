"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Auto-skip framework-marked tests when the framework isn't installed."""
    config.addinivalue_line("markers", "torch: requires torch")


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    skip_torch = pytest.mark.skip(reason="torch not installed")
    have = {
        "torch": _has_module("torch"),
        "keras": _has_module("tensorflow") or _has_module("keras"),
        "jax": _has_module("jax"),
        "hf": _has_module("transformers"),
        "sklearn": _has_module("sklearn"),
        "onnx": _has_module("onnx"),
    }
    for item in items:
        for marker in item.iter_markers():
            if marker.name in have and not have[marker.name]:
                item.add_marker(pytest.mark.skip(reason=f"{marker.name} extra not installed"))
