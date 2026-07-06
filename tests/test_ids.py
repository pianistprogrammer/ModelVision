"""Tests for the ID-minting utilities."""

from __future__ import annotations

from modelvision.core.ids import join, sanitize, uniquify


def test_sanitize_keeps_safe_chars() -> None:
    assert sanitize("layer_0") == "layer_0"
    assert sanitize("layer-0") == "layer-0"
    assert sanitize("features.0") == "features_0"


def test_sanitize_replaces_unsafe() -> None:
    assert sanitize("layer/0") == "layer_0"
    assert sanitize("layer 0!") == "layer_0"
    assert sanitize("!!!") == "_"


def test_join_dotted() -> None:
    assert join("features", "0", "conv") == "features.0.conv"
    assert join("features", "", "conv") == "features.conv"


def test_uniquify_stable() -> None:
    assert uniquify(["a", "b", "a", "a"]) == ["a", "b", "a_2", "a_3"]
    assert uniquify(["x"]) == ["x"]
    assert uniquify([]) == []
