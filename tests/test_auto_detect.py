"""Tests for framework auto-detection (works without any framework installed)."""

from __future__ import annotations

import pytest

from modelvision.core.exceptions import AmbiguousFrameworkError
from modelvision.inspectors.auto import detect_framework


class _FakeTorchModule:
    pass


_FakeTorchModule.__module__ = "torch.nn.modules.linear"


class _FakeKerasModule:
    pass


_FakeKerasModule.__module__ = "keras.layers.core"


class _FakeHFModule:
    pass


_FakeHFModule.__module__ = "transformers.models.bert.modeling_bert"


class _FakePlainObject:
    pass


def test_detects_torch() -> None:
    assert detect_framework(_FakeTorchModule()) == "torch"


def test_detects_keras() -> None:
    assert detect_framework(_FakeKerasModule()) == "keras"


def test_detects_huggingface_over_torch() -> None:
    # HF models are also torch nn.Modules — the more specific prefix wins.
    assert detect_framework(_FakeHFModule()) == "huggingface"


def test_detects_onnx_from_path() -> None:
    assert detect_framework("model.onnx") == "onnx"
    assert detect_framework("dir/subdir/model.ONNX") == "onnx"


def test_raises_on_unknown() -> None:
    with pytest.raises(AmbiguousFrameworkError):
        detect_framework(_FakePlainObject())
