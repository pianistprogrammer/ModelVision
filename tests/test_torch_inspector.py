"""Tests for the PyTorch inspector.

Uses small hand-built fixture models so CI stays fast. Each test is
marked ``torch`` so it auto-skips when the extra isn't installed.
"""

from __future__ import annotations

import warnings

import pytest

from modelvision import Group, ModelVisionWarning, inspect
from modelvision._api import render

pytestmark = pytest.mark.torch


def _import_torch():  # type: ignore[no-untyped-def]
    torch = pytest.importorskip("torch")
    return torch, torch.nn


# ---------------------------------------------------------------------------
# Fixture models
# ---------------------------------------------------------------------------


def _tiny_mlp():  # type: ignore[no-untyped-def]
    _, nn = _import_torch()

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    return TinyMLP()


def _tiny_cnn():  # type: ignore[no-untyped-def]
    _, nn = _import_torch()

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16 * 16, 10))

        def forward(self, x):
            return self.classifier(self.features(x))

    return TinyCNN()


def _tied_model():  # type: ignore[no-untyped-def]
    _, nn = _import_torch()

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(4, 4)
            self.a = shared
            self.b = shared

        def forward(self, x):
            return self.b(self.a(x))

    return Tied()


def _module_list():  # type: ignore[no-untyped-def]
    _, nn = _import_torch()

    class MLModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return MLModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tiny_mlp_nodes_and_params() -> None:
    g = inspect(_tiny_mlp())
    assert [n.id for n in g.nodes] == ["fc1", "fc2"]
    assert g.nodes[0].layer_type == "Linear"
    assert g.nodes[0].attributes["in_features"] == 4
    assert g.metadata["total_params"] == (4 * 8 + 8) + (8 * 2 + 2)
    assert len(g.edges) == 1
    assert g.edges[0].source_id == "fc1" and g.edges[0].target_id == "fc2"


def test_tiny_cnn_groups_and_edges() -> None:
    g = inspect(_tiny_cnn())
    # 4 leaves in features + 2 in classifier.
    assert len(g.nodes) == 6
    # Two groups: features, classifier — each owns ≥ 2 leaves.
    assert {grp.id for grp in g.groups} == {"features", "classifier"}
    # Sequential edges within each parent scope only. Cross-scope edges
    # would require symbolic_shapes=True (opt-in, per PRD §13 Q1).
    kinds = [e.kind for e in g.edges]
    # 3 edges inside `features` (4 leaves) + 1 edge inside `classifier` (2 leaves).
    assert kinds.count("data") == 4


def test_module_list_is_unrolled() -> None:
    g = inspect(_module_list())
    assert [n.id for n in g.nodes] == ["layers.0", "layers.1", "layers.2"]


def test_weight_tying_emits_shared_edge() -> None:
    g = inspect(_tied_model())
    shared = [e for e in g.edges if e.kind == "shared"]
    assert len(shared) == 1
    assert shared[0].source_id == "a"
    assert shared[0].target_id == "b"
    assert shared[0].label == "shared"


def test_empty_module_placeholder() -> None:
    _, nn = _import_torch()

    class Empty(nn.Module):
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = inspect(Empty())
    assert len(g.nodes) == 1
    assert g.nodes[0].layer_type == "EmptyModule"
    assert any(issubclass(rec.category, ModelVisionWarning) for rec in w)


def test_data_parallel_is_unwrapped() -> None:
    torch, nn = _import_torch()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    dp = torch.nn.DataParallel(Inner())
    g = inspect(dp)
    assert {n.id for n in g.nodes} == {"linear"}


def test_render_writes_valid_svg(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "cnn.svg"
    render(_tiny_cnn(), output=out, theme="dark")
    text = out.read_text()
    assert text.startswith("<?xml")
    assert 'data-node-id="features.0"' in text
    assert "</svg>" in text


def test_render_inline_returns_svg_string() -> None:
    result = render(_tiny_mlp(), theme="light")
    assert isinstance(result, str)
    assert result.startswith("<?xml")


def test_render_with_user_palette_and_group() -> None:
    # Palette applies to Conv2d nodes in ``features.*``, but the group's
    # ``fill`` overrides at a higher priority level — verify both by
    # picking a group whose nodes are non-Conv (classifier).
    svg = render(
        _tiny_cnn(),
        theme="dark",
        layer_palette={"Conv2d": "#123456"},
        groups=[Group(id="cls", node_pattern="classifier.*", fill="#654321")],
    )
    assert "#123456" in svg  # Conv2d in features gets the palette
    assert "#654321" in svg  # classifier.* gets the group fill
