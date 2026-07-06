"""Golden-file SVG regression tests.

Runs only under ``pytest -m golden`` (the CI ``golden`` job restricts this
to Ubuntu 3.11 + ``all`` extras so cross-platform float stringification
drift doesn't break merges).

To regenerate goldens after intentional visual changes::

    python -m tests.regen_goldens

Then eyeball the diff and commit ``tests/golden/*.svg`` if it looks right.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelvision import Edge, LayerNode, ModelGraph
from modelvision.layout.vertical import layout_vertical
from modelvision.renderers.svg_renderer import render_svg
from modelvision.themes import get_theme

pytestmark = pytest.mark.golden

GOLDEN_DIR = Path(__file__).parent / "golden"


# ---------------------------------------------------------------------------
# Fixture graphs — deliberately small and framework-independent so the
# goldens don't depend on torch/keras/etc. versions.
# ---------------------------------------------------------------------------


def _linear_chain() -> ModelGraph:
    return ModelGraph(
        nodes=[
            LayerNode(id="conv", name="conv", layer_type="Conv2d", framework="test", params=448),
            LayerNode(id="bn", name="bn", layer_type="BatchNorm2d", framework="test", params=32),
            LayerNode(id="act", name="act", layer_type="ReLU", framework="test"),
            LayerNode(id="fc", name="fc", layer_type="Linear", framework="test", params=1290),
        ],
        edges=[
            Edge(source_id="conv", target_id="bn"),
            Edge(source_id="bn", target_id="act"),
            Edge(source_id="act", target_id="fc"),
        ],
        metadata={"model_class": "LinearChain"},
    )


def _fan_in() -> ModelGraph:
    """Two-branch graph — used to verify skip-connection merge insertion."""
    from modelvision.core.postprocess import post_process

    g = ModelGraph(
        nodes=[
            LayerNode(id="input", name="input", layer_type="Input", framework="test"),
            LayerNode(id="left", name="left", layer_type="Conv2d", framework="test", params=100),
            LayerNode(id="right", name="right", layer_type="Conv2d", framework="test", params=100),
            LayerNode(id="add", name="add", layer_type="Add", framework="test"),
        ],
        edges=[
            Edge(source_id="input", target_id="left"),
            Edge(source_id="input", target_id="right"),
            Edge(source_id="left", target_id="add"),
            Edge(source_id="right", target_id="add"),
        ],
    )
    return post_process(g)


_FIXTURES: dict[str, tuple[ModelGraph, str]] = {
    "linear_chain_light": (_linear_chain(), "light"),
    "linear_chain_dark": (_linear_chain(), "dark"),
    "linear_chain_high_contrast": (_linear_chain(), "high_contrast"),
    "fan_in_pastel": (_fan_in(), "pastel"),
}


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_svg_matches_golden(name: str) -> None:
    graph, theme = _FIXTURES[name]
    actual = render_svg(layout_vertical(graph), theme=get_theme(theme))
    expected_path = GOLDEN_DIR / f"{name}.svg"
    if not expected_path.exists():  # pragma: no cover
        pytest.fail(
            f"Golden file {expected_path} is missing. "
            "Run `python -m tests.regen_goldens` to create it."
        )
    assert actual == expected_path.read_text(), (
        f"SVG output for {name!r} does not match {expected_path.name}. "
        "If this is an intentional visual change, run "
        "`python -m tests.regen_goldens` and commit the diff."
    )


def _regen_all() -> None:
    """Regenerate every golden. Called by ``python -m tests.regen_goldens``."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, (graph, theme) in _FIXTURES.items():
        svg = render_svg(layout_vertical(graph), theme=get_theme(theme))
        (GOLDEN_DIR / f"{name}.svg").write_text(svg)
        print(f"wrote {GOLDEN_DIR / f'{name}.svg'}")


if __name__ == "__main__":  # pragma: no cover
    _regen_all()
