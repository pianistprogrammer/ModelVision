"""Tests for the expanded LLM-friendly CLI surface.

Covers the ``list`` subcommands, ``--json``/``--stdout``/``--dry-run``
flags, and confirms every renderer option is reachable via the CLI.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from modelvision.cli import main

TINY_PY = """\
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc2(self.fc1(x))
"""


@pytest.fixture
def tiny_source(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "model.py"
    p.write_text(TINY_PY)
    return p


# ---------------------------------------------------------------------------
# ``mvision list`` subcommands
# ---------------------------------------------------------------------------


def test_list_palettes_default_output() -> None:
    result = CliRunner().invoke(main, ["list", "palettes"])
    assert result.exit_code == 0, result.output
    assert "okabe_ito" in result.output


def test_list_palettes_json() -> None:
    result = CliRunner().invoke(main, ["list", "palettes", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "okabe_ito" in data
    assert isinstance(data["okabe_ito"], list)
    assert data["okabe_ito"][0].startswith("#")


def test_list_themes_json() -> None:
    result = CliRunner().invoke(main, ["list", "themes", "--json"])
    assert result.exit_code == 0, result.output
    themes = json.loads(result.output)
    assert set(themes) == {"light", "dark", "pastel", "grayscale", "high_contrast"}


def test_list_layouts_json() -> None:
    result = CliRunner().invoke(main, ["list", "layouts", "--json"])
    assert result.exit_code == 0, result.output
    layouts = json.loads(result.output)
    assert "flow" in layouts
    assert "vertical" in layouts


def test_list_frameworks_json() -> None:
    result = CliRunner().invoke(main, ["list", "frameworks", "--json"])
    assert result.exit_code == 0, result.output
    fws = json.loads(result.output)
    assert "torch" in fws
    assert "auto" in fws


# ---------------------------------------------------------------------------
# ``mvision render`` — new flags & modes
# ---------------------------------------------------------------------------


def test_render_stdout_writes_svg_to_stdout(tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    result = CliRunner().invoke(
        main, ["render", str(tiny_source), "TinyMLP", "--stdout"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("<?xml")
    assert "</svg>" in result.output


def test_render_palette_flag(tmp_path, tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    out = tmp_path / "d.svg"
    result = CliRunner().invoke(
        main,
        ["render", str(tiny_source), "TinyMLP",
         "-o", str(out), "--palette", "okabe_ito", "--legend"],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    # Legend group appears in the SVG.
    assert 'class="mv-legend"' in text
    # An Okabe-Ito palette color surfaces — Linear maps to index 1 = #56B4E9.
    assert "#56b4e9" in text.lower()


def test_render_volumetric_flag(tmp_path, tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    out = tmp_path / "d.svg"
    result = CliRunner().invoke(
        main,
        ["render", str(tiny_source), "TinyMLP", "-o", str(out),
         "--volumetric", "--palette", "vivid"],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    # Volumetric mode emits polygon faces for the extruded top / right.
    assert "<polygon" in text


def test_render_flow_layout_needs_input_shape(tmp_path, tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    out = tmp_path / "d.svg"
    result = CliRunner().invoke(
        main,
        ["render", str(tiny_source), "TinyMLP", "-o", str(out),
         "--layout", "flow", "--input-shape", "1x4"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_render_dry_run_emits_json(tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    result = CliRunner().invoke(
        main,
        ["render", str(tiny_source), "TinyMLP", "--dry-run", "--theme", "dark"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["class_name"] == "TinyMLP"
    assert payload["resolved_kwargs"]["theme"] == "dark"
    assert payload["graph"]["nodes"] > 0
    assert payload["graph"]["total_params"] == 4 * 8 + 8 + 8 * 2 + 2


def test_render_without_output_or_stdout_fails(tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    result = CliRunner().invoke(main, ["render", str(tiny_source), "TinyMLP"])
    assert result.exit_code == 2
    assert "output" in result.output.lower() or "stdout" in result.output.lower()


def test_render_accessibility_enforce(tmp_path, tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    out = tmp_path / "d.svg"
    result = CliRunner().invoke(
        main,
        ["render", str(tiny_source), "TinyMLP", "-o", str(out),
         "--accessibility", "enforce"],
    )
    assert result.exit_code == 0, result.output


def test_render_unknown_palette_rejected_by_choice(tiny_source) -> None:  # type: ignore[no-untyped-def]
    result = CliRunner().invoke(
        main,
        ["render", str(tiny_source), "TinyMLP",
         "--palette", "not_a_real_palette", "--stdout"],
    )
    assert result.exit_code == 2  # click usage error
    assert "not_a_real_palette" in result.output


# ---------------------------------------------------------------------------
# ``mvision inspect``
# ---------------------------------------------------------------------------


def test_inspect_json_dumps_graph(tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    result = CliRunner().invoke(
        main, ["inspect", str(tiny_source), "TinyMLP", "--json"]
    )
    assert result.exit_code == 0, result.output
    graph = json.loads(result.output)
    assert "nodes" in graph
    assert "edges" in graph
    assert any(n["id"] == "fc1" for n in graph["nodes"])


def test_inspect_json_to_file(tmp_path, tiny_source) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    out = tmp_path / "graph.json"
    result = CliRunner().invoke(
        main,
        ["inspect", str(tiny_source), "TinyMLP", "--json", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    graph = json.loads(out.read_text())
    assert graph["metadata"]["framework"] == "torch"


# ---------------------------------------------------------------------------
# Bareword fallback — backward-compat with the M4 CLI shape.
# ---------------------------------------------------------------------------


def test_bareword_still_works(tmp_path, tiny_source) -> None:  # type: ignore[no-untyped-def]
    """``mvision model.py MyNet -o out.svg`` (no ``render`` verb) still works."""
    pytest.importorskip("torch")
    out = tmp_path / "d.svg"
    result = CliRunner().invoke(main, [str(tiny_source), "TinyMLP", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
