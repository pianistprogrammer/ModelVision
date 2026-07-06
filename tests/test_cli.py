"""CLI tests via click's CliRunner. Uses the torch fixture when available."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from modelvision.cli import main

pytestmark = pytest.mark.torch


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


def _tiny_source(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "model.py"
    p.write_text(TINY_PY)
    return p


def test_cli_renders_svg(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    src = _tiny_source(tmp_path)
    out = tmp_path / "diagram.svg"
    runner = CliRunner()
    result = runner.invoke(main, [str(src), "TinyMLP", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.read_text().startswith("<?xml")


def test_cli_renders_html(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    src = _tiny_source(tmp_path)
    out = tmp_path / "diagram.html"
    runner = CliRunner()
    result = runner.invoke(main, [str(src), "TinyMLP", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text().startswith("<!doctype html>")


def test_cli_summary_prints_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    src = _tiny_source(tmp_path)
    runner = CliRunner()
    # The old top-level ``--summary`` flag is now the ``inspect`` subcommand.
    result = runner.invoke(main, ["inspect", str(src), "TinyMLP"])
    assert result.exit_code == 0, result.output
    # Rich table headers should appear.
    assert "TinyMLP" in result.output
    assert "Linear" in result.output


def test_cli_missing_class_name_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = _tiny_source(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, [str(src)])
    assert result.exit_code == 1
    assert "class name" in result.output.lower()


def test_cli_theme_and_palette(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    src = _tiny_source(tmp_path)
    out = tmp_path / "diagram.svg"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(src), "TinyMLP", "--output", str(out), "--theme", "dark", "--layer-palette", "Linear=#ff00ff"],
    )
    assert result.exit_code == 0, result.output
    assert "#ff00ff" in out.read_text()
