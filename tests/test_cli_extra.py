"""Additional CLI tests — exercise error paths and non-torch extensions."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from modelvision.cli import main


def test_cli_unknown_extension_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "unknown.xyz"
    src.write_text("")
    runner = CliRunner()
    result = runner.invoke(main, [str(src)])
    assert result.exit_code == 1
    assert "extension" in result.output.lower()


def test_cli_class_not_in_source_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    src = tmp_path / "model.py"
    src.write_text(
        "import torch.nn as nn\n\nclass Other(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.l = nn.Linear(4, 4)\n"
    )
    runner = CliRunner()
    result = runner.invoke(main, [str(src), "Nonexistent"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_cli_init_args_json_passed_to_constructor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    src = tmp_path / "model.py"
    src.write_text(
        "import torch.nn as nn\n"
        "class Cfg(nn.Module):\n"
        "    def __init__(self, width=4):\n"
        "        super().__init__()\n"
        "        self.l = nn.Linear(width, width)\n"
    )
    out = tmp_path / "d.svg"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(src), "Cfg", "-o", str(out), "--init-args", json.dumps({"width": 8})],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_onnx_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "t", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    src = tmp_path / "m.onnx"
    onnx.save(model, str(src))

    out = tmp_path / "m.svg"
    runner = CliRunner()
    result = runner.invoke(main, [str(src), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
