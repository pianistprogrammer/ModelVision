"""Verify every runnable example script actually runs end-to-end.

These tests are slow — they exec the example scripts in a subprocess
so we get real environment behavior. Marked ``slow`` so the default
``pytest -m "not slow"`` skips them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).parent.parent / "examples"


def _run(script: str, cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Ensure the source tree is on the path when running from a checkout.
    repo_root = str(Path(__file__).parent.parent.resolve())
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_example_torch_vgg(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torchvision")
    result = _run("torch_vgg.py", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "vgg16.svg").exists()


def test_example_hf_bert(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("transformers")
    result = _run("hf_bert.py", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "bert.svg").exists()


def test_example_sklearn(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("sklearn")
    result = _run("sklearn_pipeline.py", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pipeline.svg").exists()
