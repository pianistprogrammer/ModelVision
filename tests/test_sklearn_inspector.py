"""Tests for the scikit-learn inspector."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sklearn


def test_pipeline_linear_chain() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from modelvision import inspect

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("clf", LogisticRegression()),
    ])
    g = inspect(pipe)
    assert g.metadata["framework"] == "sklearn"
    ids = [n.id for n in g.nodes]
    assert ids == ["scaler", "pca", "clf"]
    # Linear chain: 2 edges.
    assert len(g.edges) == 2
    # Pipeline group present.
    assert any(gr.id == "pipeline" for gr in g.groups)


def test_column_transformer_branches() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from modelvision import inspect

    ct = ColumnTransformer([
        ("num", StandardScaler(), ["a", "b"]),
        ("cat", OneHotEncoder(), ["c"]),
    ])
    g = inspect(ct)
    assert {n.id for n in g.nodes} == {"num", "cat"}
    assert any(gr.name == "ColumnTransformer" for gr in g.groups)


def test_pipeline_end_to_end_render(tmp_path) -> None:  # type: ignore[no-untyped-def]
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from modelvision import render

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    out = tmp_path / "pipe.svg"
    render(pipe, output=str(out), theme="pastel")
    assert out.exists()
    assert out.read_text().startswith("<?xml")
