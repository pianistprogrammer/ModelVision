"""Example 06 — Rich scikit-learn pipeline visualization.

Handles Pipeline, ColumnTransformer, FeatureUnion, and GridSearchCV.

Requires the ``sklearn`` extra::

    uv add "modelvision[sklearn]"

Run::

    python examples/06_sklearn.py
"""

from __future__ import annotations

import modelvision as mv


def main() -> None:
    from sklearn.compose import ColumnTransformer  # type: ignore[import-not-found]
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # -----------------------------------------------------------------
    # 1. Simple linear pipeline.
    # -----------------------------------------------------------------
    simple = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10)),
            ("clf", GradientBoostingClassifier()),
        ]
    )
    mv.render(simple, "06_simple.svg", theme="light")

    # -----------------------------------------------------------------
    # 2. ColumnTransformer with distinct numeric / categorical branches.
    # -----------------------------------------------------------------
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ct = ColumnTransformer(
        [
            ("num", numeric, ["age", "income", "score"]),
            ("cat", categorical, ["city", "occupation"]),
        ]
    )
    full = Pipeline([("pre", ct), ("clf", GradientBoostingClassifier())])

    mv.render(
        full,
        "06_columntransformer.svg",
        theme="dark",
        layer_palette={
            "SimpleImputer": "#f97316",
            "StandardScaler": "#3b82f6",
            "OneHotEncoder": "#10b981",
            "PCA": "#8b5cf6",
            "GradientBoostingClassifier": "#e11d48",
        },
    )

    # -----------------------------------------------------------------
    # 3. GridSearchCV wrapping a pipeline — inspects the best estimator.
    # -----------------------------------------------------------------
    grid = GridSearchCV(
        simple,
        param_grid={"pca__n_components": [5, 10]},
        cv=3,
    )
    # GridSearchCV without ``.fit()`` doesn't have ``best_estimator_`` yet —
    # ModelVision falls back to inspecting the ``estimator`` field.
    grid.best_estimator_ = simple  # type: ignore[attr-defined]
    mv.render(grid, "06_gridsearch.svg", theme="grayscale")

    print("wrote 06_simple.svg, 06_columntransformer.svg, 06_gridsearch.svg")


if __name__ == "__main__":
    main()
