"""A three-stage sklearn pipeline rendered as a diagram."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import modelvision as mv


def main() -> None:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
            ("clf", LogisticRegression()),
        ]
    )
    mv.render(pipeline, output="pipeline.svg", theme="light")


if __name__ == "__main__":
    main()
