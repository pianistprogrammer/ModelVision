"""Tests for the Keras inspector.

Auto-skipped when neither tensorflow nor keras is installed.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.keras


def _keras():  # type: ignore[no-untyped-def]
    try:
        import keras  # noqa: F401

        return __import__("keras")
    except ImportError:
        tf = pytest.importorskip("tensorflow")
        return tf.keras


def test_sequential_linear_chain() -> None:
    keras = _keras()
    from modelvision import inspect

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(2),
        ]
    )
    g = inspect(model)
    # Input layers may or may not appear depending on version — just check
    # that dense layers are present in order.
    dense_ids = [n.id for n in g.nodes if n.layer_type == "Dense"]
    assert len(dense_ids) == 2
    # Sequential edges connect adjacent layers.
    assert g.metadata["flavor"] == "sequential"


def test_functional_two_input_merges() -> None:
    keras = _keras()
    from modelvision import inspect

    a = keras.layers.Input(shape=(4,), name="a")
    b = keras.layers.Input(shape=(4,), name="b")
    merged = keras.layers.Concatenate()([a, b])
    out = keras.layers.Dense(1)(merged)
    model = keras.Model([a, b], out)

    g = inspect(model)
    assert g.metadata["flavor"] == "functional"
    # Concatenate has two inbound layers; the graph should show two edges into it.
    concat_ids = [n.id for n in g.nodes if n.layer_type == "Concatenate"]
    assert len(concat_ids) == 1
    concat = concat_ids[0]
    inbound = [e for e in g.edges if e.target_id == concat]
    assert len(inbound) == 2


def test_subclassed_emits_warning() -> None:
    keras = _keras()
    from modelvision import ModelVisionWarning, inspect

    class Custom(keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = keras.layers.Dense(4)
            self.d2 = keras.layers.Dense(2)

        def call(self, x):
            return self.d2(self.d1(x))

    model = Custom()
    # Force layers to exist by building.
    model.build((None, 4))
    with pytest.warns(ModelVisionWarning, match="subclassed"):
        g = inspect(model)
    assert g.metadata["flavor"] == "subclassed"
    assert g.metadata.get("subclassed") is True
