"""Extra huggingface tests to cover the folding logic and less-common paths."""

from __future__ import annotations

import warnings

import pytest


def test_hf_no_config_no_folding() -> None:
    """A torch nn.Module that happens to be under transformers namespace but has
    no ``config`` attribute must round-trip through the delegate without folding."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import torch.nn as nn

    # Just delegate — no config → no folding path.
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)

    Fake.__module__ = "transformers.custom"  # trick auto-detect

    from modelvision import inspect

    g = inspect(Fake())
    assert g.metadata["framework"] == "huggingface"
    assert len(g.nodes) == 1


def test_hf_folding_warns() -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from modelvision import inspect

    config = transformers.BertConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=50,
    )
    model = transformers.BertModel(config)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        inspect(model)
    assert any("Folded" in str(r.message) for r in records)


def test_hf_expand_groups_skips_folding() -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from modelvision import inspect

    config = transformers.BertConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=50,
    )
    g_folded = inspect(transformers.BertModel(config))
    g_expanded = inspect(transformers.BertModel(config), expand_groups=True)
    assert len(g_expanded.nodes) > len(g_folded.nodes)


def test_hf_t5_config() -> None:
    transformers = pytest.importorskip("transformers")
    from modelvision import inspect

    config = transformers.T5Config(
        d_model=32, num_layers=2, num_decoder_layers=2, num_heads=2, vocab_size=100
    )
    g = inspect(config)
    assert g.metadata["model_type"] == "t5"
    # Encoder-decoder produces two groups.
    assert {gr.id for gr in g.groups} == {"encoder", "decoder"}


def test_hf_bart_config() -> None:
    transformers = pytest.importorskip("transformers")
    from modelvision import inspect

    config = transformers.BartConfig(
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
    )
    g = inspect(config)
    assert g.metadata["model_type"] == "bart"
