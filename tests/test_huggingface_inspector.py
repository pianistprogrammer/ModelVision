"""Tests for the HuggingFace inspector.

Config-only path (Path B) is always available since ``transformers``
templates use only config attributes. Model-instance path (Path A) is
auto-skipped without torch.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.hf


def test_bert_config_only() -> None:
    transformers = pytest.importorskip("transformers")
    from modelvision import inspect

    config = transformers.BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=100,
    )
    g = inspect(config)
    assert g.metadata["framework"] == "huggingface"
    assert g.metadata["config_only"] is True
    assert g.metadata["model_type"] == "bert"
    # Embedding + 4 block nodes + pooler = 6.
    assert len(g.nodes) == 6
    # Encoder group.
    assert {gr.id for gr in g.groups} == {"encoder"}
    # The attention node carries a repeat count.
    attn = next(n for n in g.nodes if n.layer_type == "MultiHeadAttention")
    assert attn.attributes.get("repeat") == 2


def test_gpt2_config_only() -> None:
    transformers = pytest.importorskip("transformers")
    from modelvision import inspect

    config = transformers.GPT2Config(
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_positions=128,
        vocab_size=100,
    )
    g = inspect(config)
    assert g.metadata["model_type"] == "gpt2"
    # Shared-weight edge between wte and lm_head.
    shared = [e for e in g.edges if e.kind == "shared"]
    assert len(shared) == 1


def test_vit_config_only() -> None:
    transformers = pytest.importorskip("transformers")
    from modelvision import inspect

    config = transformers.ViTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_labels=10,
    )
    g = inspect(config)
    assert g.metadata["model_type"] == "vit"
    assert any(n.layer_type == "Conv2d" for n in g.nodes)


def test_unknown_model_type_falls_back_to_generic() -> None:
    transformers = pytest.importorskip("transformers")
    from modelvision import inspect

    # Use a config that has num_hidden_layers/hidden_size but a rare model_type.
    class _MysteryConfig(transformers.PretrainedConfig):
        model_type = "totally_new_arch"

        def __init__(self, **kw):
            super().__init__(**kw)
            self.num_hidden_layers = 3
            self.hidden_size = 16
            self.num_attention_heads = 2
            self.vocab_size = 100
            self.intermediate_size = 32

    g = inspect(_MysteryConfig())
    assert g.metadata["framework"] == "huggingface"
    # Generic template produces at least embedding + block + pooler.
    assert len(g.nodes) >= 3


def test_model_instance_path_delegates_to_torch() -> None:
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
    g = inspect(model, expand_groups=True)  # skip folding to inspect raw graph
    assert g.metadata["framework"] == "huggingface"
    # Real BertModel has many leaves.
    assert len(g.nodes) > 10
