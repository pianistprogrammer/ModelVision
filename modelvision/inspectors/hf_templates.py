"""HuggingFace architecture templates.

For config-only inspection (no model instance available) we synthesize
a canonical block diagram from the config's key hyperparameters. Each
template is a small function ``build(config) -> ModelGraph`` — the
registry maps ``config.model_type`` to a template, with a generic
transformer fallback that consumes ``num_hidden_layers``,
``hidden_size``, and ``num_attention_heads``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from modelvision.core.ir import Edge, LayerNode, ModelGraph, SegmentGroup


def is_config(obj: Any) -> bool:
    """True if ``obj`` is (or subclasses) :class:`transformers.PretrainedConfig`."""
    for cls in type(obj).__mro__:
        mod = getattr(cls, "__module__", "") or ""
        if mod.startswith("transformers.") and cls.__name__ == "PretrainedConfig":
            return True
    return False


def build_from_config(config: Any) -> ModelGraph:
    """Dispatch by ``config.model_type`` to the right template."""
    model_type = getattr(config, "model_type", None) or "generic"
    template = _REGISTRY.get(model_type, _generic_transformer)
    graph = template(config)
    graph.metadata["framework"] = "huggingface"
    graph.metadata["model_type"] = model_type
    graph.metadata["config_only"] = True
    return graph


# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------


def _mk_node(nid: str, name: str, layer_type: str, **attrs: Any) -> LayerNode:
    return LayerNode(
        id=nid,
        name=name,
        layer_type=layer_type,
        framework="huggingface",
        attributes={k: v for k, v in attrs.items() if v is not None},
    )


def _encoder_block(prefix: str, config: Any, index: int, *, is_last: bool) -> tuple[list[LayerNode], list[Edge]]:
    """One transformer encoder block: attention → norm → FFN → norm."""
    hs = getattr(config, "hidden_size", None)
    nh = getattr(config, "num_attention_heads", None)
    ff = getattr(config, "intermediate_size", None)
    activation = getattr(config, "hidden_act", None)

    ids = [f"{prefix}.attn", f"{prefix}.norm1", f"{prefix}.ffn", f"{prefix}.norm2"]
    nodes = [
        _mk_node(ids[0], "self_attn", "MultiHeadAttention", num_heads=nh, embed_dim=hs),
        _mk_node(ids[1], "layernorm", "LayerNorm", num_features=hs),
        _mk_node(ids[2], "feed_forward", "MLP", intermediate_size=ff, activation=activation),
        _mk_node(ids[3], "layernorm", "LayerNorm", num_features=hs),
    ]
    edges = [
        Edge(source_id=ids[0], target_id=ids[1]),
        Edge(source_id=ids[1], target_id=ids[2]),
        Edge(source_id=ids[2], target_id=ids[3]),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _generic_transformer(config: Any) -> ModelGraph:
    """Encoder stack — used for BERT-family and as fallback."""
    num_layers = getattr(config, "num_hidden_layers", 1) or 1
    hidden = getattr(config, "hidden_size", None)
    vocab = getattr(config, "vocab_size", None)

    nodes: list[LayerNode] = [
        _mk_node("embedding", "embedding", "Embedding", num_embeddings=vocab, embedding_dim=hidden),
    ]
    edges: list[Edge] = []
    groups: list[SegmentGroup] = []
    prev_id = "embedding"

    # We render the *first* block in detail and mark it with a repeat
    # count so the renderer can print "× N" in M5 when expand_groups=False.
    block_prefix = "encoder.layer.0"
    block_nodes, block_edges = _encoder_block(block_prefix, config, 0, is_last=False)
    if num_layers > 1:
        block_nodes[0].attributes["repeat"] = num_layers
    nodes.extend(block_nodes)
    edges.append(Edge(source_id=prev_id, target_id=block_nodes[0].id))
    edges.extend(block_edges)
    groups.append(
        SegmentGroup(
            id="encoder",
            name=f"Encoder × {num_layers}",
            node_ids=[n.id for n in block_nodes],
        )
    )
    prev_id = block_nodes[-1].id

    nodes.append(_mk_node("pooler", "pooler", "Linear", out_features=hidden))
    edges.append(Edge(source_id=prev_id, target_id="pooler"))

    return ModelGraph(nodes=nodes, edges=edges, groups=groups)


def _causal_decoder(config: Any) -> ModelGraph:
    """GPT-family: embedding → decoder × N → LM head."""
    num_layers = getattr(config, "num_hidden_layers", 1) or 1
    hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    vocab = getattr(config, "vocab_size", None)

    nodes: list[LayerNode] = [
        _mk_node("wte", "token_embedding", "Embedding", num_embeddings=vocab, embedding_dim=hidden),
        _mk_node("wpe", "positional_embedding", "Embedding", num_embeddings=getattr(config, "n_positions", None), embedding_dim=hidden),
    ]
    edges: list[Edge] = [Edge(source_id="wte", target_id="wpe")]
    groups: list[SegmentGroup] = []

    block_prefix = "h.0"
    block_nodes, block_edges = _encoder_block(block_prefix, config, 0, is_last=False)
    if num_layers > 1:
        block_nodes[0].attributes["repeat"] = num_layers
    nodes.extend(block_nodes)
    edges.append(Edge(source_id="wpe", target_id=block_nodes[0].id))
    edges.extend(block_edges)
    groups.append(
        SegmentGroup(
            id="decoder",
            name=f"Decoder × {num_layers}",
            node_ids=[n.id for n in block_nodes],
        )
    )

    nodes.append(_mk_node("ln_f", "final_norm", "LayerNorm", num_features=hidden))
    nodes.append(_mk_node("lm_head", "lm_head", "Linear", out_features=vocab))
    edges.append(Edge(source_id=block_nodes[-1].id, target_id="ln_f"))
    edges.append(Edge(source_id="ln_f", target_id="lm_head"))
    edges.append(Edge(source_id="wte", target_id="lm_head", label="shared", kind="shared"))
    return ModelGraph(nodes=nodes, edges=edges, groups=groups)


def _encoder_decoder(config: Any) -> ModelGraph:
    """T5-family: encoder + decoder + cross-attention."""
    num_enc = getattr(config, "num_layers", None) or getattr(config, "num_hidden_layers", 1) or 1
    num_dec = getattr(config, "num_decoder_layers", None) or num_enc

    nodes: list[LayerNode] = [
        _mk_node("embed", "embedding", "Embedding", num_embeddings=getattr(config, "vocab_size", None), embedding_dim=getattr(config, "d_model", None)),
    ]
    edges: list[Edge] = []
    groups: list[SegmentGroup] = []

    enc_block, enc_edges = _encoder_block("encoder.block.0", config, 0, is_last=False)
    enc_block[0].attributes["repeat"] = num_enc
    nodes.extend(enc_block)
    edges.append(Edge(source_id="embed", target_id=enc_block[0].id))
    edges.extend(enc_edges)
    groups.append(SegmentGroup(id="encoder", name=f"Encoder × {num_enc}", node_ids=[n.id for n in enc_block]))

    dec_block, dec_edges = _encoder_block("decoder.block.0", config, 0, is_last=False)
    dec_block[0].attributes["repeat"] = num_dec
    # Insert a cross-attention node between attn and norm1 in the decoder.
    cross = _mk_node("decoder.block.0.cross_attn", "cross_attn", "MultiHeadAttention", num_heads=getattr(config, "num_heads", None), embed_dim=getattr(config, "d_model", None))
    dec_nodes_with_cross = [dec_block[0], cross, dec_block[1], dec_block[2], dec_block[3]]
    nodes.extend(dec_nodes_with_cross)
    edges.append(Edge(source_id=enc_block[-1].id, target_id=dec_block[0].id))
    edges.extend([
        Edge(source_id=dec_block[0].id, target_id=cross.id),
        Edge(source_id=cross.id, target_id=dec_block[1].id),
        Edge(source_id=dec_block[1].id, target_id=dec_block[2].id),
        Edge(source_id=dec_block[2].id, target_id=dec_block[3].id),
    ])
    edges.append(Edge(source_id=enc_block[-1].id, target_id=cross.id, label="context"))
    groups.append(SegmentGroup(id="decoder", name=f"Decoder × {num_dec}", node_ids=[n.id for n in dec_nodes_with_cross]))

    nodes.append(_mk_node("lm_head", "lm_head", "Linear", out_features=getattr(config, "vocab_size", None)))
    edges.append(Edge(source_id=dec_block[-1].id, target_id="lm_head"))
    return ModelGraph(nodes=nodes, edges=edges, groups=groups)


def _vit(config: Any) -> ModelGraph:
    """Vision Transformer."""
    num_layers = getattr(config, "num_hidden_layers", 1) or 1
    hidden = getattr(config, "hidden_size", None)

    nodes = [
        _mk_node("patch_embed", "patch_embedding", "Conv2d", out_channels=hidden, kernel_size=getattr(config, "patch_size", None)),
        _mk_node("cls_token", "cls_token", "Parameter", embedding_dim=hidden),
        _mk_node("pos_embed", "positional_embedding", "Embedding", embedding_dim=hidden),
    ]
    edges = [
        Edge(source_id="patch_embed", target_id="cls_token"),
        Edge(source_id="cls_token", target_id="pos_embed"),
    ]
    block, block_edges = _encoder_block("encoder.layer.0", config, 0, is_last=False)
    block[0].attributes["repeat"] = num_layers
    nodes.extend(block)
    edges.append(Edge(source_id="pos_embed", target_id=block[0].id))
    edges.extend(block_edges)
    nodes.append(_mk_node("classifier", "classifier", "Linear", out_features=getattr(config, "num_labels", None)))
    edges.append(Edge(source_id=block[-1].id, target_id="classifier"))
    return ModelGraph(
        nodes=nodes,
        edges=edges,
        groups=[SegmentGroup(id="encoder", name=f"Encoder × {num_layers}", node_ids=[n.id for n in block])],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Callable[[Any], ModelGraph]] = {
    # Encoder-only.
    "bert": _generic_transformer,
    "roberta": _generic_transformer,
    "distilbert": _generic_transformer,
    "albert": _generic_transformer,
    # Vision.
    "vit": _vit,
    "clip_vision_model": _vit,
    # Decoder-only.
    "gpt2": _causal_decoder,
    "gpt_neo": _causal_decoder,
    "gptj": _causal_decoder,
    "llama": _causal_decoder,
    "mistral": _causal_decoder,
    # Encoder-decoder.
    "t5": _encoder_decoder,
    "bart": _encoder_decoder,
    "whisper": _encoder_decoder,
    # Generic fallback used when model_type is unknown.
    "generic": _generic_transformer,
}


__all__ = ["build_from_config", "is_config"]
