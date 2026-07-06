"""GGUF inspector — visualize llama.cpp / Ollama model files.

Reads the metadata header from a ``.gguf`` file (Llama, Mistral, Qwen,
Phi, Gemma, etc.) and synthesizes a canonical transformer diagram from
the architecture-specific hyperparameters recorded in the file:
``block_count``, ``embedding_length``, ``feed_forward_length``,
``attention.head_count``, ``vocab_size``, ``context_length``.

No model weights are loaded — we only parse the header, so this is
fast even for a 70 GB Llama-3-70B file.

GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

from modelvision.core.exceptions import InspectionError, mv_warn
from modelvision.core.ir import Edge, LayerNode, ModelGraph, SegmentGroup
from modelvision.inspectors.base import BaseInspector

# GGUF value types — see the spec table.
_GGUF_UINT8 = 0
_GGUF_INT8 = 1
_GGUF_UINT16 = 2
_GGUF_INT16 = 3
_GGUF_UINT32 = 4
_GGUF_INT32 = 5
_GGUF_FLOAT32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT64 = 10
_GGUF_INT64 = 11
_GGUF_FLOAT64 = 12

_MAGIC = b"GGUF"
_MAX_ARRAY_LEN = 16_384  # sanity cap — even the longest tokenizer array fits


class GGUFInspector(BaseInspector):
    framework = "gguf"

    def can_handle(self, model: Any) -> bool:
        if isinstance(model, str):
            return model.lower().endswith(".gguf")
        return False

    def inspect(self, model: Any, **_: Any) -> ModelGraph:
        path = Path(str(model))
        if not path.exists():
            raise InspectionError(f"GGUF file not found: {path}")

        metadata = _read_gguf_header(path)
        arch = metadata.get("general.architecture", "unknown")
        graph = _build_transformer_graph(metadata, arch)
        graph.metadata.update({
            "framework": "gguf",
            "model_class": f"GGUF-{arch}",
            "config_only": True,
            "gguf_metadata": {
                k: v for k, v in metadata.items()
                if not k.startswith(("tokenizer.", "general.file_type"))
                and not isinstance(v, (list, tuple))
            },
        })
        return graph


# ---------------------------------------------------------------------------
# GGUF header parsing
# ---------------------------------------------------------------------------


def _read_gguf_header(path: Path) -> dict[str, Any]:
    """Parse the metadata section of a GGUF file.

    Returns a ``{key: value}`` dict. Only strings, integers, floats, and
    small arrays are decoded — tensor bodies are skipped entirely, so
    reading a 70 GB Llama-3 file is still O(kilobytes).
    """
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise InspectionError(
                f"{path} is not a GGUF file (magic bytes = {magic!r})."
            )
        version = _u32(f)
        if version not in (2, 3):
            mv_warn(f"GGUF version {version} is untested; proceeding anyway.")

        _tensor_count = _u64(f)
        kv_count = _u64(f)

        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            value_type = _u32(f)
            value = _read_gguf_value(f, value_type)
            meta[key] = value
        return meta


def _read_gguf_value(f: BinaryIO, value_type: int) -> Any:
    """Dispatch table for GGUF value types."""
    if value_type == _GGUF_UINT8:
        return f.read(1)[0]
    if value_type == _GGUF_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if value_type == _GGUF_UINT16:
        return _u16(f)
    if value_type == _GGUF_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if value_type == _GGUF_UINT32:
        return _u32(f)
    if value_type == _GGUF_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if value_type == _GGUF_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if value_type == _GGUF_BOOL:
        return bool(f.read(1)[0])
    if value_type == _GGUF_STRING:
        return _read_gguf_string(f)
    if value_type == _GGUF_UINT64:
        return _u64(f)
    if value_type == _GGUF_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if value_type == _GGUF_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    if value_type == _GGUF_ARRAY:
        elem_type = _u32(f)
        length = _u64(f)
        if length > _MAX_ARRAY_LEN:
            # Skip tokenizer / big arrays without decoding them — we
            # don't need them for the diagram.
            _skip_gguf_array(f, elem_type, length)
            return f"<array of {length} elements, skipped>"
        return [_read_gguf_value(f, elem_type) for _ in range(length)]
    raise InspectionError(f"Unknown GGUF value type: {value_type}")


def _skip_gguf_array(f: BinaryIO, elem_type: int, length: int) -> None:
    """Advance ``f`` past ``length`` elements of ``elem_type`` without decoding."""
    fixed_sizes = {
        _GGUF_UINT8: 1, _GGUF_INT8: 1,
        _GGUF_UINT16: 2, _GGUF_INT16: 2,
        _GGUF_UINT32: 4, _GGUF_INT32: 4, _GGUF_FLOAT32: 4, _GGUF_BOOL: 1,
        _GGUF_UINT64: 8, _GGUF_INT64: 8, _GGUF_FLOAT64: 8,
    }
    if elem_type in fixed_sizes:
        f.seek(fixed_sizes[elem_type] * length, 1)
        return
    if elem_type == _GGUF_STRING:
        for _ in range(length):
            slen = _u64(f)
            f.seek(slen, 1)
        return
    # Nested arrays / unknown — bail; the caller will report the array
    # as skipped.
    raise InspectionError(f"Can't skip GGUF array of element type {elem_type}")


def _read_gguf_string(f: BinaryIO) -> str:
    length = _u64(f)
    return f.read(length).decode("utf-8", errors="replace")


def _u16(f: BinaryIO) -> int:
    return struct.unpack("<H", f.read(2))[0]


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", f.read(8))[0]


# ---------------------------------------------------------------------------
# Metadata → ModelGraph
# ---------------------------------------------------------------------------


def _build_transformer_graph(meta: dict[str, Any], arch: str) -> ModelGraph:
    """Synthesize a canonical decoder-only transformer diagram from GGUF metadata.

    The block layout matches Llama / Mistral / Qwen / Phi / Gemma:

        embed_tokens → RMSNorm → [Attn → RMSNorm → MLP → RMSNorm] × N → lm_head

    We show one representative block explicitly and mark it ``× N`` so
    the diagram stays compact even for 70-layer models. The layout
    engine's segment-group folding does the rest.
    """
    def m(key: str, default: Any = None) -> Any:
        # Try arch-prefixed key first, then generic.
        return meta.get(f"{arch}.{key}", meta.get(key, default))

    n_layer = m("block_count") or 1
    hidden_size = m("embedding_length")
    ffn_size = m("feed_forward_length")
    n_heads = m("attention.head_count")
    n_kv_heads = m("attention.head_count_kv") or n_heads
    context_len = m("context_length")
    vocab_size = m("vocab_size")
    if vocab_size is None:
        # Try inferring from the tokenizer model list length.
        vocab_size = _lookup_vocab_size(meta)

    def _node(nid: str, name: str, layer_type: str, **attrs: Any) -> LayerNode:
        return LayerNode(
            id=nid, name=name, layer_type=layer_type, framework="gguf",
            attributes={k: v for k, v in attrs.items() if v is not None},
        )

    nodes: list[LayerNode] = [
        _node("embed_tokens", "embed_tokens", "Embedding",
              num_embeddings=vocab_size, embedding_dim=hidden_size),
        _node("input_norm", "input_norm", "LayerNorm", num_features=hidden_size),
    ]
    edges: list[Edge] = [
        Edge(source_id="embed_tokens", target_id="input_norm"),
    ]

    # One representative block, marked with the repeat count.
    block_prefix = "block.0"
    block_ids = [
        f"{block_prefix}.attn",
        f"{block_prefix}.attn_norm",
        f"{block_prefix}.ffn",
        f"{block_prefix}.ffn_norm",
    ]
    block_nodes = [
        _node(block_ids[0], "self_attn", "MultiHeadAttention",
              num_heads=n_heads, num_kv_heads=n_kv_heads, embed_dim=hidden_size,
              context_length=context_len),
        _node(block_ids[1], "attn_norm", "LayerNorm", num_features=hidden_size),
        _node(block_ids[2], "feed_forward", "MLP",
              intermediate_size=ffn_size, hidden_size=hidden_size),
        _node(block_ids[3], "ffn_norm", "LayerNorm", num_features=hidden_size),
    ]
    if n_layer > 1:
        # The renderer draws a "× N" badge whenever a node has this attr.
        block_nodes[0].attributes["repeat"] = n_layer

    nodes.extend(block_nodes)
    edges.append(Edge(source_id="input_norm", target_id=block_ids[0]))
    edges.append(Edge(source_id=block_ids[0], target_id=block_ids[1]))
    edges.append(Edge(source_id=block_ids[1], target_id=block_ids[2]))
    edges.append(Edge(source_id=block_ids[2], target_id=block_ids[3]))

    # Final projection to vocab.
    nodes.append(_node("lm_head", "lm_head", "Linear",
                       in_features=hidden_size, out_features=vocab_size))
    edges.append(Edge(source_id=block_ids[3], target_id="lm_head"))
    # Weight tying — most decoder-only LMs share embed_tokens and lm_head.
    edges.append(
        Edge(source_id="embed_tokens", target_id="lm_head", label="shared", kind="shared")
    )

    groups = [
        SegmentGroup(
            id="decoder",
            name=f"Decoder × {n_layer}",
            node_ids=block_ids,
        ),
    ]

    return ModelGraph(nodes=nodes, edges=edges, groups=groups)


def _lookup_vocab_size(meta: dict[str, Any]) -> int | None:
    """Read vocab size from tokenizer metadata when the top-level key is missing."""
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"):
        v = meta.get(key)
        if isinstance(v, str) and v.startswith("<array of "):
            # We stored the skipped-array note there — parse the count.
            try:
                return int(v.split()[2])
            except (ValueError, IndexError):
                pass
        if isinstance(v, list):
            return len(v)
    return None


__all__ = ["GGUFInspector"]
