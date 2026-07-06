"""Tests for the GGUF inspector.

We don't check any real ``.gguf`` file into the repo — instead we
synthesize a minimal one on the fly using the format spec.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from modelvision import inspect
from modelvision.core.exceptions import InspectionError


def _write_gguf(
    path: Path,
    metadata: dict[str, object],
) -> None:
    """Write a minimal GGUF file with only the metadata we ask for.

    Enough to satisfy our inspector's parser; tensor_count is 0 so we
    don't need to write any tensor data.
    """
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)              # version
    buf += struct.pack("<Q", 0)              # tensor_count
    buf += struct.pack("<Q", len(metadata))  # kv_count
    for key, value in metadata.items():
        _write_string(buf, key)
        if isinstance(value, str):
            buf += struct.pack("<I", 8)  # GGUF_STRING
            _write_string(buf, value)
        elif isinstance(value, bool):
            buf += struct.pack("<I", 7)  # GGUF_BOOL
            buf += bytes([1 if value else 0])
        elif isinstance(value, int):
            buf += struct.pack("<I", 4)  # GGUF_UINT32
            buf += struct.pack("<I", value)
        elif isinstance(value, float):
            buf += struct.pack("<I", 6)  # GGUF_FLOAT32
            buf += struct.pack("<f", value)
        else:
            raise TypeError(f"Unsupported metadata value type: {type(value)}")

    path.write_bytes(bytes(buf))


def _write_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def test_gguf_inspector_parses_llama_metadata(tmp_path: Path) -> None:
    gguf = tmp_path / "test.gguf"
    _write_gguf(gguf, {
        "general.architecture": "llama",
        "general.name": "TestModel",
        "llama.block_count": 12,
        "llama.embedding_length": 256,
        "llama.feed_forward_length": 512,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 4,
        "llama.context_length": 1024,
        "llama.vocab_size": 32000,
    })
    graph = inspect(str(gguf))

    assert graph.metadata["framework"] == "gguf"
    assert graph.metadata["config_only"] is True
    # embed_tokens + input_norm + 4 block nodes + lm_head = 7.
    assert len(graph.nodes) == 7

    attn = next(n for n in graph.nodes if n.layer_type == "MultiHeadAttention")
    assert attn.attributes["num_heads"] == 8
    assert attn.attributes["num_kv_heads"] == 4
    assert attn.attributes["repeat"] == 12
    assert attn.attributes["context_length"] == 1024

    embed = next(n for n in graph.nodes if n.layer_type == "Embedding")
    assert embed.attributes["num_embeddings"] == 32000
    assert embed.attributes["embedding_dim"] == 256

    # Weight-tied lm_head + embed_tokens.
    shared_edges = [e for e in graph.edges if e.kind == "shared"]
    assert len(shared_edges) == 1
    assert (shared_edges[0].source_id, shared_edges[0].target_id) == (
        "embed_tokens", "lm_head",
    )


def test_gguf_bad_magic_raises(tmp_path: Path) -> None:
    bad = tmp_path / "notgguf.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(InspectionError, match="not a GGUF"):
        inspect(str(bad))


def test_gguf_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InspectionError, match="not found"):
        inspect(str(tmp_path / "does-not-exist.gguf"))


def test_gguf_auto_detect_from_suffix() -> None:
    from modelvision.inspectors.auto import detect_framework

    assert detect_framework("model.gguf") == "gguf"
    assert detect_framework("dir/subdir/Model.GGUF") == "gguf"
