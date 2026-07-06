"""Deterministic node-ID minting.

Node IDs are dotted paths that mirror the framework's own module
hierarchy (``features.0.conv``, ``encoder.layer.3.attention.self``).
Determinism matters because these IDs are the *primary key* users hand
back to :attr:`Group.nodes` and :attr:`node_styles` — and they anchor
golden-file SVG regression tests.
"""

from __future__ import annotations

import re
from collections import Counter


def sanitize(part: str) -> str:
    """Coerce an arbitrary name into an ID-safe segment.

    Preserves alphanumerics, ``_`` and ``-``; replaces everything else
    with ``_``. Empty input becomes ``"_"``.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", part).strip("_") or "_"
    return cleaned


def join(*parts: str) -> str:
    """Join sanitized parts with ``.`` separators, skipping empty parts."""
    return ".".join(sanitize(p) for p in parts if p != "")


def uniquify(ids: list[str]) -> list[str]:
    """Given a list of proposed IDs, append ``_2``/``_3``/... to collisions.

    Stable: first occurrence keeps its original ID; later duplicates are
    suffixed in encounter order.
    """
    counts: Counter[str] = Counter()
    out: list[str] = []
    for i in ids:
        counts[i] += 1
        out.append(i if counts[i] == 1 else f"{i}_{counts[i]}")
    return out
