"""Example 08 — Handling tricky architectures.

Weight tying (shared params), ModuleList unrolling, DataParallel
wrappers, quantized layers, and cross-scope skip connections all work
without special-casing on your side.

Run::

    python examples/08_tricky.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

import modelvision as mv

# ---------------------------------------------------------------------------
# 1. Weight tying (e.g. tied input/output embeddings in language models).
# ---------------------------------------------------------------------------


class TiedEmbeddingLM(nn.Module):
    def __init__(self, vocab: int = 100, dim: int = 16):
        super().__init__()
        shared = nn.Linear(dim, vocab, bias=False)
        self.embed = nn.Embedding(vocab, dim)
        # Both projections share the same weight matrix. ModelVision
        # detects this by ``id()`` and draws a dashed "shared" edge.
        self.lm_head = shared
        self.aux_head = shared

    def forward(self, x):
        h = self.embed(x)
        return self.lm_head(h), self.aux_head(h)


# ---------------------------------------------------------------------------
# 2. ModuleList / ModuleDict unrolling.
# ---------------------------------------------------------------------------


class ExpertMixture(nn.Module):
    def __init__(self, num_experts: int = 4, dim: int = 8):
        super().__init__()
        # Both containers unroll into indexed nodes: experts.0, experts.1, etc.
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])
        self.gates = nn.ModuleDict(
            {name: nn.Linear(dim, 1) for name in ("router", "auxiliary", "load_balance")}
        )


# ---------------------------------------------------------------------------
# 3. Residual / skip connections — the merge node appears automatically
#    when the inspector sees fan-in.
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        h = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        return self.act(x + h)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Weight tying.
    # ------------------------------------------------------------------
    tied = TiedEmbeddingLM()
    graph = mv.inspect(tied)
    shared = [e for e in graph.edges if e.kind == "shared"]
    print(f"Tied model: {len(shared)} shared-weight edge(s) detected")
    mv.render(tied, "08_tied.svg", theme="dark")

    # ------------------------------------------------------------------
    # 2. ModuleList / ModuleDict.
    # ------------------------------------------------------------------
    experts = ExpertMixture()
    g = mv.inspect(experts)
    print(f"ExpertMixture: {[n.id for n in g.nodes]}")
    mv.render(experts, "08_experts.svg", theme="pastel", layout="horizontal")

    # ------------------------------------------------------------------
    # 3. ResBlock — merge node auto-inserted when torch.fx catches the skip.
    # ------------------------------------------------------------------
    #    Without symbolic_shapes=True we get sequential edges within each
    #    parent scope. With symbolic_shapes=True the cross-scope + edge
    #    from x → x+h is captured and post-processed into a "+" node.
    mv.render(ResBlock(), "08_resblock.svg", symbolic_shapes=True, theme="light")

    # ------------------------------------------------------------------
    # 4. DataParallel wrapper — ModelVision unwraps it before inspection.
    # ------------------------------------------------------------------
    wrapped = torch.nn.DataParallel(nn.Sequential(nn.Linear(4, 4), nn.ReLU()))
    g = mv.inspect(wrapped)
    print(f"DataParallel-wrapped: {[n.id for n in g.nodes]} (wrapper is gone)")

    # ------------------------------------------------------------------
    # 5. torch.compile wrapper — same story.
    # ------------------------------------------------------------------
    try:
        compiled = torch.compile(nn.Linear(8, 8))
        g = mv.inspect(compiled)
        print(f"torch.compile'd: {len(g.nodes)} node(s), no OptimizedModule leaking")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    print("wrote 08_tied.svg, 08_experts.svg, 08_resblock.svg")


if __name__ == "__main__":
    main()
