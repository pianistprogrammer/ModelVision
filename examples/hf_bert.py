"""BERT rendered from a HuggingFace config alone (no weights loaded)."""

from __future__ import annotations

from transformers import BertConfig

import modelvision as mv


def main() -> None:
    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=30522,
    )
    mv.render(config, output="bert.svg", theme="pastel")


if __name__ == "__main__":
    main()
