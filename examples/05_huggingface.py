"""Example 05 — HuggingFace Transformers, both config-only and instance paths.

Requires the ``huggingface`` extra::

    uv add "modelvision[huggingface,torch]"

Run::

    python examples/05_huggingface.py
"""

from __future__ import annotations

import modelvision as mv


def main() -> None:
    from transformers import (  # type: ignore[import-not-found]
        BertConfig,
        GPT2Config,
        T5Config,
        ViTConfig,
    )

    # -----------------------------------------------------------------
    # 1. Config-only rendering — no weights, no download, no forward pass.
    # -----------------------------------------------------------------
    mv.render(
        BertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, vocab_size=30522),
        "05_bert.svg",
        theme="pastel",
        title="BERT-base",
    )

    mv.render(
        GPT2Config(n_embd=768, n_layer=12, n_head=12, n_positions=1024, vocab_size=50257),
        "05_gpt2.svg",
        theme="dark",
        title="GPT-2 small",
    )

    mv.render(
        T5Config(d_model=512, num_layers=6, num_decoder_layers=6, num_heads=8, vocab_size=32128),
        "05_t5.svg",
        theme="light",
        title="T5-small",
        layout="horizontal",
    )

    mv.render(
        ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_labels=1000),
        "05_vit.svg",
        theme="high_contrast",
        title="ViT-base",
    )
    print("wrote all config-only 05_*.svg files")

    # -----------------------------------------------------------------
    # 2. Model-instance path — delegates to torch, folds repeated blocks.
    # -----------------------------------------------------------------
    try:
        from transformers import BertModel  # type: ignore[import-not-found]

        # Tiny BertModel — no download.
        model = BertModel(
            BertConfig(
                hidden_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=128,
                vocab_size=200,
            )
        )

        # Default: encoder layers folded into a "× 4" block.
        mv.render(model, "05_bert_model_folded.svg", theme="pastel")
        # Full expansion — every leaf layer visible.
        mv.render(model, "05_bert_model_expanded.svg", theme="pastel", expand_groups=True)
        print("wrote 05_bert_model_folded.svg and 05_bert_model_expanded.svg")
    except Exception as e:
        print(f"model-instance path skipped: {e}")


if __name__ == "__main__":
    main()
