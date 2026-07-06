# HuggingFace

## From a config (no weights needed)

```python
from transformers import BertConfig
import modelvision as mv

mv.render(BertConfig.from_pretrained("bert-base-uncased"), output="bert.svg")
```

Config-only rendering doesn't load any weights — perfect for
architecture papers or blog posts where you want the canonical diagram
without downloading a checkpoint.

## From a loaded model

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")
mv.render(model, output="bert.svg", expand_groups=False)
```

Repeated encoder / decoder layers are folded into a single block with
a `× N` badge by default. Pass `expand_groups=True` to render every
layer.

## Supported architectures

Shipped templates cover BERT, RoBERTa, DistilBERT, ALBERT, GPT-2,
GPT-Neo, GPT-J, LLaMA, Mistral, T5, BART, Whisper, ViT, and CLIP. The
generic-transformer fallback works for any `PretrainedConfig` that
exposes `num_hidden_layers`, `hidden_size`, and `num_attention_heads`.
