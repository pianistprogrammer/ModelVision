"""Gallery — the same tiny CNN in every framework ModelVision supports.

Renders the same functionally-equivalent model in six frameworks — PyTorch,
TensorFlow/Keras, JAX/Flax, HuggingFace, scikit-learn, ONNX — so you can
see ModelVision handles them all with a consistent visual look.

Each framework's builder runs in its own subprocess. That isolation
matters: some framework wheels on some machines (a bad jax build on
macOS, for example) can segfault at import time. Sub-processing means one
crash can't take down the others.

Run::

    python examples/13_all_frameworks.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(".").resolve()
REPO_ROOT = Path(__file__).resolve().parent.parent


# A palette that reads well in both light and dark themes.
PALETTE = {
    "Conv2d": "#fff59d",
    "Conv": "#fff59d",
    "Conv2D": "#fff59d",
    "BatchNorm2d": "#c8e6c9",
    "BatchNormalization": "#c8e6c9",
    "BatchNorm": "#c8e6c9",
    "ReLU": "#fff59d",
    "Relu": "#fff59d",
    "MaxPool2d": "#ce93d8",
    "MaxPool": "#ce93d8",
    "MaxPooling2D": "#ce93d8",
    "Linear": "#bbdefb",
    "Dense": "#bbdefb",
    "Gemm": "#bbdefb",
    "Dropout": "#e0e0e0",
    "Flatten": "#bbdefb",
    "MultiHeadAttention": "#f8bbd0",
    "MultiheadAttention": "#f8bbd0",
    "LayerNorm": "#ffe0b2",
    "MLP": "#bbdefb",
    "Embedding": "#d1c4e9",
    "StandardScaler": "#c8e6c9",
    "PCA": "#f8bbd0",
    "LogisticRegression": "#bbdefb",
    "Input": "#a5d6a7",
    "*": "#e0e0e0",
}


# ---------------------------------------------------------------------------
# Per-framework builder scripts — each written to a temp .py file and run
# in its own subprocess.
# ---------------------------------------------------------------------------


_PRELUDE = f"""
import json, warnings, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
warnings.simplefilter("ignore")
import modelvision as mv

OUT = {str(OUT_DIR)!r}
PALETTE = {PALETTE!r}
# No legend — the palette + shape subtitles are self-explanatory. Labels
# render BELOW each block per user's HTML reference.
COMMON = dict(theme="light", layer_palette=PALETTE, legend=False,
              layout="vertical", overwrite=True)
"""


TORCH_SCRIPT = (
    _PRELUDE
    + """
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 10),
)
kwargs = dict(COMMON)
kwargs["layout"] = "flow"
kwargs["input_shape"] = (1, 3, 32, 32)
mv.render(model, f"{OUT}/13_torch.svg", title="PyTorch", **kwargs)
print("OK")
"""
)


KERAS_SCRIPT = (
    _PRELUDE
    + """
try:
    import keras
except ImportError:
    from tensorflow import keras
model = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(16, 3, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])
kwargs = dict(COMMON)
kwargs["layout"] = "flow"
kwargs["input_shape"] = (1, 32, 32, 3)
mv.render(model, f"{OUT}/13_keras.svg", title="TensorFlow / Keras", **kwargs)
print("OK")
"""
)


JAX_SCRIPT = (
    _PRELUDE
    + """
import flax.linen as nn
import jax
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(features=10)(x)

kwargs = dict(COMMON)
kwargs["layout"] = "flow"
kwargs["input_shape"] = (1, 32, 32, 3)
mv.render(SimpleCNN(), f"{OUT}/13_jax.svg",
          title="JAX / Flax", **kwargs)
print("OK")
"""
)


HF_SCRIPT = (
    _PRELUDE
    + """
from transformers import BertConfig
config = BertConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                    intermediate_size=128, vocab_size=1000)
mv.render(config, f"{OUT}/13_huggingface.svg",
          title="HuggingFace (BERT-mini)", **COMMON)
print("OK")
"""
)


SKLEARN_SCRIPT = (
    _PRELUDE
    + """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=16)),
    ("clf", LogisticRegression(max_iter=1000)),
])
mv.render(pipe, f"{OUT}/13_sklearn.svg", title="scikit-learn", **COMMON)
print("OK")
"""
)


ONNX_SCRIPT = (
    _PRELUDE
    + """
import onnx
from onnx import TensorProto, helper
def W(name, shape):
    n = 1
    for d in shape: n *= d
    return helper.make_tensor(name, TensorProto.FLOAT, shape, [0.0] * n)
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])
weights = [
    W("c1_w",[16,3,3,3]), W("c1_b",[16]),
    W("bn1_s",[16]), W("bn1_b",[16]), W("bn1_m",[16]), W("bn1_v",[16]),
    W("c2_w",[32,16,3,3]), W("c2_b",[32]),
    W("bn2_s",[32]), W("bn2_b",[32]), W("bn2_m",[32]), W("bn2_v",[32]),
    W("fc_w",[10, 32*8*8]), W("fc_b",[10]),
]
nodes = [
    helper.make_node("Conv", ["x","c1_w","c1_b"], ["c1"], name="conv1",
                     kernel_shape=[3,3], pads=[1,1,1,1]),
    helper.make_node("BatchNormalization",
                     ["c1","bn1_s","bn1_b","bn1_m","bn1_v"], ["bn1"], name="bn1"),
    helper.make_node("Relu", ["bn1"], ["r1"], name="relu1"),
    helper.make_node("MaxPool", ["r1"], ["p1"], name="pool1",
                     kernel_shape=[2,2], strides=[2,2]),
    helper.make_node("Conv", ["p1","c2_w","c2_b"], ["c2"], name="conv2",
                     kernel_shape=[3,3], pads=[1,1,1,1]),
    helper.make_node("BatchNormalization",
                     ["c2","bn2_s","bn2_b","bn2_m","bn2_v"], ["bn2"], name="bn2"),
    helper.make_node("Relu", ["bn2"], ["r2"], name="relu2"),
    helper.make_node("MaxPool", ["r2"], ["p2"], name="pool2",
                     kernel_shape=[2,2], strides=[2,2]),
    helper.make_node("Flatten", ["p2"], ["f"], name="flatten"),
    helper.make_node("Gemm", ["f","fc_w","fc_b"], ["y"], name="fc"),
]
graph = helper.make_graph(nodes, "simple_cnn", [x], [y], weights)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.save(model, f"{OUT}/13_simple_cnn.onnx")
kwargs = dict(COMMON)
kwargs["layout"] = "flow"
mv.render(f"{OUT}/13_simple_cnn.onnx", f"{OUT}/13_onnx.svg", title="ONNX", **kwargs)
print("OK")
"""
)


BUILDERS = [
    ("torch", TORCH_SCRIPT),
    ("keras", KERAS_SCRIPT),
    ("jax", JAX_SCRIPT),
    ("huggingface", HF_SCRIPT),
    ("sklearn", SKLEARN_SCRIPT),
    ("onnx", ONNX_SCRIPT),
]


def _run(name: str, script: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        # A negative exit code = killed by a signal (e.g. SIGABRT/SIGSEGV
        # from a broken framework wheel). Report and continue.
        stderr_tail = "\\n".join(proc.stderr.strip().splitlines()[-3:])
        return f"crashed (exit {proc.returncode}): {stderr_tail or 'no output'}"
    if "OK" in proc.stdout:
        return "ok"
    return f"unexpected output: {proc.stdout.strip()[:80]}"


def main() -> None:
    results: list[tuple[str, str]] = []
    for name, script in BUILDERS:
        status = _run(name, script)
        results.append((name, status))
        marker = "✓" if status == "ok" else "✗"
        print(f"{marker} {name:14s}  {status}")

    print()
    print("Outputs in:", OUT_DIR)
    print("Files:")
    for name, status in results:
        if status == "ok":
            print(f"  13_{name}.svg")


if __name__ == "__main__":
    main()
