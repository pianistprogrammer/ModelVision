"""Example 07 — ONNX universal fallback.

ONNX is the recommended path for models exported from any framework — or
models whose framework isn't natively supported. This example generates
a tiny ONNX file from scratch (so it runs without downloading anything)
and renders it in every theme.

Requires the ``onnx`` extra::

    uv add "modelvision[onnx]"

Run::

    python examples/07_onnx.py
"""

from __future__ import annotations

import modelvision as mv


def build_tiny_onnx(path: str) -> None:
    """A synthetic classifier: Conv → BN → ReLU → Pool → Flatten → Gemm."""
    import onnx  # type: ignore[import-not-found]
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])

    # Weights — all zeros, we're never running this.
    conv_w = helper.make_tensor(
        "conv_w", TensorProto.FLOAT, [16, 3, 3, 3], [0.0] * (16 * 3 * 3 * 3)
    )
    bn_scale = helper.make_tensor("bn_scale", TensorProto.FLOAT, [16], [1.0] * 16)
    bn_bias = helper.make_tensor("bn_bias", TensorProto.FLOAT, [16], [0.0] * 16)
    bn_mean = helper.make_tensor("bn_mean", TensorProto.FLOAT, [16], [0.0] * 16)
    bn_var = helper.make_tensor("bn_var", TensorProto.FLOAT, [16], [1.0] * 16)
    fc_w = helper.make_tensor("fc_w", TensorProto.FLOAT, [10, 16], [0.0] * (10 * 16))
    fc_b = helper.make_tensor("fc_b", TensorProto.FLOAT, [10], [0.0] * 10)

    nodes = [
        helper.make_node(
            "Conv",
            ["x", "conv_w"],
            ["conv_out"],
            name="conv1",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["bn_out"],
            name="bn1",
        ),
        helper.make_node("Relu", ["bn_out"], ["relu_out"], name="relu1"),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["gap_out"], name="gap"),
        helper.make_node("Flatten", ["gap_out"], ["flat_out"], name="flatten"),
        helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["y"], name="classifier"),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_classifier",
        [x],
        [y],
        [conv_w, bn_scale, bn_bias, bn_mean, bn_var, fc_w, fc_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)


def main() -> None:
    build_tiny_onnx("07_tiny.onnx")

    # 1. Basic render — auto-detects ONNX from the file extension.
    mv.render("07_tiny.onnx", "07_onnx_light.svg", theme="light")

    # 2. Interactive HTML — click any node to see its ONNX attributes.
    mv.render("07_tiny.onnx", "07_onnx.html", theme="dark")

    # 3. Style ONNX op types just like framework layers — the inspector
    #    remaps ONNX names to canonical types (Conv → Conv2d, Gemm → Linear).
    mv.render(
        "07_tiny.onnx",
        "07_onnx_styled.svg",
        theme="pastel",
        layer_palette={
            "Conv2d": "#3b82f6",
            "BatchNorm2d": "#f59e0b",
            "ReLU": "#22c55e",
            "AdaptiveAvgPool2d": "#ef4444",
            "Flatten": "#a1a1aa",
            "Linear": "#8b5cf6",
        },
    )
    print("wrote 07_tiny.onnx + 07_onnx_light.svg + 07_onnx.html + 07_onnx_styled.svg")


if __name__ == "__main__":
    main()
