#!/usr/bin/env python
"""Write two tiny ONNX graphs with the input and output shapes of the real recognisers.

``tiny_allosaurus.onnx`` maps ``feats`` ``(1, T, 120)`` to ``logits`` ``(1, T, 230)`` by
one matrix product, and ``tiny_wav2vec2.onnx`` maps ``input_values`` ``(1, N)`` to
``logits`` ``(1, T, 392)`` by one strided convolution. Their weights come from
:func:`weights`, so a test can compute the expected logits in numpy and check the whole
backend path (resampling, frontend, session feed, CTC decoding, unit mapping) without
the real models.

Usage, in an environment with ``onnx`` installed::

    python generate_tiny_backends.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KERNEL, STRIDE = 80, 320


def weights(name: str) -> np.ndarray:
    """The weights of a tiny backend graph, from a fixed seed."""
    rng = np.random.default_rng(20261009)
    if name == "allosaurus":
        return rng.standard_normal((120, 230)).astype(np.float32)
    w = rng.standard_normal((392, 1, KERNEL)).astype(np.float32)
    return w


def _model(helper, graph):
    """Opset 17 at IR version 8, which every onnxruntime the package supports loads."""
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def main() -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(weights("allosaurus"), "W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["feats", "W"], ["logits"])],
        "tiny_allosaurus",
        [helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "T", 120])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "T", 230])],
        [w],
    )
    onnx.save(_model(helper, graph), HERE / "tiny_allosaurus.onnx")

    w = numpy_helper.from_array(weights("wav2vec2"), "W")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    graph = helper.make_graph(
        [
            helper.make_node("Unsqueeze", ["input_values", "axes"], ["x"]),
            helper.make_node("Conv", ["x", "W"], ["y"], strides=[STRIDE]),
            helper.make_node("Transpose", ["y"], ["logits"], perm=[0, 2, 1]),
        ],
        "tiny_wav2vec2",
        [helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, "N"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "T", 392])],
        [w, axes],
    )
    onnx.save(_model(helper, graph), HERE / "tiny_wav2vec2.onnx")


if __name__ == "__main__":
    main()
