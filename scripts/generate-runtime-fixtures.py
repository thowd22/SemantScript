"""Regenerate the deterministic ONNX models used by the Node runtime tests.

This is a development-only provenance tool. Runtime tests consume the committed
models and never invoke Python.
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

FIXTURES = Path(__file__).resolve().parent.parent / "runtime" / "test" / "fixtures"
OPSET = 17


def save(name: str, graph: onnx.GraphProto) -> None:
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="semantscript-test-fixture",
    )
    onnx.checker.check_model(model)
    first = model.SerializeToString(deterministic=True)
    second = model.SerializeToString(deterministic=True)
    if first != second:
        raise RuntimeError(f"{name} did not serialize deterministically")
    (FIXTURES / name).write_bytes(first)


def encoder(batch: str | int = "BATCH") -> onnx.GraphProto:
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch, "SEQUENCE"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [batch, "SEQUENCE"]),
    ]
    output = helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, [batch, 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_float"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["ids_float", "mask_float"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axes"], ["sentence_embedding"], keepdims=1),
    ]
    return helper.make_graph(nodes, "fixture-encoder", inputs, [output], [axes])


def quantizable_encoder(scale: float = 100.0) -> onnx.GraphProto:
    """The fixture encoder followed by a MatMul with a constant weight.

    Dynamic quantization converts a MatMul with a constant operand, which the
    plain fixture encoder lacks, so this is the graph `releases derive --int8`
    runs on in the tests and the CI package job. The weight is positive, so
    every answer is the plain encoder's (the head's third support value); it
    only sharpens the head's confidence so an uncalibrated fixture still
    passes the ECE gate on fixture-labelled records. A weight of 100 and the
    masked sum both quantize exactly, so the int8 graph decides as the
    float32 one does.
    """

    graph = encoder()
    graph.name = "fixture-quantizable-encoder"
    graph.node[-1].output[0] = "summed"
    graph.initializer.append(helper.make_tensor("projection", TensorProto.FLOAT, [1, 1], [scale]))
    graph.node.append(helper.make_node("MatMul", ["summed", "projection"], ["sentence_embedding"]))
    return graph


def adapter(batch: str | int = "BATCH", hidden: str | int = 1) -> onnx.GraphProto:
    input_value = helper.make_tensor_value_info(
        "sentence_embedding", TensorProto.FLOAT, [batch, hidden]
    )
    output_value = helper.make_tensor_value_info(
        "function_embedding", TensorProto.FLOAT, [batch, hidden]
    )
    return helper.make_graph(
        [helper.make_node("Identity", ["sentence_embedding"], ["function_embedding"])],
        "fixture-adapter",
        [input_value],
        [output_value],
    )


def head(output_width: int = 3, batch: str | int = "BATCH") -> onnx.GraphProto:
    input_value = helper.make_tensor_value_info("function_embedding", TensorProto.FLOAT, [batch, 1])
    output_value = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, output_width])
    weights = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [1, output_width],
        [index / 100 for index in range(output_width)],
    )
    bias_values = [float(index) - 1.0 for index in range(output_width)]
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [output_width], bias_values)
    return helper.make_graph(
        [helper.make_node("Gemm", ["function_embedding", "weights", "bias"], ["logits"])],
        "fixture-head",
        [input_value],
        [output_value],
        [weights, bias],
    )


save("encoder.onnx", encoder())
save("adapter.onnx", adapter())
save("head.onnx", head())
save("wrong-head.onnx", head(4))
save("fixed-batch-encoder.onnx", encoder(1))
save("fixed-batch-adapter.onnx", adapter(1))
save("fixed-batch-head.onnx", head(batch=1))
save("symbolic-hidden-adapter.onnx", adapter(hidden="HIDDEN"))
save("wide-adapter.onnx", adapter(hidden=2))
save("quantizable-encoder.onnx", quantizable_encoder())
