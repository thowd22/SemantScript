/**
 * A minimal, deterministic ONNX protobuf writer for the testing stub's three
 * graphs (TASK-14.8). It writes exactly the fields `onnx.helper` writes for the
 * runtime's test fixtures, in field-number order, so the stub needs no model
 * files and no Python: an encoder that sums masked token ids to [BATCH,1], an
 * identity adapter and a linear head to [BATCH,K]. The stub never reads the
 * head's logits; the graphs only have to run so the real worker, request
 * scopes and pass counts are exercised.
 */

const ONNX_IR_VERSION = 8;
export const STUB_ONNX_OPSET = 17;
const FLOAT = 1;
const INT64 = 7;
const ATTRIBUTE_INT = 2;

const utf8 = new TextEncoder();

class ProtoWriter {
  readonly #chunks: number[] = [];

  varint(field: number, value: number): this {
    this.#key(field, 0);
    this.#varint(value);
    return this;
  }

  bytes(field: number, value: Uint8Array): this {
    this.#key(field, 2);
    this.#varint(value.length);
    for (const byte of value) this.#chunks.push(byte);
    return this;
  }

  string(field: number, value: string): this {
    return this.bytes(field, utf8.encode(value));
  }

  message(field: number, value: ProtoWriter): this {
    return this.bytes(field, value.finish());
  }

  finish(): Uint8Array {
    return Uint8Array.from(this.#chunks);
  }

  #key(field: number, wireType: number): void {
    this.#varint(field * 8 + wireType);
  }

  #varint(value: number): void {
    pushVarint(this.#chunks, value);
  }
}

function pushVarint(target: number[], value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError("protobuf varints here are non-negative integers");
  }
  let remaining = value;
  while (remaining >= 0x80) {
    target.push((remaining % 0x80) | 0x80);
    remaining = Math.floor(remaining / 0x80);
  }
  target.push(remaining);
}

function packedVarints(values: readonly number[]): Uint8Array {
  const bytes: number[] = [];
  for (const value of values) pushVarint(bytes, value);
  return Uint8Array.from(bytes);
}

function packedFloats(values: readonly number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (const [index, value] of values.entries()) {
    view.setFloat32(index * 4, value, true);
  }
  return bytes;
}

type Dimension = string | number;

function valueInfo(
  name: string,
  elementType: number,
  shape: readonly Dimension[],
): ProtoWriter {
  const dims = new ProtoWriter();
  for (const dimension of shape) {
    dims.message(
      1,
      typeof dimension === "string"
        ? new ProtoWriter().string(2, dimension)
        : new ProtoWriter().varint(1, dimension),
    );
  }
  const tensorType = new ProtoWriter().varint(1, elementType).message(2, dims);
  return new ProtoWriter()
    .string(1, name)
    .message(2, new ProtoWriter().message(1, tensorType));
}

interface NodeSpec {
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly opType: string;
  readonly intAttributes?: readonly (readonly [string, number])[];
}

function node(spec: NodeSpec): ProtoWriter {
  const writer = new ProtoWriter();
  for (const input of spec.inputs) writer.string(1, input);
  for (const output of spec.outputs) writer.string(2, output);
  writer.string(4, spec.opType);
  for (const [name, value] of spec.intAttributes ?? []) {
    writer.message(
      5,
      new ProtoWriter()
        .string(1, name)
        .varint(3, value)
        .varint(20, ATTRIBUTE_INT),
    );
  }
  return writer;
}

function floatTensor(
  name: string,
  dims: readonly number[],
  values: readonly number[],
): ProtoWriter {
  const writer = new ProtoWriter();
  for (const dim of dims) writer.varint(1, dim);
  return writer.varint(2, FLOAT).bytes(4, packedFloats(values)).string(8, name);
}

function int64Tensor(
  name: string,
  dims: readonly number[],
  values: readonly number[],
): ProtoWriter {
  const writer = new ProtoWriter();
  for (const dim of dims) writer.varint(1, dim);
  return writer
    .varint(2, INT64)
    .bytes(7, packedVarints(values))
    .string(8, name);
}

interface GraphSpec {
  readonly name: string;
  readonly nodes: readonly NodeSpec[];
  readonly initializers: readonly ProtoWriter[];
  readonly inputs: readonly ProtoWriter[];
  readonly outputs: readonly ProtoWriter[];
}

function model(producer: string, graph: GraphSpec): Uint8Array {
  const graphWriter = new ProtoWriter();
  for (const spec of graph.nodes) graphWriter.message(1, node(spec));
  graphWriter.string(2, graph.name);
  for (const initializer of graph.initializers) {
    graphWriter.message(5, initializer);
  }
  for (const input of graph.inputs) graphWriter.message(11, input);
  for (const output of graph.outputs) graphWriter.message(12, output);
  return new ProtoWriter()
    .varint(1, ONNX_IR_VERSION)
    .string(2, producer)
    .message(7, graphWriter)
    .message(8, new ProtoWriter().string(1, "").varint(2, STUB_ONNX_OPSET))
    .finish();
}

export interface StubGraphNames {
  /** The model's producer_name. */
  readonly producer: string;
  /** The graph name; the role is appended ("<prefix>-encoder"). */
  readonly graphPrefix: string;
}

const STUB_NAMES: StubGraphNames = {
  producer: "semantscript-stub",
  graphPrefix: "stub",
};

/** input_ids, attention_mask int64 [BATCH,SEQUENCE] -> sentence_embedding float32 [BATCH,1]. */
export function stubEncoderOnnx(
  names: StubGraphNames = STUB_NAMES,
): Uint8Array {
  const sequence = ["BATCH", "SEQUENCE"] as const;
  return model(names.producer, {
    name: `${names.graphPrefix}-encoder`,
    nodes: [
      {
        inputs: ["input_ids"],
        outputs: ["ids_float"],
        opType: "Cast",
        intAttributes: [["to", FLOAT]],
      },
      {
        inputs: ["attention_mask"],
        outputs: ["mask_float"],
        opType: "Cast",
        intAttributes: [["to", FLOAT]],
      },
      {
        inputs: ["ids_float", "mask_float"],
        outputs: ["masked"],
        opType: "Mul",
      },
      {
        inputs: ["masked", "axes"],
        outputs: ["sentence_embedding"],
        opType: "ReduceSum",
        intAttributes: [["keepdims", 1]],
      },
    ],
    initializers: [int64Tensor("axes", [1], [1])],
    inputs: [
      valueInfo("input_ids", INT64, sequence),
      valueInfo("attention_mask", INT64, sequence),
    ],
    outputs: [valueInfo("sentence_embedding", FLOAT, ["BATCH", 1])],
  });
}

/** sentence_embedding float32 [BATCH,1] -> function_embedding float32 [BATCH,1]. */
export function stubAdapterOnnx(
  names: StubGraphNames = STUB_NAMES,
): Uint8Array {
  return model(names.producer, {
    name: `${names.graphPrefix}-adapter`,
    nodes: [
      {
        inputs: ["sentence_embedding"],
        outputs: ["function_embedding"],
        opType: "Identity",
      },
    ],
    initializers: [],
    inputs: [valueInfo("sentence_embedding", FLOAT, ["BATCH", 1])],
    outputs: [valueInfo("function_embedding", FLOAT, ["BATCH", 1])],
  });
}

/** function_embedding float32 [BATCH,1] -> logits float32 [BATCH,K] through one Gemm. */
export function stubHeadOnnx(
  logitCount: number,
  names: StubGraphNames = STUB_NAMES,
  weights: readonly number[] = new Array<number>(logitCount).fill(0),
  bias: readonly number[] = new Array<number>(logitCount).fill(0),
): Uint8Array {
  if (
    !Number.isSafeInteger(logitCount) ||
    logitCount < 1 ||
    weights.length !== logitCount ||
    bias.length !== logitCount
  ) {
    throw new RangeError("a stub head needs one weight and bias per logit");
  }
  return model(names.producer, {
    name: `${names.graphPrefix}-head`,
    nodes: [
      {
        inputs: ["function_embedding", "weights", "bias"],
        outputs: ["logits"],
        opType: "Gemm",
      },
    ],
    initializers: [
      floatTensor("weights", [1, logitCount], weights),
      floatTensor("bias", [logitCount], bias),
    ],
    inputs: [valueInfo("function_embedding", FLOAT, ["BATCH", 1])],
    outputs: [valueInfo("logits", FLOAT, ["BATCH", logitCount])],
  });
}

/** A whitespace WordLevel tokenizer: every canonical input tokenizes, mostly to [UNK]. */
export function stubTokenizerJson(): Uint8Array {
  const tokenizer = {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [],
    normalizer: null,
    pre_tokenizer: { type: "Whitespace" },
    post_processor: null,
    decoder: null,
    model: {
      type: "WordLevel",
      vocab: {
        "[UNK]": 0,
        semantscript: 1,
        input: 2,
        true: 3,
        false: 4,
        string: 5,
        number: 6,
        object: 7,
      },
      unk_token: "[UNK]",
    },
  };
  return utf8.encode(`${JSON.stringify(tokenizer, null, 2)}\n`);
}
