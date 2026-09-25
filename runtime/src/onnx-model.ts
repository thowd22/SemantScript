const textDecoder = new TextDecoder("utf-8", { fatal: true });
const DEFAULT_DOMAIN = "";
const STANDARD_DOMAIN_ALIAS = "ai.onnx";
const MAXIMUM_PROTO_DEPTH = 64;

export interface OnnxContainerInspection {
  readonly defaultOpset: number;
  readonly operatorDomains: ReadonlySet<string>;
}

/**
 * Performs the load-time container checks that ONNX Runtime metadata does not
 * expose: declared opset imports, custom operator domains, and external tensor
 * data. Tensor names, dtypes, and shapes are checked against live ORT session
 * metadata in the worker before it reports ready.
 */
export function inspectOnnxContainer(
  bytes: Uint8Array,
): OnnxContainerInspection {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
    throw new TypeError("ONNX model bytes must be a non-empty Uint8Array");
  }

  const reader = new ProtoReader(bytes);
  const imports = new Map<string, number>();
  const operatorDomains = new Set<string>();
  let sawGraph = false;

  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 7 && wire === 2) {
      inspectGraph(reader.message(), operatorDomains, 0);
      sawGraph = true;
    } else if (field === 8 && wire === 2) {
      const entry = inspectOpsetImport(reader.message());
      if (imports.has(entry.domain)) {
        throw new TypeError(
          `ONNX model repeats opset domain ${JSON.stringify(entry.domain)}`,
        );
      }
      imports.set(entry.domain, entry.version);
    } else if (field === 20 && wire === 2) {
      throw new TypeError(
        "ONNX training graphs are not allowed in runtime artifacts",
      );
    } else if (field === 25 && wire === 2) {
      throw new TypeError(
        "ONNX local function definitions are not allowed in runtime artifacts",
      );
    } else {
      reader.skip(wire);
    }
  }

  if (!sawGraph) {
    throw new TypeError("ONNX model does not contain a graph");
  }
  const defaultOpset =
    imports.get(DEFAULT_DOMAIN) ?? imports.get(STANDARD_DOMAIN_ALIAS);
  if (defaultOpset === undefined) {
    throw new TypeError(
      "ONNX model does not import the standard operator domain",
    );
  }
  for (const domain of imports.keys()) {
    if (domain !== DEFAULT_DOMAIN && domain !== STANDARD_DOMAIN_ALIAS) {
      throw new TypeError(
        `ONNX model imports unsupported custom domain ${JSON.stringify(domain)}`,
      );
    }
  }
  for (const domain of operatorDomains) {
    if (domain !== DEFAULT_DOMAIN && domain !== STANDARD_DOMAIN_ALIAS) {
      throw new TypeError(
        `ONNX model uses unsupported custom domain ${JSON.stringify(domain)}`,
      );
    }
  }

  return Object.freeze({ defaultOpset, operatorDomains });
}

function inspectOpsetImport(reader: ProtoReader): {
  readonly domain: string;
  readonly version: number;
} {
  let domain = DEFAULT_DOMAIN;
  let version: number | undefined;
  while (!reader.done) {
    const tag = reader.tag();
    if (tag.field === 1 && tag.wire === 2) {
      domain = reader.string();
    } else if (tag.field === 2 && tag.wire === 0) {
      version = reader.safeUnsignedInteger("ONNX opset version");
    } else {
      reader.skip(tag.wire);
    }
  }
  if (version === undefined || version < 1) {
    throw new TypeError("ONNX opset import requires a positive version");
  }
  return { domain, version };
}

function inspectGraph(
  reader: ProtoReader,
  domains: Set<string>,
  depth: number,
): void {
  assertDepth(depth);
  while (!reader.done) {
    const tag = reader.tag();
    if (tag.field === 1 && tag.wire === 2) {
      inspectNode(reader.message(), domains, depth + 1);
    } else if (tag.field === 5 && tag.wire === 2) {
      inspectTensor(reader.message());
    } else if (tag.field === 15 && tag.wire === 2) {
      inspectSparseTensor(reader.message());
    } else {
      reader.skip(tag.wire);
    }
  }
}

function inspectNode(
  reader: ProtoReader,
  domains: Set<string>,
  depth: number,
): void {
  assertDepth(depth);
  let domain = DEFAULT_DOMAIN;
  while (!reader.done) {
    const tag = reader.tag();
    if (tag.field === 5 && tag.wire === 2) {
      inspectAttribute(reader.message(), domains, depth + 1);
    } else if (tag.field === 7 && tag.wire === 2) {
      domain = reader.string();
    } else {
      reader.skip(tag.wire);
    }
  }
  domains.add(domain);
}

function inspectAttribute(
  reader: ProtoReader,
  domains: Set<string>,
  depth: number,
): void {
  assertDepth(depth);
  while (!reader.done) {
    const tag = reader.tag();
    if ((tag.field === 5 || tag.field === 10) && tag.wire === 2) {
      inspectTensor(reader.message());
    } else if ((tag.field === 6 || tag.field === 11) && tag.wire === 2) {
      inspectGraph(reader.message(), domains, depth + 1);
    } else if ((tag.field === 22 || tag.field === 23) && tag.wire === 2) {
      inspectSparseTensor(reader.message());
    } else {
      reader.skip(tag.wire);
    }
  }
}

function inspectSparseTensor(reader: ProtoReader): void {
  while (!reader.done) {
    const tag = reader.tag();
    if ((tag.field === 1 || tag.field === 2) && tag.wire === 2) {
      inspectTensor(reader.message());
    } else {
      reader.skip(tag.wire);
    }
  }
}

function inspectTensor(reader: ProtoReader): void {
  while (!reader.done) {
    const tag = reader.tag();
    if (tag.field === 13) {
      throw new TypeError("ONNX external tensor data is not supported");
    }
    if (tag.field === 14 && tag.wire === 0) {
      const location = reader.safeUnsignedInteger("ONNX tensor data location");
      if (location !== 0) {
        throw new TypeError("ONNX external tensor data is not supported");
      }
    } else {
      reader.skip(tag.wire);
    }
  }
}

function assertDepth(depth: number): void {
  if (depth > MAXIMUM_PROTO_DEPTH) {
    throw new TypeError("ONNX graph nesting exceeds the runtime limit");
  }
}

class ProtoReader {
  readonly #bytes: Uint8Array;
  readonly #end: number;
  #offset = 0;

  constructor(bytes: Uint8Array) {
    this.#bytes = bytes;
    this.#end = bytes.byteLength;
  }

  get done(): boolean {
    return this.#offset === this.#end;
  }

  tag(): { readonly field: number; readonly wire: number } {
    const tag = this.#varint();
    const field = Number(tag >> 3n);
    const wire = Number(tag & 7n);
    if (
      !Number.isSafeInteger(field) ||
      field < 1 ||
      wire > 5 ||
      wire === 3 ||
      wire === 4
    ) {
      throw new TypeError("ONNX protobuf contains an invalid field tag");
    }
    return { field, wire };
  }

  safeUnsignedInteger(label: string): number {
    const value = this.#varint();
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new TypeError(`${label} exceeds the JavaScript safe-integer range`);
    }
    return Number(value);
  }

  string(): string {
    try {
      return textDecoder.decode(this.bytes());
    } catch (error) {
      throw new TypeError("ONNX protobuf contains invalid UTF-8", {
        cause: error,
      });
    }
  }

  bytes(): Uint8Array {
    const length = this.safeUnsignedInteger("ONNX length-delimited field");
    const end = this.#offset + length;
    if (!Number.isSafeInteger(end) || end > this.#end) {
      throw new TypeError(
        "ONNX protobuf length exceeds the containing message",
      );
    }
    const value = this.#bytes.subarray(this.#offset, end);
    this.#offset = end;
    return value;
  }

  message(): ProtoReader {
    return new ProtoReader(this.bytes());
  }

  skip(wire: number): void {
    switch (wire) {
      case 0:
        this.#varint();
        return;
      case 1:
        this.#advance(8);
        return;
      case 2:
        this.bytes();
        return;
      case 5:
        this.#advance(4);
        return;
      default:
        throw new TypeError(
          `ONNX protobuf uses unsupported wire type ${String(wire)}`,
        );
    }
  }

  #advance(length: number): void {
    const end = this.#offset + length;
    if (end > this.#end) {
      throw new TypeError("ONNX protobuf field exceeds the containing message");
    }
    this.#offset = end;
  }

  #varint(): bigint {
    let result = 0n;
    for (let index = 0; index < 10; index += 1) {
      const byte = this.#bytes[this.#offset];
      if (byte === undefined) {
        throw new TypeError("ONNX protobuf ends inside a varint");
      }
      this.#offset += 1;
      result |= BigInt(byte & 0x7f) << BigInt(index * 7);
      if ((byte & 0x80) === 0) {
        if (index === 9 && byte > 1) {
          throw new TypeError("ONNX protobuf varint exceeds 64 bits");
        }
        return result;
      }
    }
    throw new TypeError("ONNX protobuf varint exceeds 64 bits");
  }
}
