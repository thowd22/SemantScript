const encoder = new TextEncoder();

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | readonly JsonValue[]
  | { readonly [name: string]: JsonValue };

type TypedJsonNode =
  | readonly ["null"]
  | readonly ["boolean", boolean]
  | readonly ["number", string]
  | readonly ["string", string]
  | readonly ["array", readonly TypedJsonNode[]]
  | readonly ["object", readonly (readonly [string, TypedJsonNode])[]];

export function semanticJsonBytes(value: unknown): Uint8Array {
  return encoder.encode(semanticJsonString(value));
}

export function semanticJsonString(value: unknown): string {
  return JSON.stringify(["semantscript-semantic-json", 1, toTypedNode(value)]);
}

export function compareBytes(left: Uint8Array, right: Uint8Array): number {
  const length = Math.min(left.length, right.length);

  for (let index = 0; index < length; index += 1) {
    const difference = (left[index] ?? 0) - (right[index] ?? 0);

    if (difference !== 0) {
      return difference;
    }
  }

  return left.length - right.length;
}

export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function toTypedNode(value: unknown): TypedJsonNode {
  if (value === null) {
    return ["null"];
  }

  if (typeof value === "boolean") {
    return ["boolean", value];
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError("semantic JSON numbers must be finite");
    }

    return ["number", numberToBinary64Hex(value)];
  }

  if (typeof value === "string") {
    assertUnicodeScalarString(value);
    return ["string", value];
  }

  if (Array.isArray(value)) {
    return ["array", value.map(toTypedNode)];
  }

  if (typeof value !== "object") {
    throw new TypeError(
      "semantic JSON values must contain only JSON-compatible data",
    );
  }

  const entries = Object.entries(value)
    .map(([name, entryValue]) => {
      assertUnicodeScalarString(name);
      return [name, toTypedNode(entryValue)] as const;
    })
    .sort(([left], [right]) =>
      compareBytes(encoder.encode(left), encoder.encode(right)),
    );
  return ["object", entries];
}

function numberToBinary64Hex(value: number): string {
  const buffer = new ArrayBuffer(8);
  new DataView(buffer).setFloat64(0, value, false);
  return bytesToHex(new Uint8Array(buffer));
}

function assertUnicodeScalarString(value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);

    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);

      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        throw new TypeError(
          "semantic JSON strings cannot contain unpaired UTF-16 surrogates",
        );
      }

      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new TypeError(
        "semantic JSON strings cannot contain unpaired UTF-16 surrogates",
      );
    }
  }
}
