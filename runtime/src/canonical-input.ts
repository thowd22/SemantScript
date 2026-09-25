import { types as nodeTypes } from "node:util";

const textEncoder = new TextEncoder();
const MAX_INPUT_DEPTH = 100;
const MAX_INPUT_NODES = 100_000;

export interface PrimitiveInputType {
  readonly kind: "string" | "boolean" | "number" | "null";
}

export interface LiteralInputType {
  readonly kind: "literal";
  readonly value: string | number | boolean | null;
}

export type EnumInputType =
  | {
      readonly kind: "enum";
      readonly name: string;
      readonly base: "string";
      readonly values: readonly string[];
    }
  | {
      readonly kind: "enum";
      readonly name: string;
      readonly base: "number";
      readonly values: readonly number[];
    };

export interface ArrayInputType {
  readonly kind: "array";
  readonly items: InputType;
}

export interface TupleInputType {
  readonly kind: "tuple";
  readonly items: readonly InputType[];
}

export interface ObjectInputField {
  readonly name: string;
  readonly optional: boolean;
  readonly type: InputType;
}

export interface ObjectInputType {
  readonly kind: "object";
  readonly name: string;
  readonly fields: readonly ObjectInputField[];
}

export interface UnionInputType {
  readonly kind: "union";
  /** Variants are in the compiler-defined semantic-JSON byte order. */
  readonly variants: readonly InputType[];
}

export type InputType =
  | PrimitiveInputType
  | LiteralInputType
  | EnumInputType
  | ArrayInputType
  | TupleInputType
  | ObjectInputType
  | UnionInputType;

export interface CanonicalInputEntry {
  readonly name: string;
  readonly index: number;
  readonly tsType?: string;
  readonly type: InputType;
}

export type SemaInputErrorReason =
  | "accessor"
  | "array-property"
  | "class-instance"
  | "cycle"
  | "extra"
  | "invalid-unicode"
  | "limit"
  | "missing"
  | "no-union-variant"
  | "non-finite"
  | "promise"
  | "proxy"
  | "schema"
  | "sparse-array"
  | "symbol-key"
  | "tuple-length"
  | "type"
  | "value";

export type SemaInputPath = readonly (string | number)[];

/** A call's inputs do not match the function's declared input types or exceed the byte limit (`reason` says how). */
export class SemaInputError extends TypeError {
  public readonly code = "SEMA_INPUT_INVALID";
  public readonly path: SemaInputPath;
  public readonly reason: SemaInputErrorReason;

  public constructor(
    reason: SemaInputErrorReason,
    path: SemaInputPath,
    message: string,
  ) {
    super(`${formatPath(path)}: ${message}`);
    this.name = "SemaInputError";
    this.path = Object.freeze([...path]);
    this.reason = reason;
  }
}

type PrimitiveTypedValue =
  | readonly ["null"]
  | readonly ["boolean", boolean]
  | readonly ["string", string]
  | readonly ["number", string];

type TypedInputValue =
  | PrimitiveTypedValue
  | readonly ["literal", PrimitiveTypedValue]
  | readonly ["enum", string, number, PrimitiveTypedValue]
  | readonly ["union", number, TypedInputValue]
  | readonly ["array", readonly TypedInputValue[]]
  | readonly ["tuple", readonly TypedInputValue[]]
  | readonly ["object", readonly (readonly [string, TypedInputValue])[]];

type ExactJson = string | number | boolean | readonly ExactJson[];

interface WorkBudget {
  remaining: number;
}

interface EncodeContext {
  readonly active: Set<object>;
  readonly budget: WorkBudget;
}

interface InspectedRecord {
  readonly descriptors: Readonly<Record<string, PropertyDescriptor>>;
  readonly enumerableKeys: readonly string[];
}

export const CANONICAL_INPUT_V1 = "semantscript.canonical-input/v1";
export const CANONICAL_INPUT_V2 = "semantscript.canonical-input/v2";
export type CanonicalInputVersion = 1 | 2;
export type CanonicalInputEncoding =
  typeof CANONICAL_INPUT_V1 | typeof CANONICAL_INPUT_V2;

const CANONICAL_INPUT_ENCODINGS: Readonly<
  Record<CanonicalInputVersion, CanonicalInputEncoding>
> = {
  1: CANONICAL_INPUT_V1,
  2: CANONICAL_INPUT_V2,
};

/** Maps a manifest encoding identifier to its serializer version, or undefined when unimplemented. */
export function canonicalInputVersion(
  encoding: string,
): CanonicalInputVersion | undefined {
  if (encoding === CANONICAL_INPUT_V1) return 1;
  if (encoding === CANONICAL_INPUT_V2) return 2;
  return undefined;
}

export interface CanonicalInputSerializationOptions {
  /** Maximum UTF-8 byte length of the complete canonical input. */
  readonly maximumBytes?: number;
  /**
   * Encoding version: 1 is the exact-JSON envelope with hexadecimal binary64
   * numbers, 2 the compact text (name=value pairs, {k=v} objects, [v] arrays,
   * bare identifiers and JavaScript number spelling). Both are canonical and
   * injective over validated inputs; an artifact declares the one it was
   * trained with.
   */
  readonly version?: CanonicalInputVersion;
}

/** Validates and serializes runtime inputs as canonical-input bytes of the requested version. */
export function serializeCanonicalInputs(
  schema: readonly CanonicalInputEntry[],
  inputs: unknown,
  options: CanonicalInputSerializationOptions = {},
): Uint8Array {
  const version = resolveVersion(options.version);
  const envelope = encodeCanonicalInputEnvelope(schema, inputs);

  if (options.maximumBytes !== undefined) {
    if (
      !Number.isSafeInteger(options.maximumBytes) ||
      options.maximumBytes < 1
    ) {
      throw new RangeError("maximumBytes must be a positive safe integer");
    }
    if (version === 1)
      return writeBoundedExactJson(envelope, options.maximumBytes);
    const bytes = textEncoder.encode(writeCompact(envelope));
    if (bytes.length > options.maximumBytes) {
      inputFailure(
        "limit",
        [],
        `canonical input exceeds the ${String(options.maximumBytes)} byte limit`,
      );
    }
    return bytes;
  }

  return textEncoder.encode(
    version === 1 ? writeExactJson(envelope) : writeCompact(envelope),
  );
}

/** String form intended for golden-vector tests and diagnostics. */
export function serializeCanonicalInputsString(
  schema: readonly CanonicalInputEntry[],
  inputs: unknown,
  options: Pick<CanonicalInputSerializationOptions, "version"> = {},
): string {
  const version = resolveVersion(options.version);
  const envelope = encodeCanonicalInputEnvelope(schema, inputs);
  return version === 1 ? writeExactJson(envelope) : writeCompact(envelope);
}

function resolveVersion(
  version: CanonicalInputVersion | undefined,
): CanonicalInputVersion {
  if (version === undefined) return 1;
  if (!(version in CANONICAL_INPUT_ENCODINGS)) {
    throw new RangeError("canonical input version must be 1 or 2");
  }
  return version;
}

function encodeCanonicalInputEnvelope(
  schema: readonly CanonicalInputEntry[],
  inputs: unknown,
): ExactJson {
  const orderedSchema = validateAndOrderSchema(schema);
  const inspectedInputs = inspectPlainRecord(inputs, [], "input record");
  const expectedNames = new Set(orderedSchema.map(({ name }) => name));

  for (const key of inspectedInputs.enumerableKeys) {
    assertUnicodeScalarString(key, [key]);
    const descriptor = inspectedInputs.descriptors[key];
    assertDataDescriptor(descriptor, [key]);

    if (!expectedNames.has(key)) {
      inputFailure("extra", [key], `unexpected input ${quoteForMessage(key)}`);
    }
  }

  const context: EncodeContext = {
    active: new Set(),
    budget: { remaining: MAX_INPUT_NODES },
  };
  const pairs: Array<readonly [string, TypedInputValue]> = [];

  for (const entry of orderedSchema) {
    const path = [entry.name] as const;
    const descriptor = inspectedInputs.descriptors[entry.name];

    if (!descriptor?.enumerable || !("value" in descriptor)) {
      inputFailure(
        "missing",
        path,
        `missing required input ${quoteForMessage(entry.name)}`,
      );
    }

    pairs.push([
      entry.name,
      encodeInputValue(entry.type, descriptor.value, path, context, 0),
    ]);
  }

  return ["semantscript-input", 1, pairs];
}

function encodeInputValue(
  type: InputType,
  value: unknown,
  path: SemaInputPath,
  context: EncodeContext,
  depth: number,
): TypedInputValue {
  consumeWork(context.budget, path, depth);
  rejectHostObject(value, path);

  switch (type.kind) {
    case "null":
      if (value !== null) {
        expectedType(path, "null");
      }
      return ["null"];
    case "boolean":
      if (typeof value !== "boolean") {
        expectedType(path, "boolean");
      }
      return ["boolean", value];
    case "string":
      if (typeof value !== "string") {
        expectedType(path, "string");
      }
      assertUnicodeScalarString(value, path);
      return ["string", value];
    case "number":
      return encodeNumber(value, path);
    case "literal":
      return ["literal", encodeLiteral(type.value, value, path)];
    case "enum":
      return encodeEnum(type, value, path);
    case "array":
      return encodeArray(type, value, path, context, depth);
    case "tuple":
      return encodeTuple(type, value, path, context, depth);
    case "object":
      return encodeObject(type, value, path, context, depth);
    case "union":
      return encodeUnion(type, value, path, context, depth);
  }
}

function encodeLiteral(
  expected: string | number | boolean | null,
  value: unknown,
  path: SemaInputPath,
): PrimitiveTypedValue {
  const encoded = encodePrimitive(value, path);

  if (!primitiveSameValue(value, expected)) {
    inputFailure("value", path, "value does not match the required literal");
  }

  return encoded;
}

function encodeEnum(
  type: EnumInputType,
  value: unknown,
  path: SemaInputPath,
): TypedInputValue {
  let declarationIndex: number;
  let encoded: PrimitiveTypedValue;

  if (type.base === "string") {
    if (typeof value !== "string") {
      expectedType(path, `member of enum ${type.name}`);
    }
    assertUnicodeScalarString(value, path);
    declarationIndex = type.values.findIndex(
      (candidate) => candidate === value,
    );
    encoded = ["string", value];
  } else {
    encoded = encodeNumber(value, path);
    declarationIndex = type.values.findIndex((candidate) =>
      Object.is(candidate, value),
    );
  }

  if (declarationIndex < 0) {
    inputFailure("value", path, `value is not a member of enum ${type.name}`);
  }

  return ["enum", type.name, declarationIndex, encoded];
}

function encodeArray(
  type: ArrayInputType,
  value: unknown,
  path: SemaInputPath,
  context: EncodeContext,
  depth: number,
): TypedInputValue {
  if (!Array.isArray(value)) {
    expectedType(path, "array");
  }

  if (value.length > context.budget.remaining) {
    inputFailure(
      "limit",
      path,
      `input work exceeds the limit of ${String(MAX_INPUT_NODES)} nodes`,
    );
  }

  const descriptors = inspectArray(value, path);
  return withActiveValue(value, path, context, () => [
    "array",
    Array.from({ length: value.length }, (_, index) => {
      const descriptor = descriptors[String(index)];
      assertDataDescriptor(descriptor, [...path, index]);
      return encodeInputValue(
        type.items,
        descriptor.value,
        [...path, index],
        context,
        depth + 1,
      );
    }),
  ]);
}

function encodeTuple(
  type: TupleInputType,
  value: unknown,
  path: SemaInputPath,
  context: EncodeContext,
  depth: number,
): TypedInputValue {
  if (!Array.isArray(value)) {
    expectedType(path, "tuple");
  }

  if (value.length !== type.items.length) {
    inputFailure(
      "tuple-length",
      path,
      `expected tuple length ${String(type.items.length)}, received ${String(value.length)}`,
    );
  }

  const descriptors = inspectArray(value, path);
  return withActiveValue(value, path, context, () => [
    "tuple",
    type.items.map((itemType, index) => {
      const descriptor = descriptors[String(index)];
      assertDataDescriptor(descriptor, [...path, index]);
      return encodeInputValue(
        itemType,
        descriptor.value,
        [...path, index],
        context,
        depth + 1,
      );
    }),
  ]);
}

function encodeObject(
  type: ObjectInputType,
  value: unknown,
  path: SemaInputPath,
  context: EncodeContext,
  depth: number,
): TypedInputValue {
  const inspected = inspectPlainRecord(value, path, `object ${type.name}`);
  const fields = new Map(type.fields.map((field) => [field.name, field]));

  for (const key of inspected.enumerableKeys) {
    assertUnicodeScalarString(key, [...path, key]);
    const descriptor = inspected.descriptors[key];
    assertDataDescriptor(descriptor, [...path, key]);

    if (!fields.has(key)) {
      inputFailure(
        "extra",
        [...path, key],
        `unexpected property ${quoteForMessage(key)}`,
      );
    }
  }

  return withActiveValue(value as object, path, context, () => {
    const pairs: Array<readonly [string, TypedInputValue]> = [];

    for (const field of type.fields) {
      const fieldPath = [...path, field.name];
      const descriptor = inspected.descriptors[field.name];

      if (!descriptor?.enumerable || !("value" in descriptor)) {
        if (!field.optional) {
          inputFailure(
            "missing",
            fieldPath,
            `missing required property ${quoteForMessage(field.name)}`,
          );
        }
        continue;
      }

      pairs.push([
        field.name,
        encodeInputValue(
          field.type,
          descriptor.value,
          fieldPath,
          context,
          depth + 1,
        ),
      ]);
    }

    pairs.sort(([left], [right]) => compareUtf8(left, right));
    return ["object", pairs];
  });
}

function encodeUnion(
  type: UnionInputType,
  value: unknown,
  path: SemaInputPath,
  context: EncodeContext,
  depth: number,
): TypedInputValue {
  for (const [index, variant] of type.variants.entries()) {
    try {
      return [
        "union",
        index,
        encodeInputValue(variant, value, path, context, depth + 1),
      ];
    } catch (error) {
      if (
        !(error instanceof SemaInputError) ||
        isFatalInputFailure(error.reason)
      ) {
        throw error;
      }
    }
  }

  inputFailure(
    "no-union-variant",
    path,
    "value does not match any union variant",
  );
}

function encodePrimitive(
  value: unknown,
  path: SemaInputPath,
): PrimitiveTypedValue {
  if (value === null) {
    return ["null"];
  }

  if (typeof value === "boolean") {
    return ["boolean", value];
  }

  if (typeof value === "string") {
    assertUnicodeScalarString(value, path);
    return ["string", value];
  }

  if (typeof value === "number") {
    return encodeNumber(value, path);
  }

  expectedType(path, "primitive literal");
}

function encodeNumber(
  value: unknown,
  path: SemaInputPath,
): readonly ["number", string] {
  if (typeof value !== "number") {
    expectedType(path, "number");
  }

  if (!Number.isFinite(value)) {
    inputFailure("non-finite", path, "numbers must be finite");
  }

  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return ["number", bytesToHex(bytes)];
}

function inspectPlainRecord(
  value: unknown,
  path: SemaInputPath,
  description: string,
): InspectedRecord {
  rejectHostObject(value, path);

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    expectedType(path, description);
  }

  const prototype = Object.getPrototypeOf(value) as unknown;

  if (prototype !== Object.prototype && prototype !== null) {
    inputFailure(
      "class-instance",
      path,
      `${description} must be a plain object`,
    );
  }

  if (Object.getOwnPropertySymbols(value).length > 0) {
    inputFailure(
      "symbol-key",
      path,
      `${description} cannot contain symbol-keyed properties`,
    );
  }

  const descriptors = Object.getOwnPropertyDescriptors(value);
  const enumerableKeys = Object.keys(descriptors).filter(
    (key) => descriptors[key]?.enumerable,
  );
  return { descriptors, enumerableKeys };
}

function inspectArray(
  value: readonly unknown[],
  path: SemaInputPath,
): Readonly<Record<string, PropertyDescriptor>> {
  if (Object.getOwnPropertySymbols(value).length > 0) {
    inputFailure(
      "symbol-key",
      path,
      "arrays cannot contain symbol-keyed properties",
    );
  }

  const descriptors = Object.getOwnPropertyDescriptors(value);
  const enumerableKeys = Object.keys(descriptors).filter(
    (key) => descriptors[key]?.enumerable,
  );

  for (const key of enumerableKeys) {
    const descriptor = descriptors[key];
    assertDataDescriptor(descriptor, [...path, key]);

    if (!isArrayIndexForLength(key, value.length)) {
      inputFailure(
        "array-property",
        [...path, key],
        "arrays cannot have enumerable non-index properties",
      );
    }
  }

  if (enumerableKeys.length !== value.length) {
    inputFailure("sparse-array", path, "arrays must be dense");
  }

  for (let index = 0; index < value.length; index += 1) {
    if (!descriptors[String(index)]?.enumerable) {
      inputFailure("sparse-array", [...path, index], "arrays must be dense");
    }
  }

  return descriptors;
}

function rejectHostObject(value: unknown, path: SemaInputPath): void {
  if (
    value === null ||
    (typeof value !== "object" && typeof value !== "function")
  ) {
    return;
  }

  if (nodeTypes.isProxy(value)) {
    inputFailure("proxy", path, "proxies are not supported as inputs");
  }

  if (typeof value === "function") {
    return;
  }

  if (nodeTypes.isPromise(value)) {
    inputFailure("promise", path, "promises are not supported as inputs");
  }

  const prototype = Object.getPrototypeOf(value) as unknown;

  if (
    (Array.isArray(value) && prototype !== Array.prototype) ||
    (!Array.isArray(value) &&
      prototype !== Object.prototype &&
      prototype !== null)
  ) {
    inputFailure(
      "class-instance",
      path,
      "class instances and host objects are not supported",
    );
  }
}

function withActiveValue<T>(
  value: object,
  path: SemaInputPath,
  context: EncodeContext,
  encode: () => T,
): T {
  if (context.active.has(value)) {
    inputFailure("cycle", path, "cyclic input values are not supported");
  }

  context.active.add(value);

  try {
    return encode();
  } finally {
    context.active.delete(value);
  }
}

function validateAndOrderSchema(
  schema: readonly CanonicalInputEntry[],
): readonly CanonicalInputEntry[] {
  const ordered = [...schema].sort((left, right) => left.index - right.index);
  const names = new Set<string>();
  const schemaBudget: WorkBudget = { remaining: MAX_INPUT_NODES };

  for (const [expectedIndex, entry] of ordered.entries()) {
    if (entry.index !== expectedIndex) {
      schemaFailure(
        `input indices must be unique and dense from zero; missing index ${String(expectedIndex)}`,
      );
    }

    assertUnicodeScalarString(entry.name, ["$schema", expectedIndex, "name"]);

    if (names.has(entry.name)) {
      schemaFailure(`duplicate input name ${quoteForMessage(entry.name)}`);
    }

    names.add(entry.name);
    validateInputType(entry.type, new Set(), schemaBudget, 0);
  }

  return ordered;
}

function validateInputType(
  type: InputType,
  active: Set<object>,
  budget: WorkBudget,
  depth: number,
): string {
  consumeWork(budget, ["$schema"], depth);

  if (active.has(type)) {
    schemaFailure("input schemas cannot be recursive at runtime");
  }

  active.add(type);

  try {
    switch (type.kind) {
      case "null":
      case "boolean":
      case "string":
      case "number":
        return type.kind;
      case "literal":
        assertSchemaPrimitive(type.value);
        return JSON.stringify(["literal", primitiveSchemaKey(type.value)]);
      case "enum": {
        assertUnicodeScalarString(type.name, ["$schema", "enum", "name"]);

        if (type.name.length === 0 || type.values.length === 0) {
          schemaFailure(
            "enum input schemas require a name and at least one value",
          );
        }

        const values = type.values.map((value) => {
          if (type.base === "string") {
            if (typeof value !== "string") {
              schemaFailure("string enums can contain only strings");
            }
            assertUnicodeScalarString(value, ["$schema", type.name]);
          } else if (typeof value !== "number" || !Number.isFinite(value)) {
            schemaFailure("number enums can contain only finite numbers");
          }
          return primitiveSchemaKey(value);
        });

        if (new Set(values).size !== values.length) {
          schemaFailure(`enum ${type.name} contains duplicate values`);
        }

        return JSON.stringify(["enum", type.name, type.base, values]);
      }
      case "array":
        return JSON.stringify([
          "array",
          validateInputType(type.items, active, budget, depth + 1),
        ]);
      case "tuple":
        return JSON.stringify([
          "tuple",
          type.items.map((item) =>
            validateInputType(item, active, budget, depth + 1),
          ),
        ]);
      case "object": {
        assertUnicodeScalarString(type.name, ["$schema", "object", "name"]);
        const names = new Set<string>();
        const fields = type.fields.map((field) => {
          assertUnicodeScalarString(field.name, [
            "$schema",
            type.name,
            "field",
          ]);

          if (names.has(field.name)) {
            schemaFailure(
              `object ${type.name} contains duplicate field ${quoteForMessage(field.name)}`,
            );
          }

          names.add(field.name);
          return [
            field.name,
            field.optional,
            validateInputType(field.type, active, budget, depth + 1),
          ] as const;
        });
        return JSON.stringify(["object", type.name, fields]);
      }
      case "union": {
        if (type.variants.length < 2) {
          schemaFailure("union input schemas require at least two variants");
        }

        const variants = type.variants.map((variant) =>
          validateInputType(variant, active, budget, depth + 1),
        );

        if (new Set(variants).size !== variants.length) {
          schemaFailure("union input schema contains duplicate variants");
        }

        return JSON.stringify(["union", variants]);
      }
    }
  } finally {
    active.delete(type);
  }
}

function assertSchemaPrimitive(
  value: unknown,
): asserts value is string | number | boolean | null {
  if (value === null || typeof value === "boolean") {
    return;
  }

  if (typeof value === "string") {
    assertUnicodeScalarString(value, ["$schema", "literal"]);
    return;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return;
  }

  schemaFailure("literal input schemas require a finite primitive value");
}

function primitiveSchemaKey(value: string | number | boolean | null): string {
  if (typeof value === "number") {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value, false);
    return `number:${bytesToHex(bytes)}`;
  }

  return `${value === null ? "null" : typeof value}:${String(value)}`;
}

function primitiveSameValue(
  value: unknown,
  expected: string | number | boolean | null,
): boolean {
  return typeof expected === "number"
    ? Object.is(value, expected)
    : value === expected;
}

function assertDataDescriptor(
  descriptor: PropertyDescriptor | undefined,
  path: SemaInputPath,
): asserts descriptor is PropertyDescriptor & { readonly value: unknown } {
  if (!descriptor || !("value" in descriptor)) {
    inputFailure(
      "accessor",
      path,
      "input validation does not invoke getters or setters",
    );
  }
}

function assertUnicodeScalarString(value: string, path: SemaInputPath): void {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);

    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);

      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        inputFailure(
          "invalid-unicode",
          path,
          "strings cannot contain unpaired UTF-16 surrogates",
        );
      }
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      inputFailure(
        "invalid-unicode",
        path,
        "strings cannot contain unpaired UTF-16 surrogates",
      );
    }
  }
}

function consumeWork(
  budget: WorkBudget,
  path: SemaInputPath,
  depth: number,
): void {
  if (depth > MAX_INPUT_DEPTH) {
    inputFailure(
      "limit",
      path,
      `input nesting exceeds the limit of ${String(MAX_INPUT_DEPTH)}`,
    );
  }

  budget.remaining -= 1;

  if (budget.remaining < 0) {
    inputFailure(
      "limit",
      path,
      `input work exceeds the limit of ${String(MAX_INPUT_NODES)} nodes`,
    );
  }
}

function isFatalInputFailure(reason: SemaInputErrorReason): boolean {
  return [
    "accessor",
    "array-property",
    "class-instance",
    "cycle",
    "invalid-unicode",
    "limit",
    "non-finite",
    "promise",
    "proxy",
    "sparse-array",
    "symbol-key",
  ].includes(reason);
}

function isArrayIndexForLength(key: string, length: number): boolean {
  if (!/^(?:0|[1-9][0-9]*)$/u.test(key)) {
    return false;
  }

  const index = Number(key);
  return (
    Number.isSafeInteger(index) &&
    index >= 0 &&
    index < length &&
    String(index) === key
  );
}

function compareUtf8(left: string, right: string): number {
  const leftBytes = textEncoder.encode(left);
  const rightBytes = textEncoder.encode(right);
  const length = Math.min(leftBytes.length, rightBytes.length);

  for (let index = 0; index < length; index += 1) {
    const difference = (leftBytes[index] ?? 0) - (rightBytes[index] ?? 0);

    if (difference !== 0) {
      return difference;
    }
  }

  return leftBytes.length - rightBytes.length;
}

function writeExactJson(value: ExactJson): string {
  if (typeof value === "string") {
    return quoteExactJsonString(value);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new TypeError(
        "canonical input envelope integers must be unsigned safe integers",
      );
    }
    return String(value);
  }

  return `[${value.map(writeExactJson).join(",")}]`;
}

// canonical-input/v2: the same typed tree rendered as compact text. Tags that
// the value itself carries (literal, enum, union) are dropped; the schema fixes
// arrays versus tuples per path, so both use brackets. Strings and keys are bare
// only when they cannot be mistaken for another token, which keeps the
// encoding injective over validated inputs.
const BARE_TOKEN = /^[A-Za-z_][A-Za-z0-9_.-]*$/u;
const RESERVED_TOKENS = new Set(["true", "false", "null"]);

function expectList(value: ExactJson | undefined): readonly ExactJson[] {
  // ExactJson objects are only arrays, so the typeof check narrows without `any`.
  if (value === undefined || typeof value !== "object") {
    throw new TypeError("canonical input typed value is malformed");
  }
  return value;
}

function compactPair(pair: ExactJson): string {
  const entry = expectList(pair);
  const [key, value] = entry;
  if (entry.length !== 2 || typeof key !== "string" || value === undefined) {
    throw new TypeError("canonical input pair is malformed");
  }
  return `${compactString(key)}=${compactValue(value)}`;
}

function writeCompact(envelope: ExactJson): string {
  const parts = expectList(envelope);
  if (parts.length !== 3) {
    throw new TypeError("canonical input envelope is malformed");
  }
  return expectList(parts[2]).map(compactPair).join(" ");
}

function compactValue(value: ExactJson): string {
  const typed = expectList(value);
  const [tag] = typed;
  if (typeof tag !== "string") {
    throw new TypeError("canonical input typed value is malformed");
  }
  switch (tag) {
    case "null":
      return "null";
    case "boolean":
      return typed[1] === true ? "true" : "false";
    case "string":
      return compactString(expectString(typed[1]));
    case "number":
      return compactNumber(hexToDouble(expectString(typed[1])));
    case "literal":
      return compactValue(expectJson(typed[1]));
    case "enum":
      return compactValue(expectJson(typed[3]));
    case "union":
      return compactValue(expectJson(typed[2]));
    case "array":
    case "tuple":
      return `[${expectList(typed[1]).map(compactValue).join(",")}]`;
    case "object":
      return `{${expectList(typed[1]).map(compactPair).join(",")}}`;
    default:
      throw new TypeError(
        `canonical input typed value has unknown tag ${JSON.stringify(tag)}`,
      );
  }
}

function compactString(value: string): string {
  return BARE_TOKEN.test(value) && !RESERVED_TOKENS.has(value)
    ? value
    : quoteExactJsonString(value);
}

/** ECMAScript Number-to-string spelling, except that negative zero keeps its sign. */
function compactNumber(value: number): string {
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function hexToDouble(hex: string): number {
  if (hex.length !== 16)
    throw new TypeError("canonical input number is not 16 hex digits");
  const view = new DataView(new ArrayBuffer(8));
  for (let index = 0; index < 8; index += 1) {
    const byte = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
    if (!Number.isInteger(byte))
      throw new TypeError("canonical input number is not hexadecimal");
    view.setUint8(index, byte);
  }
  return view.getFloat64(0, false);
}

function expectString(value: ExactJson | undefined): string {
  if (typeof value !== "string")
    throw new TypeError("canonical input typed value is malformed");
  return value;
}

function expectJson(value: ExactJson | undefined): ExactJson {
  if (value === undefined)
    throw new TypeError("canonical input typed value is malformed");
  return value;
}

function writeBoundedExactJson(
  value: ExactJson,
  maximumBytes: number,
): Uint8Array {
  const writer = new BoundedUtf8Writer(maximumBytes);
  writeBoundedExactJsonValue(value, writer);
  return writer.finish();
}

function writeBoundedExactJsonValue(
  value: ExactJson,
  writer: BoundedUtf8Writer,
): void {
  if (typeof value === "string") {
    writer.writeJsonString(value);
    return;
  }

  if (typeof value === "boolean") {
    writer.writeAscii(value ? "true" : "false");
    return;
  }

  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new TypeError(
        "canonical input envelope integers must be unsigned safe integers",
      );
    }
    writer.writeAscii(String(value));
    return;
  }

  writer.writeAscii("[");
  for (let index = 0; index < value.length; index += 1) {
    if (index > 0) writer.writeAscii(",");
    const entry = value[index];
    if (entry === undefined) {
      throw new TypeError(
        "canonical input arrays cannot contain absent elements",
      );
    }
    writeBoundedExactJsonValue(entry, writer);
  }
  writer.writeAscii("]");
}

class BoundedUtf8Writer {
  readonly #maximumBytes: number;
  #bytes: Uint8Array;
  #length = 0;

  constructor(maximumBytes: number) {
    this.#maximumBytes = maximumBytes;
    this.#bytes = new Uint8Array(Math.min(maximumBytes, 4096));
  }

  writeAscii(value: string): void {
    for (let index = 0; index < value.length; index += 1) {
      this.#ensureWritable();
      this.#bytes[this.#length] = value.charCodeAt(index);
      this.#length += 1;
    }
  }

  writeJsonString(value: string): void {
    this.writeAscii('"');
    let unescapedStart = 0;

    for (let index = 0; index < value.length; index += 1) {
      const code = value.charCodeAt(index);
      let escape: string | undefined;

      switch (code) {
        case 0x08:
          escape = "\\b";
          break;
        case 0x09:
          escape = "\\t";
          break;
        case 0x0a:
          escape = "\\n";
          break;
        case 0x0c:
          escape = "\\f";
          break;
        case 0x0d:
          escape = "\\r";
          break;
        case 0x22:
          escape = '\\"';
          break;
        case 0x5c:
          escape = "\\\\";
          break;
        default:
          if (code <= 0x1f) {
            escape = `\\u00${code.toString(16).padStart(2, "0")}`;
          } else if (code >= 0xd800 && code <= 0xdbff) {
            const next = value.charCodeAt(index + 1);
            if (!(next >= 0xdc00 && next <= 0xdfff)) {
              throw new TypeError(
                "canonical input strings cannot contain unpaired UTF-16 surrogates",
              );
            }
            index += 1;
          } else if (code >= 0xdc00 && code <= 0xdfff) {
            throw new TypeError(
              "canonical input strings cannot contain unpaired UTF-16 surrogates",
            );
          }
          break;
      }

      if (escape !== undefined) {
        this.#writeUtf8Range(value, unescapedStart, index);
        this.writeAscii(escape);
        unescapedStart = index + 1;
      }
    }

    this.#writeUtf8Range(value, unescapedStart, value.length);
    this.writeAscii('"');
  }

  finish(): Uint8Array {
    return this.#bytes.slice(0, this.#length);
  }

  #writeUtf8Range(value: string, start: number, end: number): void {
    while (start < end) {
      this.#ensureWritable();
      let chunkEnd = Math.min(start + 4096, end);
      if (
        chunkEnd < end &&
        value.charCodeAt(chunkEnd - 1) >= 0xd800 &&
        value.charCodeAt(chunkEnd - 1) <= 0xdbff
      ) {
        chunkEnd -= 1;
      }
      const chunk = value.slice(start, chunkEnd);
      const { read, written } = textEncoder.encodeInto(
        chunk,
        this.#bytes.subarray(this.#length),
      );
      this.#length += written;
      start += read;
      if (read === 0 && start < end) this.#growOrFail();
    }
  }

  #ensureWritable(): void {
    if (this.#length < this.#bytes.length) return;
    this.#growOrFail();
  }

  #growOrFail(): void {
    if (this.#length >= this.#maximumBytes) {
      inputFailure(
        "limit",
        [],
        `canonical input exceeds the ${String(this.#maximumBytes)} byte limit`,
      );
    }

    if (this.#bytes.length >= this.#maximumBytes) {
      inputFailure(
        "limit",
        [],
        `canonical input exceeds the ${String(this.#maximumBytes)} byte limit`,
      );
    }

    const nextLength = Math.min(
      this.#maximumBytes,
      Math.max(this.#bytes.length * 2, this.#bytes.length + 4096),
    );
    const next = new Uint8Array(nextLength);
    next.set(this.#bytes);
    this.#bytes = next;
  }
}

function quoteExactJsonString(value: string): string {
  let result = '"';

  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);

    switch (code) {
      case 0x08:
        result += "\\b";
        continue;
      case 0x09:
        result += "\\t";
        continue;
      case 0x0a:
        result += "\\n";
        continue;
      case 0x0c:
        result += "\\f";
        continue;
      case 0x0d:
        result += "\\r";
        continue;
      case 0x22:
        result += '\\"';
        continue;
      case 0x5c:
        result += "\\\\";
        continue;
      default:
        break;
    }

    if (code <= 0x1f) {
      result += `\\u00${code.toString(16).padStart(2, "0")}`;
      continue;
    }

    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);

      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        throw new TypeError(
          "canonical input strings cannot contain unpaired UTF-16 surrogates",
        );
      }

      result += value[index] ?? "";
      result += value[index + 1] ?? "";
      index += 1;
      continue;
    }

    if (code >= 0xdc00 && code <= 0xdfff) {
      throw new TypeError(
        "canonical input strings cannot contain unpaired UTF-16 surrogates",
      );
    }

    result += value[index] ?? "";
  }

  return `${result}"`;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function expectedType(path: SemaInputPath, expected: string): never {
  inputFailure("type", path, `expected ${expected}`);
}

function schemaFailure(message: string): never {
  inputFailure("schema", ["$schema"], message);
}

function inputFailure(
  reason: SemaInputErrorReason,
  path: SemaInputPath,
  message: string,
): never {
  throw new SemaInputError(reason, path, message);
}

function quoteForMessage(value: string): string {
  return JSON.stringify(value);
}

function formatPath(path: SemaInputPath): string {
  if (path.length === 0) {
    return "inputs";
  }

  return `inputs${path
    .map((segment) =>
      typeof segment === "number"
        ? `[${String(segment)}]`
        : `[${quoteForMessage(segment)}]`,
    )
    .join("")}`;
}
