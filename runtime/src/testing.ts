/**
 * `@semantscript/core/testing` (TASK-14.8): a stub artifact for testing
 * application code that calls sema expressions, without a trained model.
 *
 * The stub is an ordinary on-disk artifact derived from the project's IR
 * bundle (function ids, inputs, output heads, adapter and depth-routed encoder
 * references, runtime policy) with tiny generated ONNX graphs, and it loads
 * through `loadSemaArtifact`. The real inference worker runs every call, so
 * request scopes, pass counts, canonical-input validation, the confidence
 * policy and fallbacks behave exactly as with a trained artifact; only the
 * value each function answers comes from the test.
 */
import { randomBytes, createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { Buffer } from "node:buffer";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

import {
  loadSemaArtifact,
  type LoadSemaArtifactOptions,
  type SemaArtifactHandle,
  type SemaStageEntry,
  type SemaStageOutcome,
} from "./artifact-runtime.js";
import type {
  ArtifactFunctionV1,
  ArtifactResourceV1,
  HeadBindingV1,
  InputEntryV1,
  OnnxAbiV1,
  RuntimeHeadTypeV1,
} from "./artifact-types.js";
import {
  CANONICAL_INPUT_V1,
  CANONICAL_INPUT_V2,
  type CanonicalInputEntry,
  type CanonicalInputVersion,
  canonicalInputVersion,
  serializeCanonicalInputs,
} from "./canonical-input.js";
import { expectedValue, normalizedEntropy } from "./head-math.js";
import {
  type InferenceHeadPlan,
  type InferenceScalarDiagnostic,
  type InferenceSupportValue,
  type InferenceWorkerResult,
  stringifyInferenceResult,
} from "./inference-protocol.js";
import { parseInferenceResultPayload } from "./inference-runtime.js";
import {
  STUB_ONNX_OPSET,
  stubAdapterOnnx,
  stubEncoderOnnx,
  stubHeadOnnx,
  stubTokenizerJson,
} from "./stub-onnx.js";
import {
  registerStubAnswerer,
  SemaStubError,
  STUB_PROVENANCE,
  type StubAnswerRequest,
  unregisterStubAnswerer,
} from "./stub-registry.js";

export { SemaStubError, type SemaStubErrorReason } from "./stub-registry.js";

export type SemaStubScalar = boolean | string | number;
/** A scalar output's value, or a flat object output's fields. */
export type SemaStubValue =
  SemaStubScalar | Readonly<Record<string, SemaStubScalar>>;
/** One confidence for every field, or one per field of a flat object output (missing fields: 1). */
export type SemaStubConfidence = number | Readonly<Record<string, number>>;

/** A value with its confidence (default 1). It must exceed 1/K for a K-value head, so it stays the top answer. */
export interface SemaStubAnswer {
  readonly value: SemaStubValue;
  readonly confidence?: SemaStubConfidence;
}

export interface SemaStubInputCase extends SemaStubAnswer {
  /** The call's inputs by parameter name, matched on their canonical serialization. */
  readonly inputs: Readonly<Record<string, unknown>>;
}

/** A value per input; `otherwise` answers inputs no case matches (else the call throws). */
export interface SemaStubByInput {
  readonly byInput: readonly SemaStubInputCase[];
  readonly otherwise?: SemaStubScalar | SemaStubAnswer;
}

/** A function of the call's inputs, returning a scalar or `{ value, confidence }`. */
export interface SemaStubCompute {
  readonly compute: (
    inputs: Readonly<Record<string, unknown>>,
  ) => SemaStubScalar | SemaStubAnswer;
}

export type SemaStubAnswerSpec =
  SemaStubScalar | SemaStubAnswer | SemaStubByInput | SemaStubCompute;

/** A function id (`nf_...`), or the compiled function whose body calls exactly one sema expression. */
export type SemaStubFunctionKey =
  | string
  // Any compiled application function: only its source text is read.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ((...args: any[]) => unknown);

export type SemaStubAnswers =
  | ReadonlyMap<SemaStubFunctionKey, SemaStubAnswerSpec>
  | readonly (readonly [SemaStubFunctionKey, SemaStubAnswerSpec])[]
  | Readonly<Record<string, SemaStubAnswerSpec>>;

export interface CreateSemaStubArtifactOptions {
  /** What each function answers. A call to a function without an answer throws `SemaStubError` ("unanswered"). */
  readonly answers: SemaStubAnswers;
  /** Where to write the artifact root; default a fresh temporary directory, removed by `dispose()`. */
  readonly directory?: string;
  /** The manifest's application id (default "semantscript-stub"). */
  readonly applicationId?: string;
}

export interface SemaStubArtifact {
  /** The artifact root, for `loadSemaArtifact(root, options)`. */
  readonly root: string;
  readonly manifestSha256: string;
  readonly functionIds: ReadonlySet<string>;
  /** Unregisters the answers and removes a temporary root. The release cannot load afterwards. */
  dispose(): Promise<void>;
}

export interface LoadSemaStubArtifactOptions
  extends
    CreateSemaStubArtifactOptions,
    Omit<LoadSemaArtifactOptions, "watch" | "onReload" | "onReloadError"> {}

export interface SemaStubHandle extends SemaArtifactHandle {
  readonly root: string;
}

/** An IR bundle as the compiler emits it (`semantscript.ir.v1.json`), or its path. */
export type SemaIrBundleSource = string | Readonly<Record<string, unknown>>;

type JsonRecord = Readonly<Record<string, unknown>>;

interface StubHead {
  readonly field: string | undefined;
  readonly support: readonly InferenceSupportValue[];
}

interface StubFunction {
  readonly id: string;
  readonly inputs: readonly CanonicalInputEntry[];
  readonly heads: readonly StubHead[];
}

interface HeadAnswer {
  readonly value: InferenceSupportValue;
  readonly confidence: number;
}

type PreparedAnswer =
  | { readonly kind: "fixed"; readonly heads: readonly HeadAnswer[] }
  | {
      readonly kind: "by-input";
      readonly cases: ReadonlyMap<string, readonly HeadAnswer[]>;
      readonly otherwise: readonly HeadAnswer[] | undefined;
    }
  | {
      readonly kind: "compute";
      readonly compute: SemaStubCompute["compute"];
    };

const ZERO_SHA256 = "0".repeat(64);
const STUB_VERSION = "0.0.0";
const MAXIMUM_SEQUENCE_LENGTH = 128;
const FUNCTION_ID = /^nf_[0-9a-f]{64}$/u;
const FUNCTION_ID_IN_SOURCE = /nf_[0-9a-f]{64}/gu;
const APPLICATION_ID = /^[a-z][a-z0-9._-]{1,127}$/u;
const EMBEDDING = [
  { name: "sentence_embedding", dtype: "float32", shape: ["BATCH", 1] },
] as const;
const FUNCTION_EMBEDDING = [
  { name: "function_embedding", dtype: "float32", shape: ["BATCH", 1] },
] as const;

/**
 * Writes a stub artifact for `bundle` and registers `options.answers` for it in
 * this process. Load it with `loadSemaArtifact(stub.root)` (or use
 * `loadSemaStubArtifact`), and `dispose()` it when the test ends.
 */
export async function createSemaStubArtifact(
  bundle: SemaIrBundleSource,
  options: CreateSemaStubArtifactOptions,
): Promise<SemaStubArtifact> {
  const ir = await readBundle(bundle);
  const applicationId = options.applicationId ?? STUB_PROVENANCE;
  if (!APPLICATION_ID.test(applicationId)) {
    throw new SemaStubError(
      "invalid-bundle",
      `applicationId ${JSON.stringify(applicationId)} must match ${APPLICATION_ID.source}`,
    );
  }
  const derived = deriveArtifact(ir.functions, ir.digest, applicationId);
  const answers = prepareAnswers(
    options.answers,
    derived.stubFunctions,
    derived.canonicalInputVersion,
  );

  const temporary = options.directory === undefined;
  const root =
    options.directory === undefined
      ? await mkdtemp(join(tmpdir(), "semantscript-stub-"))
      : resolve(options.directory);
  let manifestSha256: string;
  try {
    manifestSha256 = await writeRelease(root, derived);
  } catch (error) {
    if (temporary) await rm(root, { recursive: true, force: true });
    throw error;
  }

  const functions = derived.stubFunctions;
  registerStubAnswerer(manifestSha256, (request) =>
    answerCall(request, functions, answers),
  );
  let disposed = false;
  return Object.freeze({
    root,
    manifestSha256,
    functionIds: new Set(functions.keys()),
    async dispose(): Promise<void> {
      if (disposed) return;
      disposed = true;
      unregisterStubAnswerer(manifestSha256);
      if (temporary) await rm(root, { recursive: true, force: true });
    },
  });
}

/**
 * Creates a stub artifact for `bundle` and loads it as the active artifact, the
 * way `loadSemaArtifact` loads a trained one (`fallbacks` and the other load
 * options pass through). `close()` closes it and disposes the stub.
 */
export async function loadSemaStubArtifact(
  bundle: SemaIrBundleSource,
  options: LoadSemaStubArtifactOptions,
): Promise<SemaStubHandle> {
  const { answers, directory, applicationId, ...loadOptions } = options;
  const stub = await createSemaStubArtifact(bundle, {
    answers,
    ...(directory === undefined ? {} : { directory }),
    ...(applicationId === undefined ? {} : { applicationId }),
  });
  let handle: SemaArtifactHandle;
  try {
    handle = await loadSemaArtifact(stub.root, loadOptions);
  } catch (error) {
    await stub.dispose();
    throw error;
  }
  return Object.freeze({
    root: stub.root,
    manifestSha256: handle.manifestSha256,
    functionIds: handle.functionIds,
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
    call<T>(functionId: string, inputs: Readonly<Record<string, unknown>>): T {
      return handle.call<T>(functionId, inputs);
    },
    callStage(entries: readonly SemaStageEntry[]): SemaStageOutcome {
      return handle.callStage(entries);
    },
    async close(): Promise<void> {
      try {
        await handle.close();
      } finally {
        await stub.dispose();
      }
    },
  });
}

/** The id of the one sema expression a compiled function calls, or the id itself. */
export function semaFunctionId(key: SemaStubFunctionKey): string {
  if (typeof key === "string") {
    if (!FUNCTION_ID.test(key)) {
      throw new SemaStubError(
        "unknown-function",
        `${JSON.stringify(key)} is not a semantic function id (nf_ and 64 hex digits)`,
      );
    }
    return key;
  }
  if (typeof key !== "function") {
    throw new SemaStubError(
      "unresolved-function",
      "an answer key must be a function id or a compiled function",
    );
  }
  const source = Function.prototype.toString.call(key);
  const ids = new Set(source.match(FUNCTION_ID_IN_SOURCE) ?? []);
  const [id] = ids;
  if (ids.size !== 1 || id === undefined) {
    throw new SemaStubError(
      "unresolved-function",
      `function ${JSON.stringify(key.name)} calls ${String(ids.size)} sema expressions; key its answer by function id instead`,
    );
  }
  return id;
}

// ---------------------------------------------------------------------------
// IR bundle -> manifest

interface ReadBundle {
  readonly functions: readonly JsonRecord[];
  readonly digest: string;
}

async function readBundle(source: SemaIrBundleSource): Promise<ReadBundle> {
  let value: unknown = source;
  if (typeof source === "string") {
    let text: string;
    try {
      text = await readFile(source, "utf8");
    } catch (error) {
      throw new SemaStubError(
        "invalid-bundle",
        `cannot read the IR bundle ${source}: ${errorMessage(error)}`,
      );
    }
    try {
      value = JSON.parse(text) as unknown;
    } catch (error) {
      throw new SemaStubError(
        "invalid-bundle",
        `the IR bundle ${source} is not JSON: ${errorMessage(error)}`,
      );
    }
  }
  const bundle = record(value, "the IR bundle");
  if (bundle["kind"] !== "semantscript.ir-bundle") {
    invalidBundle('the IR bundle kind must be "semantscript.ir-bundle"');
  }
  const functions = bundle["functions"];
  if (!Array.isArray(functions) || functions.length === 0) {
    invalidBundle("the IR bundle has no semantic functions");
  }
  return {
    functions: functions.map((fn, index) =>
      record(fn, `IR function ${String(index)}`),
    ),
    digest: sha256(new TextEncoder().encode(JSON.stringify(bundle))),
  };
}

interface DerivedArtifact {
  readonly manifest: JsonRecord;
  readonly files: ReadonlyMap<string, Uint8Array>;
  readonly stubFunctions: ReadonlyMap<string, StubFunction>;
  readonly canonicalInputVersion: CanonicalInputVersion;
}

interface IrHead {
  readonly field: string | undefined;
  readonly ref: string;
  readonly type: RuntimeHeadTypeV1;
  readonly logits: number;
}

interface IrFunction {
  readonly source: JsonRecord;
  readonly id: string;
  readonly inputs: readonly InputEntryV1[];
  readonly adapterRef: string;
  readonly encoderRef: string;
  readonly depthKey: string;
  readonly heads: readonly IrHead[];
  readonly runtime: ArtifactFunctionV1["runtime"];
  readonly canonicalInput: string;
}

function deriveArtifact(
  irFunctions: readonly JsonRecord[],
  bundleDigest: string,
  applicationId: string,
): DerivedArtifact {
  const functions = irFunctions.map(parseIrFunction);
  const ids = new Set<string>();
  for (const fn of functions) {
    if (ids.has(fn.id)) invalidBundle(`duplicate function id ${fn.id}`);
    ids.add(fn.id);
  }
  const encodings = new Set(functions.map((fn) => fn.canonicalInput));
  const [encoding] = encodings;
  const version =
    encoding === undefined ? undefined : canonicalInputVersion(encoding);
  if (encodings.size !== 1 || encoding === undefined || version === undefined) {
    invalidBundle("the IR functions disagree on their canonical input");
  }

  // The model's encoder is the full stack when any function runs it, else the
  // deepest prefix, as the trainer exports it.
  const encoderByDepth = new Map<string, string>();
  for (const fn of functions) {
    const known = encoderByDepth.get(fn.depthKey);
    if (known !== undefined && known !== fn.encoderRef) {
      invalidBundle(
        `functions at encoder depth ${fn.depthKey} name different encoders`,
      );
    }
    encoderByDepth.set(fn.depthKey, fn.encoderRef);
  }
  const depthKeys = [...encoderByDepth.keys()].sort();
  const applicationDepth = encoderByDepth.has("full")
    ? "full"
    : (depthKeys.at(-1) ?? "full");
  const applicationEncoder = encoderByDepth.get(applicationDepth) ?? "";

  const files = new Map<string, Uint8Array>();
  const resources: Record<string, unknown>[] = [];
  const addResource = (
    ref: string,
    role: ArtifactResourceV1["role"],
    path: string,
    bytes: Uint8Array,
    onnx?: OnnxAbiV1,
  ): void => {
    files.set(path, bytes);
    resources.push({
      ref,
      role,
      format: onnx === undefined ? "tokenizer-json" : "onnx",
      formatVersion: 1,
      path,
      byteLength: bytes.byteLength,
      sha256: sha256(bytes),
      ...(onnx === undefined
        ? { maximumSequenceLength: MAXIMUM_SEQUENCE_LENGTH }
        : { onnx }),
    });
  };
  const tokenizerRef = uniqueRef("tokenizer.stub", functions);
  addResource(
    tokenizerRef,
    "tokenizer",
    "tokenizer/tokenizer.json",
    stubTokenizerJson(),
  );
  const encoderAbi: OnnxAbiV1 = {
    opset: STUB_ONNX_OPSET,
    inputs: [
      { name: "input_ids", dtype: "int64", shape: ["BATCH", "SEQUENCE"] },
      { name: "attention_mask", dtype: "int64", shape: ["BATCH", "SEQUENCE"] },
    ],
    outputs: EMBEDDING,
    externalData: false,
  };
  const encoderBytes = stubEncoderOnnx();
  addResource(
    applicationEncoder,
    "encoder",
    "models/encoder/model.onnx",
    encoderBytes,
    encoderAbi,
  );
  for (const [index, depth] of depthKeys.entries()) {
    const ref = encoderByDepth.get(depth);
    if (depth === applicationDepth || ref === undefined) continue;
    addResource(
      ref,
      "encoder",
      `models/encoder/prefix-${String(index).padStart(3, "0")}.onnx`,
      encoderBytes,
      encoderAbi,
    );
  }
  const adapterRefs = [...new Set(functions.map((fn) => fn.adapterRef))];
  const adapterBytes = stubAdapterOnnx();
  for (const [index, ref] of adapterRefs.entries()) {
    addResource(
      ref,
      "adapter",
      `models/adapters/adapter-${String(index).padStart(3, "0")}.onnx`,
      adapterBytes,
      {
        opset: STUB_ONNX_OPSET,
        inputs: EMBEDDING,
        outputs: FUNCTION_EMBEDDING,
        externalData: false,
      },
    );
  }

  const trainingKey = randomBytes(32).toString("hex");
  const manifestFunctions: ArtifactFunctionV1[] = [];
  const stubFunctions = new Map<string, StubFunction>();
  for (const fn of functions) {
    const heads: HeadBindingV1[] = [];
    for (const [index, head] of fn.heads.entries()) {
      addResource(
        head.ref,
        "head",
        `models/heads/${fn.id}/head-${String(index).padStart(3, "0")}.onnx`,
        stubHeadOnnx(head.logits),
        {
          opset: STUB_ONNX_OPSET,
          inputs: FUNCTION_EMBEDDING,
          outputs: [
            { name: "logits", dtype: "float32", shape: ["BATCH", head.logits] },
          ],
          externalData: false,
        },
      );
      heads.push({
        outputPath: head.field === undefined ? [] : [head.field],
        headRef: head.ref,
        type: head.type,
        parameterization:
          head.type.kind === "boolean"
            ? "binary-sigmoid"
            : "categorical-softmax",
        calibration: {
          method: "temperature-scaling",
          temperature: 1,
          ece: 0,
          brier: 0,
          sampleCount: 1,
          splitSha256: ZERO_SHA256,
          eceBins: 2,
        },
        verification: { accuracy: 1, pairConsistency: 1 },
      });
    }
    const output = fn.source["output"];
    manifestFunctions.push({
      id: fn.id,
      semanticSha256: sha256Field(fn.source, "semanticSha256", fn.id),
      inputs: fn.inputs,
      inputSchemaSha256: semanticSha256(fn.inputs),
      outputSchemaSha256: semanticSha256(output),
      adapterRef: fn.adapterRef,
      ...(fn.encoderRef === applicationEncoder
        ? {}
        : { encoderRef: fn.encoderRef }),
      heads,
      runtime: fn.runtime,
      verification: {
        status: "passed",
        accuracy: 1,
        ece: 0,
        brier: 0,
        pairConsistency: 1,
        attestedCases: 1,
        exampleFailures: 0,
        constraintViolations: 0,
        typeErrors: 0,
      },
      trainingProvenance: {
        datasetSha256: ZERO_SHA256,
        trainingKeySha256: trainingKey,
        teacher: STUB_PROVENANCE,
        baseModel: STUB_PROVENANCE,
      },
    });
    stubFunctions.set(fn.id, {
      id: fn.id,
      inputs: fn.inputs,
      heads: fn.heads.map((head) => ({
        field: head.field,
        support: headSupport(head.type),
      })),
    });
  }

  const firstAdapter = adapterRefs[0] ?? "";
  const manifest = {
    kind: "semantscript.application-artifact",
    artifactVersion: 1,
    irVersion: 1,
    compatibility: {
      runtimeAbiVersion: 1,
      modelAbiVersion: 1,
      canonicalInput: encoding,
      minimumRuntimeVersion: STUB_VERSION,
      requiredCapabilities: [],
    },
    application: { id: applicationId, version: STUB_VERSION },
    build: {
      createdAt: new Date().toISOString(),
      compilerVersion: STUB_VERSION,
      trainerVersion: STUB_VERSION,
      sourceIrSha256: bundleDigest,
    },
    resources,
    model: {
      tokenizerRef,
      encoderRef: applicationEncoder,
      adapterRef: firstAdapter,
    },
    functions: manifestFunctions,
  };
  return {
    manifest,
    files,
    stubFunctions,
    canonicalInputVersion: version,
  };
}

function parseIrFunction(fn: JsonRecord, index: number): IrFunction {
  const where = `IR function ${String(index)}`;
  const id = fn["id"];
  if (typeof id !== "string" || !FUNCTION_ID.test(id)) {
    invalidBundle(`${where} has no valid id`);
  }
  const inputs = fn["inputs"];
  if (!Array.isArray(inputs)) invalidBundle(`${id} has no inputs list`);
  const model = record(fn["model"], `${id}.model`);
  const adapterRef = stringField(model, "adapter", id);
  const encoderRef = stringField(model, "encoder", id);
  const depth = model["encoderDepth"];
  if (
    depth !== undefined &&
    (!Number.isSafeInteger(depth) || Number(depth) < 1)
  ) {
    invalidBundle(`${id}.model.encoderDepth is invalid`);
  }
  const depthKey =
    typeof depth === "number"
      ? `depth-${depth.toString().padStart(3, "0")}`
      : "full";
  const bindings = model["heads"];
  if (!Array.isArray(bindings)) invalidBundle(`${id}.model.heads is invalid`);

  const output = record(fn["output"], `${id}.output`);
  const specs: { field: string | undefined; head: JsonRecord }[] = [];
  if (output["kind"] === "scalar") {
    specs.push({
      field: undefined,
      head: record(output["head"], `${id}.output.head`),
    });
  } else if (output["kind"] === "object") {
    const fields = output["fields"];
    if (!Array.isArray(fields) || fields.length === 0) {
      invalidBundle(`${id}.output.fields is invalid`);
    }
    for (const [fieldIndex, field] of fields.entries()) {
      const entry = record(field, `${id}.output.fields[${String(fieldIndex)}]`);
      specs.push({
        field: stringField(entry, "name", id),
        head: record(entry["head"], `${id}.output.fields.head`),
      });
    }
  } else {
    invalidBundle(`${id}.output.kind must be scalar or object`);
  }
  if (bindings.length !== specs.length) {
    invalidBundle(`${id}.model.heads does not match its output`);
  }
  const heads = specs.map(({ field, head }, headIndex): IrHead => {
    const binding = record(
      bindings[headIndex],
      `${id}.model.heads[${String(headIndex)}]`,
    );
    const type = runtimeHeadType(head, id);
    return {
      field,
      ref: stringField(binding, "ref", id),
      type,
      logits:
        type.kind === "boolean"
          ? 1
          : type.kind === "ordinal-number"
            ? type.supportDecimal.length
            : type.support.length,
    };
  });

  const provenance = fn["trainingProvenance"];
  const declared =
    provenance !== null &&
    typeof provenance === "object" &&
    !Array.isArray(provenance)
      ? (provenance as JsonRecord)["canonicalInput"]
      : undefined;
  const canonicalInput =
    typeof declared === "string" ? declared : CANONICAL_INPUT_V1;
  if (
    canonicalInput !== CANONICAL_INPUT_V1 &&
    canonicalInput !== CANONICAL_INPUT_V2
  ) {
    invalidBundle(`${id} declares an unknown canonical input`);
  }

  return {
    source: fn,
    id,
    inputs: inputs as readonly InputEntryV1[],
    adapterRef,
    encoderRef,
    depthKey,
    heads,
    runtime: runtimePolicy(record(fn["runtime"], `${id}.runtime`), output, id),
    canonicalInput,
  };
}

function runtimeHeadType(head: JsonRecord, id: string): RuntimeHeadTypeV1 {
  const sourceKind = head["sourceKind"];
  if (sourceKind === "boolean") {
    return { kind: "boolean", support: [false, true] };
  }
  if (sourceKind === "bounded-int" || sourceKind === "bounded-number") {
    const supportDecimal = head["supportDecimal"];
    if (
      !Array.isArray(supportDecimal) ||
      !supportDecimal.every((value) => typeof value === "string")
    ) {
      invalidBundle(`${id} bounded output supportDecimal is invalid`);
    }
    return {
      kind: "ordinal-number",
      sourceKind,
      minimum: stringField(head, "minimum", id),
      maximum: stringField(head, "maximum", id),
      step: stringField(head, "step", id),
      supportDecimal,
    };
  }
  const support = head["support"];
  if (!Array.isArray(support) || support.length < 2) {
    invalidBundle(`${id} output support is invalid`);
  }
  if (support.every((value) => typeof value === "string")) {
    return {
      kind: head["kind"] === "ordinal" ? "ordinal-string" : "nominal-string",
      support,
    };
  }
  if (
    head["kind"] === "nominal" &&
    support.every((value) => typeof value === "number")
  ) {
    return { kind: "nominal-number", support };
  }
  invalidBundle(`${id} output head cannot be represented by the model ABI`);
}

function runtimePolicy(
  runtime: JsonRecord,
  output: JsonRecord,
  id: string,
): ArtifactFunctionV1["runtime"] {
  const resultMode = runtime["resultMode"];
  const threshold = runtime["confidenceThreshold"];
  const fallbackRef = runtime["fallbackRef"];
  if (resultMode !== "value" && resultMode !== "diagnostic") {
    invalidBundle(`${id}.runtime.resultMode is invalid`);
  }
  if (
    threshold !== null &&
    (typeof threshold !== "number" || !(threshold >= 0 && threshold <= 1))
  ) {
    invalidBundle(`${id}.runtime.confidenceThreshold is invalid`);
  }
  if (fallbackRef !== null && typeof fallbackRef !== "string") {
    invalidBundle(`${id}.runtime.fallbackRef is invalid`);
  }
  return {
    resultMode,
    confidenceThreshold: threshold,
    policy:
      threshold === null
        ? "none"
        : output["kind"] === "scalar"
          ? "scalar-top1"
          : "all-fields",
    fallbackRef: threshold === null ? null : fallbackRef,
  };
}

function headSupport(
  type: RuntimeHeadTypeV1,
): readonly InferenceSupportValue[] {
  return type.kind === "ordinal-number"
    ? type.supportDecimal.map((value) => Number(value))
    : type.support;
}

function uniqueRef(
  preferred: string,
  functions: readonly IrFunction[],
): string {
  const taken = new Set<string>();
  for (const fn of functions) {
    taken.add(fn.adapterRef);
    taken.add(fn.encoderRef);
    for (const head of fn.heads) taken.add(head.ref);
  }
  let ref = preferred;
  for (let suffix = 1; taken.has(ref); suffix += 1) {
    ref = `${preferred}-${String(suffix)}`;
  }
  return ref;
}

async function writeRelease(
  root: string,
  derived: DerivedArtifact,
): Promise<string> {
  await mkdir(root, { recursive: true });
  const staging = await mkdtemp(join(root, ".staging-"));
  try {
    for (const [path, bytes] of derived.files) {
      const destination = join(staging, path);
      await mkdir(dirname(destination), { recursive: true });
      await writeFile(destination, bytes);
    }
    const manifestBytes = new TextEncoder().encode(
      `${JSON.stringify(derived.manifest, null, 2)}\n`,
    );
    const manifestSha256 = sha256(manifestBytes);
    await writeFile(join(staging, "manifest.json"), manifestBytes);
    const release = `releases/sha256-${manifestSha256}`;
    await mkdir(join(root, "releases"), { recursive: true });
    await rename(staging, join(root, release));
    const pointer = {
      kind: "semantscript.artifact-pointer",
      pointerVersion: 1,
      release,
      manifestSha256,
    };
    const pointerPath = join(root, "current.json");
    const temporaryPointer = `${pointerPath}.${randomBytes(6).toString("hex")}.tmp`;
    await writeFile(temporaryPointer, `${JSON.stringify(pointer, null, 2)}\n`);
    await rename(temporaryPointer, pointerPath);
    return manifestSha256;
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// Answers

function answerEntries(
  answers: SemaStubAnswers,
): (readonly [SemaStubFunctionKey, SemaStubAnswerSpec])[] {
  if (answers instanceof Map) {
    return [
      ...(answers as ReadonlyMap<SemaStubFunctionKey, SemaStubAnswerSpec>),
    ];
  }
  if (Array.isArray(answers)) {
    return (answers as readonly unknown[]).map((entry, index) => {
      if (!Array.isArray(entry) || entry.length !== 2) {
        throw new SemaStubError(
          "invalid-value",
          `answers[${String(index)}] must be a [function, answer] pair`,
        );
      }
      return entry as unknown as readonly [
        SemaStubFunctionKey,
        SemaStubAnswerSpec,
      ];
    });
  }
  const untyped: unknown = answers;
  if (untyped === null || typeof untyped !== "object") {
    throw new SemaStubError(
      "invalid-value",
      "answers must be a Map, a list of [function, answer] pairs or an object keyed by function id",
    );
  }
  return Object.entries(
    answers as Readonly<Record<string, SemaStubAnswerSpec>>,
  );
}

function prepareAnswers(
  answers: SemaStubAnswers,
  functions: ReadonlyMap<string, StubFunction>,
  version: CanonicalInputVersion,
): ReadonlyMap<string, PreparedAnswer> {
  const prepared = new Map<string, PreparedAnswer>();
  for (const [key, spec] of answerEntries(answers)) {
    const id = semaFunctionId(key);
    const fn = functions.get(id);
    if (fn === undefined) {
      throw new SemaStubError(
        "unknown-function",
        `${id} is not a semantic function of the IR bundle`,
        id,
      );
    }
    if (prepared.has(id)) {
      throw new SemaStubError(
        "invalid-value",
        `${id} is answered more than once`,
        id,
      );
    }
    prepared.set(id, prepareAnswer(fn, spec, version));
  }
  return prepared;
}

function prepareAnswer(
  fn: StubFunction,
  spec: SemaStubAnswerSpec,
  version: CanonicalInputVersion,
): PreparedAnswer {
  if (isRecord(spec) && "compute" in spec) {
    const compute = spec["compute"];
    if (typeof compute !== "function") {
      throw new SemaStubError(
        "invalid-value",
        `${fn.id}: compute must be a function`,
        fn.id,
      );
    }
    return {
      kind: "compute",
      compute: compute as SemaStubCompute["compute"],
    };
  }
  if (isRecord(spec) && "byInput" in spec) {
    const cases = spec["byInput"];
    if (!Array.isArray(cases)) {
      throw new SemaStubError(
        "invalid-value",
        `${fn.id}: byInput must be a list of { inputs, value, confidence? }`,
        fn.id,
      );
    }
    const byKey = new Map<string, readonly HeadAnswer[]>();
    for (const [index, entry] of (cases as readonly unknown[]).entries()) {
      if (!isRecord(entry) || !isRecord(entry["inputs"])) {
        throw new SemaStubError(
          "invalid-value",
          `${fn.id}: byInput[${String(index)}] needs an inputs record`,
          fn.id,
        );
      }
      const { inputs, ...answer } = entry;
      byKey.set(
        canonicalKey(fn, inputs, version),
        headAnswers(fn, answer, `byInput[${String(index)}]`),
      );
    }
    const otherwise = spec["otherwise"];
    return {
      kind: "by-input",
      cases: byKey,
      otherwise:
        otherwise === undefined
          ? undefined
          : headAnswers(fn, answerOf(otherwise), "otherwise"),
    };
  }
  return { kind: "fixed", heads: headAnswers(fn, answerOf(spec), "answer") };
}

function answerOf(spec: unknown): unknown {
  return isRecord(spec) ? spec : { value: spec };
}

function canonicalKey(
  fn: StubFunction,
  inputs: Readonly<Record<string, unknown>>,
  version: CanonicalInputVersion,
): string {
  try {
    return hexOf(serializeCanonicalInputs(fn.inputs, inputs, { version }));
  } catch (error) {
    throw new SemaStubError(
      "invalid-value",
      `${fn.id}: byInput inputs do not match the function's inputs: ${errorMessage(error)}`,
      fn.id,
    );
  }
}

/** Validates `{ value, confidence? }` against the function's heads, in head order. */
function headAnswers(
  fn: StubFunction,
  answer: unknown,
  where: string,
): readonly HeadAnswer[] {
  if (!isRecord(answer) || !("value" in answer)) {
    throw new SemaStubError(
      "invalid-value",
      `${fn.id}: ${where} must be a scalar or { value, confidence? }`,
      fn.id,
    );
  }
  const extra = Object.keys(answer).filter(
    (key) => key !== "value" && key !== "confidence",
  );
  if (extra.length > 0) {
    throw new SemaStubError(
      "invalid-value",
      `${fn.id}: ${where} has unexpected keys ${extra.join(", ")}`,
      fn.id,
    );
  }
  const value = answer["value"];
  const confidence = answer["confidence"] ?? 1;
  const scalar = fn.heads.length === 1 && fn.heads[0]?.field === undefined;
  if (scalar) {
    const head = fn.heads[0];
    if (head === undefined) throw new Error("unreachable: scalar head");
    if (typeof confidence !== "number") {
      throw new SemaStubError(
        "invalid-confidence",
        `${fn.id}: ${where} confidence of a scalar output must be a number`,
        fn.id,
      );
    }
    return [headAnswer(fn, head, value, confidence, where)];
  }
  if (!isRecord(value)) {
    throw new SemaStubError(
      "invalid-value",
      `${fn.id}: ${where} value must be an object with fields ${fn.heads.map((head) => head.field).join(", ")}`,
      fn.id,
    );
  }
  const fields = new Set(fn.heads.map((head) => head.field ?? ""));
  for (const key of Object.keys(value)) {
    if (!fields.has(key)) {
      throw new SemaStubError(
        "invalid-value",
        `${fn.id}: ${where} value has unexpected field ${JSON.stringify(key)}`,
        fn.id,
      );
    }
  }
  if (typeof confidence !== "number") {
    if (!isRecord(confidence)) {
      throw new SemaStubError(
        "invalid-confidence",
        `${fn.id}: ${where} confidence must be a number or a record per field`,
        fn.id,
      );
    }
    for (const key of Object.keys(confidence)) {
      if (!fields.has(key)) {
        throw new SemaStubError(
          "invalid-confidence",
          `${fn.id}: ${where} confidence names unknown field ${JSON.stringify(key)}`,
          fn.id,
        );
      }
    }
  }
  return fn.heads.map((head) => {
    const field = head.field ?? "";
    if (!Object.prototype.hasOwnProperty.call(value, field)) {
      throw new SemaStubError(
        "invalid-value",
        `${fn.id}: ${where} value is missing field ${JSON.stringify(field)}`,
        fn.id,
      );
    }
    const fieldConfidence =
      typeof confidence === "number" ? confidence : (confidence[field] ?? 1);
    if (typeof fieldConfidence !== "number") {
      throw new SemaStubError(
        "invalid-confidence",
        `${fn.id}: ${where} confidence of ${JSON.stringify(field)} must be a number`,
        fn.id,
      );
    }
    return headAnswer(
      fn,
      head,
      value[field],
      fieldConfidence,
      `${where}.${field}`,
    );
  });
}

function headAnswer(
  fn: StubFunction,
  head: StubHead,
  value: unknown,
  confidence: number,
  where: string,
): HeadAnswer {
  const supported = head.support.find((candidate) =>
    Object.is(candidate, value),
  );
  if (supported === undefined) {
    throw new SemaStubError(
      "invalid-value",
      `${fn.id}: ${where} ${JSON.stringify(value)} is not one of ${JSON.stringify(head.support)}`,
      fn.id,
    );
  }
  const floor = 1 / head.support.length;
  if (!Number.isFinite(confidence) || confidence <= floor || confidence > 1) {
    throw new SemaStubError(
      "invalid-confidence",
      `${fn.id}: ${where} confidence ${String(confidence)} must be above ${String(floor)} (one over the ${String(head.support.length)} possible values) and at most 1, so the answer stays the top value`,
      fn.id,
    );
  }
  return { value: supported, confidence };
}

function answerCall(
  request: StubAnswerRequest,
  functions: ReadonlyMap<string, StubFunction>,
  answers: ReadonlyMap<string, PreparedAnswer>,
): unknown {
  const fn = functions.get(request.functionId);
  const answer = answers.get(request.functionId);
  if (fn === undefined || answer === undefined) {
    throw new SemaStubError(
      "unanswered",
      `the stub artifact has no answer for ${request.functionId}; add one to its answers`,
      request.functionId,
    );
  }
  let heads: readonly HeadAnswer[];
  if (answer.kind === "fixed") {
    heads = answer.heads;
  } else if (answer.kind === "compute") {
    heads = headAnswers(
      fn,
      answerOf(answer.compute(request.inputs)),
      "compute()",
    );
  } else {
    const resolved =
      answer.cases.get(hexOf(request.canonicalInput)) ?? answer.otherwise;
    if (resolved === undefined) {
      throw new SemaStubError(
        "unmatched-input",
        `the stub artifact has no byInput case for this ${request.functionId} input and no otherwise`,
        request.functionId,
      );
    }
    heads = resolved;
  }
  return decodedResult(request, heads);
}

/**
 * Builds the worker's result for the stubbed answer and decodes it with the
 * runtime's own decoder, so the value the application sees is exactly what a
 * real head with this distribution would produce.
 */
function decodedResult(
  request: StubAnswerRequest,
  answers: readonly HeadAnswer[],
): unknown {
  const { plan } = request;
  const scalar =
    plan.heads.length === 1 && plan.heads[0]?.outputPath.length === 0;
  const evaluations = plan.heads.map((head, index) => {
    const answer = answers[index];
    if (answer === undefined) throw new Error("unreachable: head answer");
    return { head, answer, diagnostic: scalarDiagnostic(head, answer) };
  });
  let result: InferenceWorkerResult;
  if (scalar) {
    const [evaluation] = evaluations;
    if (evaluation === undefined) throw new Error("unreachable: scalar head");
    result = plan.diagnosticsRequired
      ? { kind: "scalar", result: evaluation.diagnostic }
      : { kind: "value", result: evaluation.answer.value };
  } else {
    const value: Record<string, InferenceSupportValue> = {};
    const fields: Record<string, InferenceScalarDiagnostic> = {};
    let minimumFieldConfidence = 1;
    let maximumFieldUncertainty = 0;
    for (const { head, answer, diagnostic } of evaluations) {
      const field = head.outputPath[0] ?? "";
      defineField(value, field, answer.value);
      defineField(fields, field, diagnostic);
      minimumFieldConfidence = Math.min(
        minimumFieldConfidence,
        diagnostic.confidence,
      );
      maximumFieldUncertainty = Math.max(
        maximumFieldUncertainty,
        diagnostic.uncertainty,
      );
    }
    result = plan.diagnosticsRequired
      ? {
          kind: "object",
          result: {
            value,
            minimumFieldConfidence,
            maximumFieldUncertainty,
            fields,
          },
        }
      : { kind: "value", result: value };
  }
  return parseInferenceResultPayload(stringifyInferenceResult(result), plan);
}

function scalarDiagnostic(
  head: InferenceHeadPlan,
  answer: HeadAnswer,
): InferenceScalarDiagnostic {
  const index = head.support.findIndex((candidate) =>
    Object.is(candidate, answer.value),
  );
  const { confidence } = answer;
  let probabilities: number[];
  if (head.parameterization === "binary-sigmoid") {
    probabilities =
      index === 1 ? [1 - confidence, confidence] : [confidence, 1 - confidence];
  } else {
    const rest = (1 - confidence) / (head.support.length - 1);
    probabilities = head.support.map((_, position) =>
      position === index ? confidence : rest,
    );
  }
  const invalid = (message: string): Error =>
    new SemaStubError("invalid-confidence", message);
  if (
    probabilities.some(
      (probability, position) =>
        position !== index && probability >= confidence,
    )
  ) {
    throw invalid(
      `confidence ${String(confidence)} does not keep ${JSON.stringify(answer.value)} the top value`,
    );
  }
  return {
    value: answer.value,
    confidence,
    uncertainty: normalizedEntropy(probabilities, invalid),
    distribution: head.support.map((value, position) => ({
      value,
      probability: probabilities[position] ?? 0,
    })),
    expectedValue: expectedValue(head, probabilities, invalid),
  };
}

// ---------------------------------------------------------------------------
// Helpers

function defineField<T>(
  target: Record<string, T>,
  field: string,
  value: T,
): void {
  Object.defineProperty(target, field, {
    configurable: true,
    enumerable: true,
    value,
    writable: true,
  });
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function record(value: unknown, where: string): JsonRecord {
  if (!isRecord(value)) invalidBundle(`${where} must be an object`);
  return value;
}

function stringField(value: JsonRecord, key: string, id: string): string {
  const field = value[key];
  if (typeof field !== "string" || field.length === 0) {
    invalidBundle(`${id}: ${key} must be a non-empty string`);
  }
  return field;
}

function sha256Field(value: JsonRecord, key: string, id: string): string {
  const field = stringField(value, key, id);
  if (!/^[0-9a-f]{64}$/u.test(field)) {
    invalidBundle(`${id}: ${key} must be a sha256 digest`);
  }
  return field;
}

function invalidBundle(message: string): never {
  throw new SemaStubError("invalid-bundle", message);
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function hexOf(bytes: Uint8Array): string {
  return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString(
    "hex",
  );
}

/** The semantic JSON digest the trainer and the loader use for schema hashes. */
function semanticSha256(value: unknown): string {
  return sha256(
    new TextEncoder().encode(
      JSON.stringify(["semantscript-semantic-json", 1, typedNode(value)]),
    ),
  );
}

type TypedNode =
  | readonly ["null"]
  | readonly ["boolean", boolean]
  | readonly ["number", string]
  | readonly ["string", string]
  | readonly ["array", readonly TypedNode[]]
  | readonly ["object", readonly (readonly [string, TypedNode])[]];

function typedNode(value: unknown): TypedNode {
  if (value === null) return ["null"];
  if (typeof value === "boolean") return ["boolean", value];
  if (typeof value === "string") return ["string", value];
  if (typeof value === "number") {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value, false);
    return ["number", hexOf(bytes)];
  }
  if (Array.isArray(value)) return ["array", value.map(typedNode)];
  const entries = Object.entries(value as JsonRecord)
    .map(([name, entry]) => [name, typedNode(entry)] as const)
    .sort(([left], [right]) =>
      Buffer.compare(Buffer.from(left), Buffer.from(right)),
    );
  return ["object", entries];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
