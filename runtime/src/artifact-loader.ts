import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";

import type {
  ApplicationArtifactManifestV1,
  ArtifactFunctionV1,
  ArtifactPointerV1,
  ArtifactResourceBackend,
  ArtifactResourceV1,
  HeadBindingV1,
  OnnxResourceV1,
  StagedArtifactDescriptor,
  StagedArtifactResource,
  TensorDescriptorV1,
} from "./artifact-types.js";
import { parseStrictJson, StrictJsonError } from "./strict-json.js";

const SHA256 = /^[a-f0-9]{64}$/u;
const RELEASE = /^sha256-([a-f0-9]{64})$/u;
const POINTER_RELEASE = /^releases\/sha256-([a-f0-9]{64})$/u;
const LOGICAL_REF = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$/u;
const PORTABLE_PATH =
  /^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*(?:\/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)*$/u;
const SEMVER =
  /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/u;
const FUNCTION_ID = /^nf_[a-f0-9]{64}$/u;
const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/u;
const SYMBOLIC_DIMENSION = /^[A-Z][A-Z0-9_]*$/u;
const DECIMAL = /^(?!-0$)-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$/u;

export type ArtifactLoadErrorCode =
  | "SEMA_ARTIFACT_INVALID_JSON"
  | "SEMA_ARTIFACT_INVALID_POINTER"
  | "SEMA_ARTIFACT_INVALID_MANIFEST"
  | "SEMA_ARTIFACT_INCOMPATIBLE"
  | "SEMA_ARTIFACT_INTEGRITY"
  | "SEMA_ARTIFACT_PATH"
  | "SEMA_ARTIFACT_QUOTA"
  | "SEMA_ARTIFACT_RESOURCE";

export class ArtifactLoadError extends Error {
  readonly code: ArtifactLoadErrorCode;
  readonly artifactPath: string | undefined;

  constructor(
    code: ArtifactLoadErrorCode,
    message: string,
    artifactPath?: string,
    cause?: unknown,
  ) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "ArtifactLoadError";
    this.code = code;
    this.artifactPath = artifactPath;
  }
}

export interface ArtifactLoadOptions<PreparedResource = Uint8Array> {
  readonly backend?: ArtifactResourceBackend<PreparedResource>;
  readonly runtimeVersion?: string;
  readonly supportedCapabilities?: ReadonlySet<string> | readonly string[];
  readonly supportedOnnxOpsets?: ReadonlySet<number> | readonly number[];
  readonly maximumPointerBytes?: number;
  readonly maximumManifestBytes?: number;
  readonly maximumResourceBytes?: number;
  readonly maximumAggregateResourceBytes?: number;
}

interface ResolvedOptions<PreparedResource> {
  readonly backend: ArtifactResourceBackend<PreparedResource>;
  readonly runtimeVersion: string;
  readonly capabilities: ReadonlySet<string>;
  readonly opsets: ReadonlySet<number>;
  readonly pointerBytes: number;
  readonly manifestBytes: number;
  readonly resourceBytes: number;
  readonly aggregateBytes: number;
}

interface SelectedRelease {
  readonly artifactRoot: string;
  readonly releaseDirectory: string;
  readonly digest: string;
}

interface JsonRecord {
  readonly [key: string]: unknown;
}

const defaultBackend: ArtifactResourceBackend<Uint8Array> = {
  prepareResource(_resource, bytes) {
    return bytes;
  },
};

/** Load an artifact root with current.json or a direct sha256 release directory. */
export async function loadArtifact<PreparedResource = Uint8Array>(
  artifactPath: string,
  options: ArtifactLoadOptions<PreparedResource> = {},
): Promise<StagedArtifactDescriptor<PreparedResource>> {
  const settings = resolveOptions(options);
  const selected = await selectRelease(artifactPath, settings.pointerBytes);
  const manifestPath = join(selected.releaseDirectory, "manifest.json");
  const manifestBytes = await readSafeFile(
    selected.releaseDirectory,
    "manifest.json",
    settings.manifestBytes,
    "manifest",
  );
  const manifestDigest = digest(manifestBytes);

  if (manifestDigest !== selected.digest) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_INTEGRITY",
      `manifest digest ${manifestDigest} does not match selected release digest ${selected.digest}`,
      manifestPath,
    );
  }

  const parsed = parseJson(manifestBytes, settings.manifestBytes, manifestPath);
  validateManifest(parsed);
  const manifest = parsed;
  validateCompatibility(manifest, settings);
  validateRelationships(manifest, settings.opsets);

  let aggregate = 0;
  const verified: {
    readonly metadata: ArtifactResourceV1;
    readonly bytes: Uint8Array;
  }[] = [];

  for (const metadata of manifest.resources) {
    if (metadata.byteLength > settings.resourceBytes) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_QUOTA",
        `resource ${metadata.ref} exceeds quota`,
        metadata.path,
      );
    }

    aggregate += metadata.byteLength;
    if (
      !Number.isSafeInteger(aggregate) ||
      aggregate > settings.aggregateBytes
    ) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_QUOTA",
        "aggregate resource quota exceeded",
      );
    }

    const bytes = await readSafeFile(
      selected.releaseDirectory,
      metadata.path,
      settings.resourceBytes,
      `resource ${metadata.ref}`,
    );
    if (
      bytes.byteLength !== metadata.byteLength ||
      digest(bytes) !== metadata.sha256
    ) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_INTEGRITY",
        `resource ${metadata.ref} does not match its declared length and digest`,
        metadata.path,
      );
    }

    verified.push({ metadata, bytes });
  }

  const resources: StagedArtifactResource<PreparedResource>[] = [];
  try {
    for (const entry of verified) {
      const prepared = await settings.backend.prepareResource(
        entry.metadata,
        entry.bytes,
      );
      resources.push(Object.freeze({ metadata: entry.metadata, prepared }));
    }
  } catch (error) {
    await dispose(resources, settings.backend);
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_RESOURCE",
      `resource backend rejected the artifact${error instanceof Error ? `: ${error.message}` : ""}`,
      selected.releaseDirectory,
      error,
    );
  }

  deepFreeze(manifest);
  return Object.freeze({
    artifactRoot: selected.artifactRoot,
    releaseDirectory: selected.releaseDirectory,
    manifestPath,
    manifestSha256: manifestDigest,
    manifest,
    resources: Object.freeze(resources),
    functions: manifest.functions,
  });
}

function resolveOptions<P>(
  options: ArtifactLoadOptions<P>,
): ResolvedOptions<P> {
  const runtimeVersion = options.runtimeVersion ?? "0.0.0";
  if (!SEMVER.test(runtimeVersion))
    throw new TypeError("runtimeVersion must be valid semver");

  return {
    backend: options.backend ?? (defaultBackend as ArtifactResourceBackend<P>),
    runtimeVersion,
    capabilities: new Set(options.supportedCapabilities ?? []),
    opsets: new Set(options.supportedOnnxOpsets ?? [17, 18]),
    pointerBytes: positiveInteger(
      options.maximumPointerBytes ?? 1024 * 1024,
      "maximumPointerBytes",
    ),
    manifestBytes: positiveInteger(
      options.maximumManifestBytes ?? 8 * 1024 * 1024,
      "maximumManifestBytes",
    ),
    resourceBytes: positiveInteger(
      options.maximumResourceBytes ?? 1024 * 1024 * 1024,
      "maximumResourceBytes",
    ),
    aggregateBytes: positiveInteger(
      options.maximumAggregateResourceBytes ?? 2 * 1024 * 1024 * 1024,
      "maximumAggregateResourceBytes",
    ),
  };
}

async function selectRelease(
  artifactPath: string,
  pointerLimit: number,
): Promise<SelectedRelease> {
  if (artifactPath.length === 0)
    throw new TypeError("artifactPath must not be empty");
  const requested = resolve(artifactPath);
  const stats = await safeLstat(requested, "artifact directory");
  if (stats.isSymbolicLink() || !stats.isDirectory()) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      "artifact path must be a non-symlink directory",
      requested,
    );
  }

  const canonical = await realpath(requested);
  let hasPointer = false;
  try {
    await lstat(join(canonical, "current.json"));
    hasPointer = true;
  } catch (error) {
    if (!isEnoent(error))
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_PATH",
        "cannot inspect current.json",
        canonical,
        error,
      );
  }

  if (hasPointer) {
    const pointerPath = join(canonical, "current.json");
    const bytes = await readSafeFile(
      canonical,
      "current.json",
      pointerLimit,
      "deployment pointer",
    );
    const parsed = parseJson(bytes, pointerLimit, pointerPath);
    validatePointer(parsed);
    const pointer = parsed;
    const match = POINTER_RELEASE.exec(pointer.release);
    if (!match || match[1] !== pointer.manifestSha256) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_INVALID_POINTER",
        "pointer digests differ",
        pointerPath,
      );
    }

    return {
      artifactRoot: canonical,
      releaseDirectory: await safeDirectory(canonical, pointer.release),
      digest: pointer.manifestSha256,
    };
  }

  const match = RELEASE.exec(basename(canonical));
  if (!match?.[1]) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      "path is neither an artifact root nor sha256 release",
      canonical,
    );
  }

  const releases = dirname(canonical);
  return {
    artifactRoot:
      basename(releases) === "releases" ? dirname(releases) : releases,
    releaseDirectory: canonical,
    digest: match[1],
  };
}

function parseJson(bytes: Uint8Array, limit: number, path: string): unknown {
  try {
    return parseStrictJson(bytes, { maximumBytes: limit });
  } catch (error) {
    if (error instanceof StrictJsonError) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_INVALID_JSON",
        error.message,
        path,
        error,
      );
    }
    throw error;
  }
}

function validatePointer(value: unknown): asserts value is ArtifactPointerV1 {
  const pointer = object(value, "pointer", "SEMA_ARTIFACT_INVALID_POINTER");
  exact(
    pointer,
    ["kind", "pointerVersion", "release", "manifestSha256"],
    "pointer",
    "SEMA_ARTIFACT_INVALID_POINTER",
  );
  equal(
    get(pointer, "kind"),
    "semantscript.artifact-pointer",
    "pointer.kind",
    "SEMA_ARTIFACT_INVALID_POINTER",
  );
  equal(
    get(pointer, "pointerVersion"),
    1,
    "pointer.pointerVersion",
    "SEMA_ARTIFACT_INVALID_POINTER",
  );
  matches(
    get(pointer, "release"),
    POINTER_RELEASE,
    "pointer.release",
    "SEMA_ARTIFACT_INVALID_POINTER",
  );
  matches(
    get(pointer, "manifestSha256"),
    SHA256,
    "pointer.manifestSha256",
    "SEMA_ARTIFACT_INVALID_POINTER",
  );
}

function validateManifest(
  value: unknown,
): asserts value is ApplicationArtifactManifestV1 {
  const manifest = object(value, "manifest");
  exact(
    manifest,
    [
      "kind",
      "artifactVersion",
      "irVersion",
      "compatibility",
      "application",
      "build",
      "resources",
      "model",
      "functions",
    ],
    "manifest",
  );
  equal(
    get(manifest, "kind"),
    "semantscript.application-artifact",
    "manifest.kind",
  );
  equal(get(manifest, "artifactVersion"), 1, "manifest.artifactVersion");
  equal(get(manifest, "irVersion"), 1, "manifest.irVersion");

  const compatibility = object(
    get(manifest, "compatibility"),
    "manifest.compatibility",
  );
  exact(
    compatibility,
    [
      "runtimeAbiVersion",
      "modelAbiVersion",
      "canonicalInput",
      "minimumRuntimeVersion",
      "requiredCapabilities",
    ],
    "manifest.compatibility",
  );
  equal(
    get(compatibility, "runtimeAbiVersion"),
    1,
    "manifest.compatibility.runtimeAbiVersion",
  );
  equal(
    get(compatibility, "modelAbiVersion"),
    1,
    "manifest.compatibility.modelAbiVersion",
  );
  // The runtime serializes call inputs with the encoding the artifact was trained on.
  choice(
    get(compatibility, "canonicalInput"),
    ["semantscript.canonical-input/v1", "semantscript.canonical-input/v2"],
    "manifest.compatibility.canonicalInput",
  );
  matches(
    get(compatibility, "minimumRuntimeVersion"),
    SEMVER,
    "manifest.compatibility.minimumRuntimeVersion",
  );
  const capabilities = list(
    get(compatibility, "requiredCapabilities"),
    "manifest.compatibility.requiredCapabilities",
  );
  for (const [index, capability] of capabilities.entries())
    nonempty(
      capability,
      `manifest.compatibility.requiredCapabilities[${String(index)}]`,
    );
  uniquePrimitive(capabilities, "manifest.compatibility.requiredCapabilities");

  const application = object(
    get(manifest, "application"),
    "manifest.application",
  );
  exact(application, ["id", "version"], "manifest.application");
  matches(
    get(application, "id"),
    /^[a-z][a-z0-9._-]{1,127}$/u,
    "manifest.application.id",
  );
  matches(get(application, "version"), SEMVER, "manifest.application.version");

  const build = object(get(manifest, "build"), "manifest.build");
  exact(
    build,
    ["createdAt", "compilerVersion", "trainerVersion", "sourceIrSha256"],
    "manifest.build",
  );
  const createdAt = text(get(build, "createdAt"), "manifest.build.createdAt");
  if (!isRfc3339DateTime(createdAt))
    invalid("manifest.build.createdAt is not RFC 3339");
  matches(
    get(build, "compilerVersion"),
    SEMVER,
    "manifest.build.compilerVersion",
  );
  matches(
    get(build, "trainerVersion"),
    SEMVER,
    "manifest.build.trainerVersion",
  );
  matches(
    get(build, "sourceIrSha256"),
    SHA256,
    "manifest.build.sourceIrSha256",
  );

  const resources = list(get(manifest, "resources"), "manifest.resources", 4);
  for (const [index, resource] of resources.entries())
    validateResource(resource, `manifest.resources[${String(index)}]`);

  const model = object(get(manifest, "model"), "manifest.model");
  exact(model, ["tokenizerRef", "encoderRef", "adapterRef"], "manifest.model");
  logicalRef(get(model, "tokenizerRef"), "manifest.model.tokenizerRef");
  logicalRef(get(model, "encoderRef"), "manifest.model.encoderRef");
  logicalRef(get(model, "adapterRef"), "manifest.model.adapterRef");

  const functions = list(get(manifest, "functions"), "manifest.functions", 1);
  for (const [index, fn] of functions.entries())
    validateFunction(fn, `manifest.functions[${String(index)}]`);
}

function validateResource(value: unknown, path: string): void {
  const resource = object(value, path);
  const role = choice(
    get(resource, "role"),
    ["tokenizer", "encoder", "adapter", "head"],
    `${path}.role`,
  );
  const common = [
    "ref",
    "role",
    "format",
    "formatVersion",
    "path",
    "byteLength",
    "sha256",
  ];
  if (role === "tokenizer") {
    exact(resource, [...common, "maximumSequenceLength"], path);
    equal(get(resource, "format"), "tokenizer-json", `${path}.format`);
    const maximumSequenceLength = integer(
      get(resource, "maximumSequenceLength"),
      `${path}.maximumSequenceLength`,
      1,
    );
    if (maximumSequenceLength > 8192)
      invalid(`${path}.maximumSequenceLength must be <= 8192`);
  } else {
    exact(resource, [...common, "onnx"], path);
    equal(get(resource, "format"), "onnx", `${path}.format`);
    validateOnnx(get(resource, "onnx"), `${path}.onnx`);
  }
  logicalRef(get(resource, "ref"), `${path}.ref`);
  equal(get(resource, "formatVersion"), 1, `${path}.formatVersion`);
  const resourcePath = matches(
    get(resource, "path"),
    PORTABLE_PATH,
    `${path}.path`,
  );
  if (resourcePath.length > 500) invalid(`${path}.path is too long`);
  integer(get(resource, "byteLength"), `${path}.byteLength`, 1);
  matches(get(resource, "sha256"), SHA256, `${path}.sha256`);
}

function validateOnnx(value: unknown, path: string): void {
  const onnx = object(value, path);
  exactWithOptional(
    onnx,
    ["opset", "inputs", "outputs", "externalData"],
    ["precision", "quantization"],
    path,
  );
  integer(get(onnx, "opset"), `${path}.opset`, 1);
  equal(get(onnx, "externalData"), false, `${path}.externalData`);
  validateOnnxPrecision(onnx, path);
  const inputs = list(get(onnx, "inputs"), `${path}.inputs`, 1);
  const outputs = list(get(onnx, "outputs"), `${path}.outputs`, 1);
  for (const [index, tensor] of inputs.entries())
    validateTensor(tensor, `${path}.inputs[${String(index)}]`);
  for (const [index, tensor] of outputs.entries())
    validateTensor(tensor, `${path}.outputs[${String(index)}]`);
}

function validateOnnxPrecision(onnx: JsonRecord, path: string): void {
  // A graph published before quantization existed carries no precision and is float32.
  const precision =
    "precision" in onnx
      ? choice(
          get(onnx, "precision"),
          ["float32", "int8-dynamic"],
          `${path}.precision`,
        )
      : "float32";
  const quantized = precision !== "float32";
  if (!("quantization" in onnx)) {
    if (quantized)
      invalid(`${path}.quantization is required for precision ${precision}`);
    return;
  }
  if (!quantized)
    invalid(`${path}.quantization is only allowed for a quantized precision`);
  const quantization = object(
    get(onnx, "quantization"),
    `${path}.quantization`,
  );
  exact(
    quantization,
    [
      "method",
      "weightType",
      "perChannel",
      "reduceRange",
      "argmaxDisagreementTolerance",
      "attestedDisagreementTolerance",
      "eceThreshold",
      "sourceManifestSha256",
    ],
    `${path}.quantization`,
  );
  equal(get(quantization, "method"), "dynamic", `${path}.quantization.method`);
  choice(
    get(quantization, "weightType"),
    ["int8", "uint8"],
    `${path}.quantization.weightType`,
  );
  bool(get(quantization, "perChannel"), `${path}.quantization.perChannel`);
  bool(get(quantization, "reduceRange"), `${path}.quantization.reduceRange`);
  unit(
    get(quantization, "argmaxDisagreementTolerance"),
    `${path}.quantization.argmaxDisagreementTolerance`,
  );
  integer(
    get(quantization, "attestedDisagreementTolerance"),
    `${path}.quantization.attestedDisagreementTolerance`,
    0,
  );
  unit(get(quantization, "eceThreshold"), `${path}.quantization.eceThreshold`);
  matches(
    get(quantization, "sourceManifestSha256"),
    SHA256,
    `${path}.quantization.sourceManifestSha256`,
  );
}

function validateTensor(value: unknown, path: string): void {
  const tensor = object(value, path);
  exact(tensor, ["name", "dtype", "shape"], path);
  nonempty(get(tensor, "name"), `${path}.name`);
  choice(
    get(tensor, "dtype"),
    ["int64", "float16", "float32"],
    `${path}.dtype`,
  );
  const shape = list(get(tensor, "shape"), `${path}.shape`, 1);
  for (const [index, dimension] of shape.entries()) {
    if (typeof dimension === "number")
      integer(dimension, `${path}.shape[${String(index)}]`, 1);
    else
      matches(dimension, SYMBOLIC_DIMENSION, `${path}.shape[${String(index)}]`);
  }
}

function validateFunction(value: unknown, path: string): void {
  const fn = object(value, path);
  exact(
    fn,
    [
      "id",
      "semanticSha256",
      "inputs",
      "inputSchemaSha256",
      "outputSchemaSha256",
      "adapterRef",
      "heads",
      "runtime",
      "verification",
      "trainingProvenance",
    ],
    path,
    "SEMA_ARTIFACT_INVALID_MANIFEST",
    ["encoderRef"],
  );
  matches(get(fn, "id"), FUNCTION_ID, `${path}.id`);
  for (const key of [
    "semanticSha256",
    "inputSchemaSha256",
    "outputSchemaSha256",
  ])
    matches(get(fn, key), SHA256, `${path}.${key}`);
  logicalRef(get(fn, "adapterRef"), `${path}.adapterRef`);
  if (Object.prototype.hasOwnProperty.call(fn, "encoderRef"))
    logicalRef(get(fn, "encoderRef"), `${path}.encoderRef`);
  const inputs = list(get(fn, "inputs"), `${path}.inputs`);
  const inputBudget = { nodes: 0 };
  for (const [index, input] of inputs.entries())
    validateInput(input, `${path}.inputs[${String(index)}]`, inputBudget);
  const heads = list(get(fn, "heads"), `${path}.heads`, 1);
  for (const [index, head] of heads.entries())
    validateHead(head, `${path}.heads[${String(index)}]`);
  validatePolicy(get(fn, "runtime"), `${path}.runtime`);
  validateVerification(get(fn, "verification"), `${path}.verification`);
  const provenance = object(
    get(fn, "trainingProvenance"),
    `${path}.trainingProvenance`,
  );
  exact(
    provenance,
    ["datasetSha256", "trainingKeySha256", "teacher", "baseModel"],
    `${path}.trainingProvenance`,
  );
  matches(
    get(provenance, "datasetSha256"),
    SHA256,
    `${path}.trainingProvenance.datasetSha256`,
  );
  matches(
    get(provenance, "trainingKeySha256"),
    SHA256,
    `${path}.trainingProvenance.trainingKeySha256`,
  );
  nonempty(get(provenance, "teacher"), `${path}.trainingProvenance.teacher`);
  nonempty(
    get(provenance, "baseModel"),
    `${path}.trainingProvenance.baseModel`,
  );
}

function validateInput(
  value: unknown,
  path: string,
  budget: { nodes: number },
): void {
  const input = object(value, path);
  exact(input, ["name", "index", "tsType", "type"], path);
  matches(get(input, "name"), IDENTIFIER, `${path}.name`);
  integer(get(input, "index"), `${path}.index`, 0);
  nonempty(get(input, "tsType"), `${path}.tsType`);
  validateInputType(get(input, "type"), `${path}.type`, budget, 0);
}

function validateInputType(
  value: unknown,
  path: string,
  budget: { nodes: number },
  depth: number,
): void {
  budget.nodes += 1;
  if (depth > 64 || budget.nodes > 100_000)
    invalid(`${path} exceeds input type complexity limits`);
  const type = object(value, path);
  const kind = text(get(type, "kind"), `${path}.kind`);
  if (["string", "boolean", "number", "null"].includes(kind)) {
    exact(type, ["kind"], path);
  } else if (kind === "literal") {
    exact(type, ["kind", "value"], path);
    scalar(get(type, "value"), `${path}.value`);
  } else if (kind === "enum") {
    exact(type, ["kind", "name", "base", "values"], path);
    nonempty(get(type, "name"), `${path}.name`);
    const base = choice(
      get(type, "base"),
      ["string", "number"],
      `${path}.base`,
    );
    const values = list(get(type, "values"), `${path}.values`, 1);
    for (const [index, entry] of values.entries()) {
      if (base === "string") text(entry, `${path}.values[${String(index)}]`);
      else number(entry, `${path}.values[${String(index)}]`);
    }
    uniqueSemantic(values, `${path}.values`);
  } else if (kind === "array") {
    exact(type, ["kind", "items"], path);
    validateInputType(get(type, "items"), `${path}.items`, budget, depth + 1);
  } else if (kind === "tuple") {
    exact(type, ["kind", "items"], path);
    for (const [index, entry] of list(
      get(type, "items"),
      `${path}.items`,
    ).entries())
      validateInputType(
        entry,
        `${path}.items[${String(index)}]`,
        budget,
        depth + 1,
      );
  } else if (kind === "object") {
    exact(type, ["kind", "name", "fields"], path);
    nonempty(get(type, "name"), `${path}.name`);
    const names = new Set<string>();
    for (const [index, entry] of list(
      get(type, "fields"),
      `${path}.fields`,
    ).entries()) {
      const fieldPath = `${path}.fields[${String(index)}]`;
      const field = object(entry, fieldPath);
      exact(field, ["name", "optional", "type"], fieldPath);
      const name = text(get(field, "name"), `${fieldPath}.name`);
      bool(get(field, "optional"), `${fieldPath}.optional`);
      validateInputType(
        get(field, "type"),
        `${fieldPath}.type`,
        budget,
        depth + 1,
      );
      addUnique(names, name, `${path}.fields`, "field name");
    }
  } else if (kind === "union") {
    exact(type, ["kind", "variants"], path);
    const variants = list(get(type, "variants"), `${path}.variants`, 2);
    for (const [index, variant] of variants.entries())
      validateInputType(
        variant,
        `${path}.variants[${String(index)}]`,
        budget,
        depth + 1,
      );
    strictSemanticOrder(variants, `${path}.variants`);
  } else {
    invalid(`${path}.kind is unsupported`);
  }
}

function validateHead(value: unknown, path: string): void {
  const head = object(value, path);
  exact(
    head,
    [
      "outputPath",
      "headRef",
      "type",
      "parameterization",
      "calibration",
      "verification",
    ],
    path,
  );
  const outputPath = list(get(head, "outputPath"), `${path}.outputPath`, 0, 1);
  if (outputPath.length === 1) text(outputPath[0], `${path}.outputPath[0]`);
  logicalRef(get(head, "headRef"), `${path}.headRef`);
  const kind = validateHeadType(get(head, "type"), `${path}.type`);
  const parameterization = choice(
    get(head, "parameterization"),
    ["binary-sigmoid", "categorical-softmax"],
    `${path}.parameterization`,
  );
  if ((kind === "boolean") !== (parameterization === "binary-sigmoid"))
    invalid(`${path}.parameterization does not match head type`);
  validateCalibration(get(head, "calibration"), `${path}.calibration`);
  const verification = object(
    get(head, "verification"),
    `${path}.verification`,
  );
  exact(verification, ["accuracy", "pairConsistency"], `${path}.verification`);
  unit(get(verification, "accuracy"), `${path}.verification.accuracy`);
  unit(
    get(verification, "pairConsistency"),
    `${path}.verification.pairConsistency`,
  );
}

function validateHeadType(value: unknown, path: string): string {
  const type = object(value, path);
  const kind = choice(
    get(type, "kind"),
    [
      "boolean",
      "nominal-string",
      "ordinal-string",
      "nominal-number",
      "ordinal-number",
    ],
    `${path}.kind`,
  );
  if (kind === "boolean") {
    exact(type, ["kind", "support"], path);
    const support = list(get(type, "support"), `${path}.support`, 2, 2);
    equal(support[0], false, `${path}.support[0]`);
    equal(support[1], true, `${path}.support[1]`);
  } else if (kind === "nominal-string" || kind === "ordinal-string") {
    exact(type, ["kind", "support"], path);
    const support = list(get(type, "support"), `${path}.support`, 2);
    for (const [index, entry] of support.entries())
      text(entry, `${path}.support[${String(index)}]`);
    uniqueSemantic(support, `${path}.support`);
  } else if (kind === "nominal-number") {
    exact(type, ["kind", "support"], path);
    const support = list(get(type, "support"), `${path}.support`, 2);
    for (const [index, entry] of support.entries())
      number(entry, `${path}.support[${String(index)}]`);
    uniqueSemantic(support, `${path}.support`);
  } else {
    exact(
      type,
      ["kind", "sourceKind", "minimum", "maximum", "step", "supportDecimal"],
      path,
    );
    const sourceKind = choice(
      get(type, "sourceKind"),
      ["bounded-int", "bounded-number"],
      `${path}.sourceKind`,
    );
    const minimum = canonicalDecimal(get(type, "minimum"), `${path}.minimum`);
    const maximum = canonicalDecimal(get(type, "maximum"), `${path}.maximum`);
    const step = canonicalDecimal(get(type, "step"), `${path}.step`);
    const support = list(
      get(type, "supportDecimal"),
      `${path}.supportDecimal`,
      2,
    );
    const supportDecimal = support.map((entry, index) =>
      canonicalDecimal(entry, `${path}.supportDecimal[${String(index)}]`),
    );
    validateOrdinalSupport(
      sourceKind,
      minimum,
      maximum,
      step,
      supportDecimal,
      path,
    );
  }
  return kind;
}

function canonicalDecimal(value: unknown, path: string): string {
  const result = matches(value, DECIMAL, path);
  if (result.length > 128) invalid(`${path} is too long`);
  return result;
}

function validateOrdinalSupport(
  sourceKind: "bounded-int" | "bounded-number",
  minimumText: string,
  maximumText: string,
  stepText: string,
  supportText: readonly string[],
  path: string,
): void {
  const parsed = [minimumText, maximumText, stepText, ...supportText].map(
    parseDecimal,
  );
  const scale = Math.max(...parsed.map((value) => value.scale));
  const scaled = parsed.map(
    (value) => value.integer * 10n ** BigInt(scale - value.scale),
  );
  const minimum = scaled[0];
  const maximum = scaled[1];
  const step = scaled[2];
  const support = scaled.slice(3);
  if (
    minimum === undefined ||
    maximum === undefined ||
    step === undefined ||
    step <= 0n
  ) {
    invalid(`${path} requires a positive ordinal step`);
  }
  if (sourceKind === "bounded-int") {
    if (
      stepText !== "1" ||
      [minimumText, maximumText, ...supportText].some((value) =>
        value.includes("."),
      )
    ) {
      invalid(
        `${path} bounded-int support requires integral values and step 1`,
      );
    }
  }
  if (minimum >= maximum) invalid(`${path} minimum must be less than maximum`);
  for (let index = 0; index < support.length; index += 1) {
    if (support[index] !== minimum + step * BigInt(index)) {
      invalid(
        `${path}.supportDecimal is not the exact ascending min/max/step sequence`,
      );
    }
  }
  if (support.at(-1) !== maximum) {
    invalid(`${path}.supportDecimal must end at maximum`);
  }

  const binary64 = new Set<string>();
  for (const [index, value] of supportText.entries()) {
    const converted = Number(value);
    if (!Number.isFinite(converted)) {
      invalid(
        `${path}.supportDecimal[${String(index)}] does not convert to finite binary64`,
      );
    }
    const key = binary64Key(converted);
    if (binary64.has(key)) {
      invalid(
        `${path}.supportDecimal values collide after binary64 conversion`,
      );
    }
    binary64.add(key);
  }
}

function parseDecimal(value: string): {
  readonly integer: bigint;
  readonly scale: number;
} {
  const negative = value.startsWith("-");
  const unsigned = negative ? value.slice(1) : value;
  const [whole = "0", fraction = ""] = unsigned.split(".");
  const integer = BigInt(`${negative ? "-" : ""}${whole}${fraction}`);
  return { integer, scale: fraction.length };
}

function binary64Key(value: number): string {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return hex(bytes);
}

function validateCalibration(value: unknown, path: string): void {
  const calibration = object(value, path);
  exact(
    calibration,
    [
      "method",
      "temperature",
      "ece",
      "brier",
      "sampleCount",
      "splitSha256",
      "eceBins",
    ],
    path,
  );
  equal(get(calibration, "method"), "temperature-scaling", `${path}.method`);
  if (number(get(calibration, "temperature"), `${path}.temperature`) <= 0)
    invalid(`${path}.temperature must be positive`);
  unit(get(calibration, "ece"), `${path}.ece`);
  unit(get(calibration, "brier"), `${path}.brier`);
  integer(get(calibration, "sampleCount"), `${path}.sampleCount`, 1);
  matches(get(calibration, "splitSha256"), SHA256, `${path}.splitSha256`);
  integer(get(calibration, "eceBins"), `${path}.eceBins`, 2);
}

function validatePolicy(value: unknown, path: string): void {
  const policy = object(value, path);
  exact(
    policy,
    ["resultMode", "confidenceThreshold", "policy", "fallbackRef"],
    path,
  );
  choice(
    get(policy, "resultMode"),
    ["value", "diagnostic"],
    `${path}.resultMode`,
  );
  const threshold = get(policy, "confidenceThreshold");
  if (threshold !== null) unit(threshold, `${path}.confidenceThreshold`);
  choice(
    get(policy, "policy"),
    ["none", "scalar-top1", "all-fields"],
    `${path}.policy`,
  );
  const fallback = get(policy, "fallbackRef");
  if (fallback !== null) nonempty(fallback, `${path}.fallbackRef`);
}

function validateVerification(value: unknown, path: string): void {
  const verification = object(value, path);
  exact(
    verification,
    [
      "status",
      "accuracy",
      "ece",
      "brier",
      "pairConsistency",
      "attestedCases",
      "exampleFailures",
      "constraintViolations",
      "typeErrors",
    ],
    path,
  );
  equal(get(verification, "status"), "passed", `${path}.status`);
  for (const key of ["accuracy", "ece", "brier", "pairConsistency"])
    unit(get(verification, key), `${path}.${key}`);
  integer(get(verification, "attestedCases"), `${path}.attestedCases`, 1);
  for (const key of ["exampleFailures", "typeErrors"])
    equal(get(verification, key), 0, `${path}.${key}`);
  // Raw-model constraint violations are recorded, not forbidden: the trainer's release
  // gate decided whether the observed rate was tolerated, and status must be "passed".
  integer(
    get(verification, "constraintViolations"),
    `${path}.constraintViolations`,
    0,
  );
}

function validateCompatibility(
  manifest: ApplicationArtifactManifestV1,
  settings: ResolvedOptions<unknown>,
): void {
  if (
    compareSemver(
      settings.runtimeVersion,
      manifest.compatibility.minimumRuntimeVersion,
    ) < 0
  ) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_INCOMPATIBLE",
      `artifact requires runtime ${manifest.compatibility.minimumRuntimeVersion}; current is ${settings.runtimeVersion}`,
    );
  }
  for (const capability of manifest.compatibility.requiredCapabilities) {
    if (!settings.capabilities.has(capability)) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_INCOMPATIBLE",
        `unsupported capability ${capability}`,
      );
    }
  }
}

function validateRelationships(
  manifest: ApplicationArtifactManifestV1,
  opsets: ReadonlySet<number>,
): void {
  const resources = new Map<string, ArtifactResourceV1>();
  const paths = new Set<string>();
  for (const resource of manifest.resources) {
    if (resources.has(resource.ref))
      invalid(`duplicate resource ref ${resource.ref}`);
    resources.set(resource.ref, resource);
    addUnique(paths, resource.path, "manifest.resources", "path");
    if (resource.role !== "tokenizer") {
      if (!opsets.has(resource.onnx.opset)) {
        throw new ArtifactLoadError(
          "SEMA_ARTIFACT_INCOMPATIBLE",
          `unsupported ONNX opset ${String(resource.onnx.opset)}`,
          resource.path,
        );
      }
      uniqueTensorNames(resource);
      validateAbi(resource);
    }
  }

  requireRole(
    resources,
    manifest.model.tokenizerRef,
    "tokenizer",
    "manifest.model.tokenizerRef",
  );
  const encoder = requireOnnxRole(
    resources,
    manifest.model.encoderRef,
    "encoder",
    "manifest.model.encoderRef",
  );
  requireRole(
    resources,
    manifest.model.adapterRef,
    "adapter",
    "manifest.model.adapterRef",
  );
  const referenced = new Set([
    manifest.model.tokenizerRef,
    manifest.model.encoderRef,
    manifest.model.adapterRef,
  ]);
  const functionIds = new Set<string>();
  for (const [index, fn] of manifest.functions.entries()) {
    addUnique(functionIds, fn.id, "manifest.functions", "function id");
    validateFunctionRelationships(fn, index, resources, encoder, referenced);
  }
  for (const resource of manifest.resources) {
    if (!referenced.has(resource.ref)) {
      invalid(
        `resource ${resource.ref} is not selected by the model or any function`,
      );
    }
  }
}

function validateFunctionRelationships(
  fn: ArtifactFunctionV1,
  index: number,
  resources: ReadonlyMap<string, ArtifactResourceV1>,
  encoder: OnnxResourceV1,
  referenced: Set<string>,
): void {
  const path = `manifest.functions[${String(index)}]`;
  const names = new Set<string>();
  for (const [inputIndex, input] of fn.inputs.entries()) {
    if (input.index !== inputIndex)
      invalid(
        `${path}.inputs[${String(inputIndex)}].index must equal its position`,
      );
    addUnique(names, input.name, `${path}.inputs`, "input name");
  }
  if (digest(semanticBytes(fn.inputs)) !== fn.inputSchemaSha256) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_INTEGRITY",
      `${path}.inputSchemaSha256 does not match inputs`,
    );
  }

  const adapter = requireOnnxRole(
    resources,
    fn.adapterRef,
    "adapter",
    `${path}.adapterRef`,
  );
  referenced.add(adapter.ref);
  // A depth-routed function reads a prefix of the shared encoder exported on its own.
  const functionEncoder =
    fn.encoderRef === undefined
      ? encoder
      : requireOnnxRole(
          resources,
          fn.encoderRef,
          "encoder",
          `${path}.encoderRef`,
        );
  referenced.add(functionEncoder.ref);
  adjacent(
    functionEncoder.onnx.outputs[0],
    adapter.onnx.inputs[0],
    functionEncoder.ref,
    adapter.ref,
  );
  const outputPaths = new Set<string>();
  const scalarOutput =
    fn.heads.length === 1 && fn.heads[0]?.outputPath.length === 0;
  for (const [headIndex, binding] of fn.heads.entries()) {
    if (!scalarOutput && binding.outputPath.length !== 1)
      invalid(`${path}.heads must describe scalar or flat output`);
    addUnique(
      outputPaths,
      JSON.stringify(binding.outputPath),
      `${path}.heads`,
      "output path",
    );
    const head = requireOnnxRole(
      resources,
      binding.headRef,
      "head",
      `${path}.heads[${String(headIndex)}].headRef`,
    );
    referenced.add(head.ref);
    adjacent(
      adapter.onnx.outputs[0],
      head.onnx.inputs[0],
      adapter.ref,
      head.ref,
    );
    validateLogits(binding, head, `${path}.heads[${String(headIndex)}]`);
  }

  validateRuntimePolicyRelationships(fn, scalarOutput, path);
}

function validateRuntimePolicyRelationships(
  fn: ArtifactFunctionV1,
  scalarOutput: boolean,
  path: string,
): void {
  const runtimePath = `${path}.runtime`;
  const { confidenceThreshold, fallbackRef, policy, resultMode } = fn.runtime;

  if (confidenceThreshold === null) {
    if (policy !== "none") {
      invalid(
        `${runtimePath}.policy must be none when confidenceThreshold is null`,
      );
    }
    if (fallbackRef !== null) {
      invalid(
        `${runtimePath}.fallbackRef must be null when confidenceThreshold is null`,
      );
    }
    return;
  }

  const expectedPolicy = scalarOutput ? "scalar-top1" : "all-fields";
  if (policy !== expectedPolicy) {
    invalid(
      `${runtimePath}.policy must be ${expectedPolicy} for a thresholded ${scalarOutput ? "scalar" : "flat"} output`,
    );
  }

  if (resultMode === "diagnostic" && fallbackRef !== null) {
    invalid(`${runtimePath}.fallbackRef must be null for diagnostic results`);
  }
}

function uniqueTensorNames(resource: OnnxResourceV1): void {
  const inputs = new Set<string>();
  const outputs = new Set<string>();
  for (const tensor of resource.onnx.inputs)
    addUnique(inputs, tensor.name, resource.ref, "input tensor");
  for (const tensor of resource.onnx.outputs)
    addUnique(outputs, tensor.name, resource.ref, "output tensor");
}

function validateAbi(resource: OnnxResourceV1): void {
  const { inputs, outputs } = resource.onnx;
  if (resource.role === "encoder") {
    if (inputs.length !== 2 || outputs.length !== 1)
      invalid(`${resource.ref} must have two inputs and one output`);
    requireTensor(
      findTensor(inputs, "input_ids", resource.ref),
      "int64",
      ["BATCH", "SEQUENCE"],
      resource.ref,
    );
    requireTensor(
      findTensor(inputs, "attention_mask", resource.ref),
      "int64",
      ["BATCH", "SEQUENCE"],
      resource.ref,
    );
    embedding(
      findTensor(outputs, "sentence_embedding", resource.ref),
      resource.ref,
    );
  } else if (resource.role === "adapter") {
    if (inputs.length !== 1 || outputs.length !== 1)
      invalid(`${resource.ref} must have one input and one output`);
    embedding(
      findTensor(inputs, "sentence_embedding", resource.ref),
      resource.ref,
    );
    embedding(
      findTensor(outputs, "function_embedding", resource.ref),
      resource.ref,
    );
  } else {
    if (inputs.length !== 1 || outputs.length !== 1)
      invalid(`${resource.ref} must have one input and one output`);
    embedding(
      findTensor(inputs, "function_embedding", resource.ref),
      resource.ref,
    );
    const logits = findTensor(outputs, "logits", resource.ref);
    if (
      logits.dtype !== "float32" ||
      logits.shape.length !== 2 ||
      !batch(logits.shape[0])
    ) {
      invalid(`${resource.ref}.logits must be float32 [BATCH,K]`);
    }
    integer(logits.shape[1], `${resource.ref}.logits.shape[1]`, 1);
  }
}

function validateLogits(
  binding: HeadBindingV1,
  resource: OnnxResourceV1,
  path: string,
): void {
  const logits = resource.onnx.outputs[0];
  if (!logits) invalid(`${resource.ref} has no logits output`);
  const count =
    binding.type.kind === "boolean"
      ? 1
      : binding.type.kind === "ordinal-number"
        ? binding.type.supportDecimal.length
        : binding.type.support.length;
  if (logits.shape[1] !== count)
    invalid(`${path} support requires ${String(count)} logits`);
}

function requireRole(
  resources: ReadonlyMap<string, ArtifactResourceV1>,
  ref: string,
  role: ArtifactResourceV1["role"],
  path: string,
): ArtifactResourceV1 {
  const resource = resources.get(ref);
  if (!resource) invalid(`${path} references unknown resource ${ref}`);
  if (resource.role !== role)
    invalid(`${path} expected ${role}, found ${resource.role}`);
  return resource;
}

function requireOnnxRole(
  resources: ReadonlyMap<string, ArtifactResourceV1>,
  ref: string,
  role: "encoder" | "adapter" | "head",
  path: string,
): OnnxResourceV1 {
  return requireRole(resources, ref, role, path) as OnnxResourceV1;
}

function findTensor(
  tensors: readonly TensorDescriptorV1[],
  name: string,
  ref: string,
): TensorDescriptorV1 {
  const tensor = tensors.find((candidate) => candidate.name === name);
  if (!tensor) invalid(`${ref} is missing tensor ${name}`);
  return tensor;
}

function embedding(tensor: TensorDescriptorV1, ref: string): void {
  const hidden = tensor.shape[1];
  if (
    tensor.dtype !== "float32" ||
    tensor.shape.length !== 2 ||
    !batch(tensor.shape[0]) ||
    (hidden !== "HIDDEN" &&
      (typeof hidden !== "number" ||
        !Number.isSafeInteger(hidden) ||
        hidden < 1))
  ) {
    invalid(`${ref}.${tensor.name} must be float32 [BATCH,HIDDEN]`);
  }
}

function requireTensor(
  tensor: TensorDescriptorV1,
  dtype: TensorDescriptorV1["dtype"],
  shape: readonly (number | string)[],
  path: string,
): void {
  if (tensor.dtype !== dtype || !sameShape(tensor.shape, shape))
    invalid(`${path}.${tensor.name} has invalid ABI`);
}

function adjacent(
  upstream: TensorDescriptorV1 | undefined,
  downstream: TensorDescriptorV1 | undefined,
  upstreamRef: string,
  downstreamRef: string,
): void {
  if (
    !upstream ||
    !downstream ||
    upstream.dtype !== downstream.dtype ||
    !compatibleEdgeShape(upstream.shape, downstream.shape)
  ) {
    invalid(`incompatible tensor edge ${upstreamRef} -> ${downstreamRef}`);
  }
}

async function safeDirectory(
  root: string,
  relativePath: string,
): Promise<string> {
  portablePath(relativePath, "release directory");
  let current = root;
  for (const segment of relativePath.split("/")) {
    current = join(current, segment);
    const stats = await safeLstat(current, "release directory");
    if (stats.isSymbolicLink() || !stats.isDirectory()) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_PATH",
        "release contains a symlink or non-directory",
        current,
      );
    }
  }
  const canonical = await realpath(current);
  contained(root, canonical, "release directory");
  return canonical;
}

async function readSafeFile(
  root: string,
  relativePath: string,
  maximumBytes: number,
  description: string,
): Promise<Uint8Array> {
  portablePath(relativePath, description);
  const segments = relativePath.split("/");
  const filename = segments.pop();
  if (!filename)
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      `${description} has no filename`,
      relativePath,
    );
  let parent = root;
  for (const segment of segments) {
    parent = join(parent, segment);
    const stats = await safeLstat(parent, description);
    if (stats.isSymbolicLink() || !stats.isDirectory()) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_PATH",
        `${description} has an unsafe parent`,
        parent,
      );
    }
  }

  const target = join(parent, filename);
  contained(root, target, description);
  const pathStats = await safeLstat(target, description);
  if (pathStats.isSymbolicLink() || !pathStats.isFile()) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      `${description} is not a non-symlink regular file`,
      target,
    );
  }

  const handle = await open(
    target,
    constants.O_RDONLY | constants.O_NOFOLLOW,
  ).catch((error: unknown) => {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      `cannot safely open ${description}`,
      target,
      error,
    );
  });
  try {
    const before = await handle.stat();
    if (!before.isFile())
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_PATH",
        `${description} is not regular`,
        target,
      );
    if (!Number.isSafeInteger(before.size) || before.size > maximumBytes) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_QUOTA",
        `${description} exceeds quota`,
        target,
      );
    }
    contained(root, await realpath(target), description);
    const buffer = await handle.readFile();
    const after = await handle.stat();
    if (
      before.dev !== after.dev ||
      before.ino !== after.ino ||
      before.size !== after.size ||
      before.mtimeMs !== after.mtimeMs ||
      buffer.byteLength !== before.size
    ) {
      throw new ArtifactLoadError(
        "SEMA_ARTIFACT_INTEGRITY",
        `${description} changed while loading`,
        target,
      );
    }
    return standaloneFileBytes(buffer);
  } finally {
    await handle.close();
  }
}

function standaloneFileBytes(buffer: Uint8Array): Uint8Array {
  const backing = buffer.buffer;
  if (
    backing instanceof ArrayBuffer &&
    buffer.byteOffset === 0 &&
    buffer.byteLength === backing.byteLength
  ) {
    return new Uint8Array(backing);
  }

  return new Uint8Array(backing, buffer.byteOffset, buffer.byteLength).slice();
}

async function safeLstat(path: string, description: string) {
  try {
    return await lstat(path);
  } catch (error) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      `cannot inspect ${description}`,
      path,
      error,
    );
  }
}

async function dispose<P>(
  resources: readonly StagedArtifactResource<P>[],
  backend: ArtifactResourceBackend<P>,
): Promise<void> {
  if (!backend.disposeResource) return;
  for (let index = resources.length - 1; index >= 0; index -= 1) {
    const resource = resources[index];
    if (!resource) continue;
    try {
      await backend.disposeResource(resource.prepared);
    } catch {
      // Preserve the preparation error; rollback disposal is best effort.
    }
  }
}

function portablePath(path: string, description: string): void {
  if (
    path.length === 0 ||
    path.length > 500 ||
    isAbsolute(path) ||
    !PORTABLE_PATH.test(path) ||
    path.includes("\\")
  ) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      `${description} is not a portable relative path`,
      path,
    );
  }
}

function contained(root: string, target: string, description: string): void {
  const child = relative(root, target);
  if (child === ".." || child.startsWith(`..${sep}`) || isAbsolute(child)) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_PATH",
      `${description} resolves outside its root`,
      target,
    );
  }
}

function isEnoent(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function object(
  value: unknown,
  path: string,
  code: ArtifactLoadErrorCode = "SEMA_ARTIFACT_INVALID_MANIFEST",
): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new ArtifactLoadError(code, `${path} must be an object`);
  }
  return value as JsonRecord;
}

function get(value: JsonRecord, key: string): unknown {
  return value[key];
}

function exactWithOptional(
  value: JsonRecord,
  required: readonly string[],
  optional: readonly string[],
  path: string,
  code: ArtifactLoadErrorCode = "SEMA_ARTIFACT_INVALID_MANIFEST",
): void {
  const allowed = new Set([...required, ...optional]);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key))
      throw new ArtifactLoadError(
        code,
        `${path} has unsupported property ${key}`,
      );
  }
  for (const key of required) {
    if (!(key in value))
      throw new ArtifactLoadError(code, `${path} is missing ${key}`);
  }
}

function exact(
  value: JsonRecord,
  keys: readonly string[],
  path: string,
  code: ArtifactLoadErrorCode = "SEMA_ARTIFACT_INVALID_MANIFEST",
  optional: readonly string[] = [],
): void {
  const expected = new Set([...keys, ...optional]);
  for (const key of Object.keys(value)) {
    if (!expected.has(key))
      throw new ArtifactLoadError(
        code,
        `${path} has unsupported property ${key}`,
      );
  }
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(value, key))
      throw new ArtifactLoadError(code, `${path} is missing ${key}`);
  }
}

function list(
  value: unknown,
  path: string,
  minimum = 0,
  maximum = Number.MAX_SAFE_INTEGER,
): unknown[] {
  if (
    !Array.isArray(value) ||
    value.length < minimum ||
    value.length > maximum
  ) {
    invalid(
      `${path} must be an array with ${String(minimum)} to ${String(maximum)} items`,
    );
  }
  return value;
}

function text(value: unknown, path: string): string {
  if (typeof value !== "string") invalid(`${path} must be a string`);
  return value;
}

function nonempty(value: unknown, path: string): string {
  const result = text(value, path);
  if (result.length === 0) invalid(`${path} must not be empty`);
  return result;
}

function matches(
  value: unknown,
  pattern: RegExp,
  path: string,
  code: ArtifactLoadErrorCode = "SEMA_ARTIFACT_INVALID_MANIFEST",
): string {
  if (typeof value !== "string" || !pattern.test(value))
    throw new ArtifactLoadError(code, `${path} is invalid`);
  return value;
}

function isRfc3339DateTime(value: string): boolean {
  const match =
    /^(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?([Zz]|[+-]\d{2}:\d{2})$/u.exec(
      value,
    );
  if (match === null) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const hour = Number(match[4]);
  const minute = Number(match[5]);
  const second = Number(match[6]);
  if (month < 1 || month > 12 || hour > 23 || minute > 59 || second > 59)
    return false;
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (day < 1 || day > (days[month - 1] ?? 0)) return false;
  const zone = match[7] ?? "";
  if (zone.toUpperCase() !== "Z") {
    const zoneHour = Number(zone.slice(1, 3));
    const zoneMinute = Number(zone.slice(4, 6));
    if (zoneHour > 23 || zoneMinute > 59) return false;
  }
  return true;
}

function logicalRef(value: unknown, path: string): string {
  const result = matches(value, LOGICAL_REF, path);
  if (result.length > 200) invalid(`${path} is too long`);
  return result;
}

function choice<const T extends string>(
  value: unknown,
  values: readonly T[],
  path: string,
): T {
  if (typeof value !== "string" || !values.includes(value as T))
    invalid(`${path} has an unsupported value`);
  return value as T;
}

function bool(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") invalid(`${path} must be boolean`);
  return value;
}

function number(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value))
    invalid(`${path} must be finite number`);
  return value;
}

function integer(value: unknown, path: string, minimum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum)
    invalid(`${path} must be a safe integer >= ${String(minimum)}`);
  return value as number;
}

function unit(value: unknown, path: string): number {
  const result = number(value, path);
  if (result < 0 || result > 1) invalid(`${path} must be in [0,1]`);
  return result;
}

function scalar(value: unknown, path: string): void {
  if (
    value !== null &&
    typeof value !== "string" &&
    typeof value !== "boolean" &&
    typeof value !== "number"
  ) {
    invalid(`${path} must be a JSON scalar`);
  }
  if (typeof value === "number") number(value, path);
}

function equal(
  value: unknown,
  expected: string | number | boolean | null,
  path: string,
  code: ArtifactLoadErrorCode = "SEMA_ARTIFACT_INVALID_MANIFEST",
): void {
  if (value !== expected)
    throw new ArtifactLoadError(
      code,
      `${path} must equal ${JSON.stringify(expected)}`,
    );
}

function uniquePrimitive(values: readonly unknown[], path: string): void {
  const seen = new Set<unknown>();
  for (const value of values) {
    if (seen.has(value)) invalid(`${path} contains duplicates`);
    seen.add(value);
  }
}

function uniqueSemantic(values: readonly unknown[], path: string): void {
  const seen = new Set<string>();
  for (const value of values)
    addUnique(seen, hex(semanticBytes(value)), path, "semantic value");
}

function strictSemanticOrder(values: readonly unknown[], path: string): void {
  let previous: Uint8Array | undefined;
  for (const value of values) {
    const current = semanticBytes(value);
    if (previous && compareBytes(previous, current) >= 0)
      invalid(`${path} is not in strict semantic byte order`);
    previous = current;
  }
}

function addUnique(
  set: Set<string>,
  value: string,
  path: string,
  description: string,
): void {
  if (set.has(value))
    invalid(`${path} contains duplicate ${description} ${value}`);
  set.add(value);
}

function invalid(message: string): never {
  throw new ArtifactLoadError("SEMA_ARTIFACT_INVALID_MANIFEST", message);
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1)
    throw new RangeError(`${name} must be a positive safe integer`);
  return value;
}

function digest(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

type TypedNode =
  | readonly ["null"]
  | readonly ["boolean", boolean]
  | readonly ["number", string]
  | readonly ["string", string]
  | readonly ["array", readonly TypedNode[]]
  | readonly ["object", readonly (readonly [string, TypedNode])[]];

function semanticBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(
    JSON.stringify(["semantscript-semantic-json", 1, typedNode(value)]),
  );
}

function typedNode(value: unknown): TypedNode {
  if (value === null) return ["null"];
  if (typeof value === "boolean") return ["boolean", value];
  if (typeof value === "string") return ["string", value];
  if (typeof value === "number") {
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setFloat64(0, value, false);
    return ["number", hex(new Uint8Array(buffer))];
  }
  if (Array.isArray(value)) return ["array", value.map(typedNode)];
  const encoder = new TextEncoder();
  const entries = Object.entries(value as JsonRecord)
    .map(([name, entry]) => [name, typedNode(entry)] as const)
    .sort(([left], [right]) =>
      compareBytes(encoder.encode(left), encoder.encode(right)),
    );
  return ["object", entries];
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function compareBytes(left: Uint8Array, right: Uint8Array): number {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const difference = (left[index] ?? 0) - (right[index] ?? 0);
    if (difference !== 0) return difference;
  }
  return left.length - right.length;
}

function compareSemver(left: string, right: string): number {
  const leftMatch = SEMVER.exec(left);
  const rightMatch = SEMVER.exec(right);
  if (!leftMatch || !rightMatch)
    throw new TypeError("invalid semver comparison");
  for (let index = 1; index <= 3; index += 1) {
    const difference = Number(leftMatch[index]) - Number(rightMatch[index]);
    if (difference !== 0) return difference;
  }
  const leftPre = leftMatch[4];
  const rightPre = rightMatch[4];
  if (leftPre === undefined) return rightPre === undefined ? 0 : 1;
  if (rightPre === undefined) return -1;
  const leftParts = leftPre.split(".");
  const rightParts = rightPre.split(".");
  for (
    let index = 0;
    index < Math.max(leftParts.length, rightParts.length);
    index += 1
  ) {
    const leftPart = leftParts[index];
    const rightPart = rightParts[index];
    if (leftPart === undefined) return -1;
    if (rightPart === undefined) return 1;
    if (leftPart === rightPart) continue;
    const leftNumber = /^[0-9]+$/u.test(leftPart);
    const rightNumber = /^[0-9]+$/u.test(rightPart);
    if (leftNumber && rightNumber) return Number(leftPart) - Number(rightPart);
    if (leftNumber) return -1;
    if (rightNumber) return 1;
    return leftPart < rightPart ? -1 : 1;
  }
  return 0;
}

function sameShape(
  left: readonly (number | string)[],
  right: readonly (number | string)[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => {
      const expected = right[index];
      return expected === "BATCH"
        ? value === "BATCH" || value === 1
        : value === expected;
    })
  );
}

function compatibleEdgeShape(
  left: readonly (number | string)[],
  right: readonly (number | string)[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => {
      const other = right[index];
      if (
        index === 0 &&
        (value === "BATCH" || value === 1) &&
        (other === "BATCH" || other === 1)
      ) {
        return true;
      }
      if (
        (value === "HIDDEN" && typeof other === "number" && other > 0) ||
        (other === "HIDDEN" && typeof value === "number" && value > 0)
      ) {
        return true;
      }
      return value === other;
    })
  );
}

function batch(value: number | string | undefined): boolean {
  return value === "BATCH" || value === 1;
}

function deepFreeze<T>(value: T): T {
  if (
    value !== null &&
    typeof value === "object" &&
    !ArrayBuffer.isView(value) &&
    !Object.isFrozen(value)
  ) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}
