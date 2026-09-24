export type ArtifactResourceRole = "tokenizer" | "encoder" | "adapter" | "head";

export interface ArtifactPointerV1 {
  readonly kind: "semantscript.artifact-pointer";
  readonly pointerVersion: 1;
  readonly release: string;
  readonly manifestSha256: string;
}

export interface TensorDescriptorV1 {
  readonly name: string;
  readonly dtype: "int64" | "float16" | "float32";
  readonly shape: readonly (number | string)[];
}

export type OnnxPrecisionV1 = "float32" | "int8-dynamic";

export interface OnnxQuantizationV1 {
  readonly method: "dynamic";
  readonly weightType: "int8" | "uint8";
  readonly perChannel: boolean;
  readonly reduceRange: boolean;
  readonly argmaxDisagreementTolerance: number;
  readonly attestedDisagreementTolerance: number;
  readonly eceThreshold: number;
  readonly sourceManifestSha256: string;
}

export interface OnnxAbiV1 {
  readonly opset: number;
  readonly inputs: readonly TensorDescriptorV1[];
  readonly outputs: readonly TensorDescriptorV1[];
  readonly externalData: false;
  /** Absent means float32: every graph published before quantization existed. */
  readonly precision?: OnnxPrecisionV1;
  /** Present exactly when precision is a quantized one; records how the graph was derived. */
  readonly quantization?: OnnxQuantizationV1;
}

interface ArtifactResourceBaseV1 {
  readonly ref: string;
  readonly role: ArtifactResourceRole;
  readonly formatVersion: 1;
  readonly path: string;
  readonly byteLength: number;
  readonly sha256: string;
}

export interface TokenizerResourceV1 extends ArtifactResourceBaseV1 {
  readonly role: "tokenizer";
  readonly format: "tokenizer-json";
  readonly maximumSequenceLength: number;
}

export interface OnnxResourceV1 extends ArtifactResourceBaseV1 {
  readonly role: "encoder" | "adapter" | "head";
  readonly format: "onnx";
  readonly onnx: OnnxAbiV1;
}

export type ArtifactResourceV1 = TokenizerResourceV1 | OnnxResourceV1;

export type InputTypeV1 =
  | { readonly kind: "string" | "boolean" | "number" | "null" }
  | {
      readonly kind: "literal";
      readonly value: string | number | boolean | null;
    }
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
    }
  | { readonly kind: "array"; readonly items: InputTypeV1 }
  | { readonly kind: "tuple"; readonly items: readonly InputTypeV1[] }
  | {
      readonly kind: "object";
      readonly name: string;
      readonly fields: readonly InputFieldV1[];
    }
  | { readonly kind: "union"; readonly variants: readonly InputTypeV1[] };

export interface InputFieldV1 {
  readonly name: string;
  readonly optional: boolean;
  readonly type: InputTypeV1;
}

export interface InputEntryV1 {
  readonly name: string;
  readonly index: number;
  readonly tsType: string;
  readonly type: InputTypeV1;
}

export type RuntimeHeadTypeV1 =
  | { readonly kind: "boolean"; readonly support: readonly [false, true] }
  | {
      readonly kind: "nominal-string" | "ordinal-string";
      readonly support: readonly string[];
    }
  | { readonly kind: "nominal-number"; readonly support: readonly number[] }
  | {
      readonly kind: "ordinal-number";
      readonly sourceKind: "bounded-int" | "bounded-number";
      readonly minimum: string;
      readonly maximum: string;
      readonly step: string;
      readonly supportDecimal: readonly string[];
    };

export interface CalibrationV1 {
  readonly method: "temperature-scaling";
  readonly temperature: number;
  readonly ece: number;
  readonly brier: number;
  readonly sampleCount: number;
  readonly splitSha256: string;
  readonly eceBins: number;
}

export interface HeadBindingV1 {
  readonly outputPath: readonly [] | readonly [string];
  readonly headRef: string;
  readonly type: RuntimeHeadTypeV1;
  readonly parameterization: "binary-sigmoid" | "categorical-softmax";
  readonly calibration: CalibrationV1;
  readonly verification: {
    readonly accuracy: number;
    readonly pairConsistency: number;
  };
}

export interface ArtifactFunctionV1 {
  readonly id: string;
  readonly semanticSha256: string;
  readonly inputs: readonly InputEntryV1[];
  readonly inputSchemaSha256: string;
  readonly outputSchemaSha256: string;
  readonly adapterRef: string;
  readonly heads: readonly HeadBindingV1[];
  readonly runtime: {
    readonly resultMode: "value" | "diagnostic";
    readonly confidenceThreshold: number | null;
    readonly policy: "none" | "scalar-top1" | "all-fields";
    readonly fallbackRef: string | null;
  };
  readonly verification: {
    readonly status: "passed";
    readonly accuracy: number;
    readonly ece: number;
    readonly brier: number;
    readonly pairConsistency: number;
    readonly attestedCases: number;
    readonly exampleFailures: 0;
    readonly constraintViolations: number;
    readonly typeErrors: 0;
  };
  readonly trainingProvenance: {
    readonly datasetSha256: string;
    readonly trainingKeySha256: string;
    readonly teacher: string;
    readonly baseModel: string;
  };
}

export interface ApplicationArtifactManifestV1 {
  readonly kind: "semantscript.application-artifact";
  readonly artifactVersion: 1;
  readonly irVersion: 1;
  readonly compatibility: {
    readonly runtimeAbiVersion: 1;
    readonly modelAbiVersion: 1;
    readonly canonicalInput: "semantscript.canonical-input/v1";
    readonly minimumRuntimeVersion: string;
    readonly requiredCapabilities: readonly string[];
  };
  readonly application: {
    readonly id: string;
    readonly version: string;
  };
  readonly build: {
    readonly createdAt: string;
    readonly compilerVersion: string;
    readonly trainerVersion: string;
    readonly sourceIrSha256: string;
  };
  readonly resources: readonly ArtifactResourceV1[];
  readonly model: {
    readonly tokenizerRef: string;
    readonly encoderRef: string;
    readonly adapterRef: string;
  };
  readonly functions: readonly ArtifactFunctionV1[];
}

export interface ArtifactResourceBackend<PreparedResource> {
  prepareResource(
    resource: ArtifactResourceV1,
    verifiedBytes: Uint8Array,
  ): PreparedResource | Promise<PreparedResource>;
  disposeResource?(prepared: PreparedResource): void | Promise<void>;
}

export interface StagedArtifactResource<PreparedResource> {
  readonly metadata: ArtifactResourceV1;
  readonly prepared: PreparedResource;
}

export interface StagedArtifactDescriptor<PreparedResource> {
  readonly artifactRoot: string;
  readonly releaseDirectory: string;
  readonly manifestPath: string;
  readonly manifestSha256: string;
  readonly manifest: ApplicationArtifactManifestV1;
  readonly resources: readonly StagedArtifactResource<PreparedResource>[];
  readonly functions: readonly ArtifactFunctionV1[];
}
