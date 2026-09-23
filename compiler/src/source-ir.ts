import type { ResolvedConstraint, ResolvedExample } from "./definition.js";
import type { OutputSpec, TemplatePart } from "./ir-types.js";
import type { SemaSiteAnalysis } from "./analyze-site.js";

export interface SourceIrOptions {
  readonly id: string;
  readonly semanticSha256: string;
  readonly sourcePath: string;
  readonly sourceSha256: string;
  readonly encoderRef: string;
  readonly adapterRef: string;
  readonly headRefs: readonly string[];
  readonly confidenceThreshold?: number | null;
  readonly fallbackRef?: string | null;
  readonly examples?: readonly ResolvedExample[];
  readonly constraints?: readonly ResolvedConstraint[];
}

export interface SourceNeuralFunctionIr {
  readonly kind: "semantscript.neural-function";
  readonly irVersion: 1;
  readonly stage: "source";
  readonly id: string;
  readonly semanticSha256: string;
  readonly source: {
    readonly path: string;
    readonly line: number;
    readonly column: number;
    readonly sourceSha256: string;
  };
  readonly definition: {
    readonly template: readonly TemplatePart[];
    readonly examples: readonly ResolvedExample[];
    readonly constraints: readonly ResolvedConstraint[];
  };
  readonly inputs: SemaSiteAnalysis["ir"]["inputs"];
  readonly output: OutputSpec;
  readonly model: {
    readonly encoder: string;
    readonly adapter: string;
    readonly heads: readonly {
      readonly outputPath: string;
      readonly ref: string;
    }[];
  };
  readonly runtime: {
    readonly resultMode: "value" | "diagnostic";
    readonly confidenceThreshold: number | null;
    readonly fallbackRef: string | null;
    readonly synchronous: true;
  };
  readonly trainingProvenance: { readonly status: "pending" };
  readonly verification: { readonly status: "pending" };
}

export function createSourceNeuralFunctionIr(
  analysis: SemaSiteAnalysis,
  options: SourceIrOptions,
): SourceNeuralFunctionIr {
  const outputPaths = outputPathsFor(analysis.ir.output);

  if (options.headRefs.length !== outputPaths.length) {
    throw new RangeError(
      `expected ${String(outputPaths.length)} head reference${outputPaths.length === 1 ? "" : "s"}, received ${String(options.headRefs.length)}`,
    );
  }

  return {
    kind: "semantscript.neural-function",
    irVersion: 1,
    stage: "source",
    id: options.id,
    semanticSha256: options.semanticSha256,
    source: {
      path: options.sourcePath,
      line: analysis.site.location.line,
      column: analysis.site.location.column,
      sourceSha256: options.sourceSha256,
    },
    definition: {
      template: analysis.ir.template,
      examples: options.examples ?? [],
      constraints: options.constraints ?? [],
    },
    inputs: analysis.ir.inputs,
    output: analysis.ir.output,
    model: {
      encoder: options.encoderRef,
      adapter: options.adapterRef,
      heads: outputPaths.map((outputPath, index) => ({
        outputPath,
        ref: options.headRefs[index] ?? "",
      })),
    },
    runtime: {
      resultMode: analysis.ir.resultMode,
      confidenceThreshold: options.confidenceThreshold ?? null,
      fallbackRef: options.fallbackRef ?? null,
      synchronous: true,
    },
    trainingProvenance: { status: "pending" },
    verification: { status: "pending" },
  };
}

function outputPathsFor(output: OutputSpec): readonly string[] {
  if (output.kind === "scalar") {
    return [""];
  }

  return output.fields.map(({ name }) => `/${escapeJsonPointerSegment(name)}`);
}

function escapeJsonPointerSegment(value: string): string {
  return value.replaceAll("~", "~0").replaceAll("/", "~1");
}
