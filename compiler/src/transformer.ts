/**
 * ts-patch program transformer. One tsconfig entry compiles every sema site of
 * a plain tsc project and writes the IR bundle beside its output:
 *
 *     "plugins": [{ "transform": "@semantscript/compiler/transformer" }]
 *
 * Optional keys on that entry: `application` (names the encoder and adapter
 * refs) and `bundlePath` (default `outDir/semantscript.ir.v1.json`).
 */
import type ts from "typescript";

import {
  planSemaProgramBuild,
  SemaBuildError,
  type SemaBuildOptions,
} from "./build-tools.js";
import { createSemaProgramTransformer } from "./compile.js";

export interface SemaTransformerConfig extends SemaBuildOptions {
  /** ts-patch's own key naming this module; ignored here. */
  readonly transform?: string;
}

export interface SemaTransformerExtras {
  readonly addDiagnostic?: (diagnostic: ts.Diagnostic) => number;
}

export default function semantscriptTransformer(
  program: ts.Program,
  config: SemaTransformerConfig = {},
  extras: SemaTransformerExtras = {},
): ts.TransformerFactory<ts.SourceFile> {
  try {
    const build = planSemaProgramBuild(program, config);
    return createSemaProgramTransformer(build.program, build.plan);
  } catch (error) {
    if (error instanceof SemaBuildError && extras.addDiagnostic !== undefined) {
      for (const diagnostic of error.diagnostics) {
        extras.addDiagnostic(diagnostic);
      }

      return () => (sourceFile) => sourceFile;
    }

    throw error;
  }
}
