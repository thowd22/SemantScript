/**
 * esbuild plugin. One line of build config compiles every `.sem.ts` file
 * through TypeScript with the SemantScript rewrite and writes the IR bundle
 * into esbuild's `outdir` (or beside its `outfile`):
 *
 *     plugins: [semantscript()]
 */
import { Buffer } from "node:buffer";
import path from "node:path";

import type { Location, PartialMessage, Plugin } from "esbuild";
import ts from "typescript";

import {
  loadSemaProjectBuild,
  SemaBuildError,
  type SemaBuildOptions,
  type SemaProjectBuild,
} from "./build-tools.js";

export type { SemaBuildOptions } from "./build-tools.js";

export default function semantscript(options: SemaBuildOptions = {}): Plugin {
  return {
    name: "semantscript",
    setup(build) {
      const cwd = build.initialOptions.absWorkingDir ?? process.cwd();
      const bundleDirectory = outputDirectory(build.initialOptions, cwd);
      let project: SemaProjectBuild | undefined;
      let failure: PartialMessage[] | undefined;

      build.onStart(() => {
        try {
          project = loadSemaProjectBuild(options, {
            cwd,
            ...(bundleDirectory === undefined ? {} : { bundleDirectory }),
            ...(project === undefined ? {} : { oldProgram: project.program }),
          });
          failure = undefined;
          return null;
        } catch (error) {
          if (!(error instanceof SemaBuildError)) {
            throw error;
          }

          project = undefined;
          failure = toMessages(error, cwd);
          return { errors: failure };
        }
      });

      // esbuild compiles filters as Go regular expressions, which reject the `u` flag.
      build.onLoad({ filter: /\.sem\.ts$/ }, (args) => {
        if (project === undefined) {
          // onStart already failed the build with the planning diagnostics.
          return failure === undefined
            ? { errors: [{ text: "the SemantScript project did not load" }] }
            : { contents: "", loader: "js" };
        }

        try {
          const emitted = project.emit(args.path);
          const map =
            emitted.map === undefined
              ? ""
              : `\n//# sourceMappingURL=data:application/json;base64,${Buffer.from(emitted.map, "utf8").toString("base64")}\n`;
          return {
            contents: `${emitted.code}${map}`,
            loader: "js",
            resolveDir: path.dirname(args.path),
            watchFiles:
              project.configPath === undefined ? [] : [project.configPath],
          };
        } catch (error) {
          if (!(error instanceof SemaBuildError)) {
            throw error;
          }

          return { errors: toMessages(error, cwd) };
        }
      });
    },
  };
}

export { semantscript };

function outputDirectory(
  options: { readonly outdir?: string; readonly outfile?: string },
  cwd: string,
): string | undefined {
  if (options.outdir !== undefined) {
    return path.resolve(cwd, options.outdir);
  }

  if (options.outfile !== undefined) {
    return path.dirname(path.resolve(cwd, options.outfile));
  }

  return undefined;
}

function toMessages(error: SemaBuildError, cwd: string): PartialMessage[] {
  return error.diagnostics.map((diagnostic) => ({
    pluginName: "semantscript",
    text: ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
    location: toLocation(diagnostic, cwd),
  }));
}

function toLocation(
  diagnostic: ts.Diagnostic,
  cwd: string,
): Partial<Location> | null {
  if (diagnostic.file === undefined || diagnostic.start === undefined) {
    return null;
  }

  const { line, character } = diagnostic.file.getLineAndCharacterOfPosition(
    diagnostic.start,
  );
  const lineStart = diagnostic.file.getPositionOfLineAndCharacter(line, 0);
  const lineText =
    diagnostic.file.text.slice(lineStart).split(/\r?\n/u, 1)[0] ?? "";
  return {
    file: path.relative(cwd, diagnostic.file.fileName),
    line: line + 1,
    column: character,
    length: diagnostic.length ?? 0,
    lineText,
  };
}
