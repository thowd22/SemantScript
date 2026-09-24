/**
 * webpack- and Turbopack-compatible loader for `.sem.ts` modules, the
 * adoption path for Next.js:
 *
 *     // next.config.ts
 *     turbopack: { rules: { "*.sem.ts": { loaders: ["@semantscript/compiler/loader"] } } }
 *
 * The loader plans the whole project once per process and re-plans when a
 * source it compiled changes on disk. The IR bundle is written to
 * `bundlePath` (default `outDir/semantscript.ir.v1.json`, or next to the
 * tsconfig when there is no `outDir`). The emitted JavaScript is valid
 * TypeScript, so the bundler's own TypeScript step may run after it.
 */
import path from "node:path";

import {
  loadSemaProjectBuild,
  SemaBuildError,
  type SemaBuildOptions,
  type SemaProjectBuild,
} from "./build-tools.js";

export type { SemaBuildOptions } from "./build-tools.js";

/** The subset of webpack's loader context the loader uses. */
export interface SemaLoaderContext {
  readonly resourcePath: string;
  readonly rootContext?: string;
  readonly context?: string;
  getOptions?: () => SemaBuildOptions | undefined;
  addDependency?: (fileName: string) => void;
  callback: (
    error: Error | null,
    content?: string,
    sourceMap?: Record<string, unknown>,
  ) => void;
}

const projects = new Map<string, SemaProjectBuild>();

export default function semantscriptLoader(
  this: SemaLoaderContext,
  source: string,
): void {
  const options = this.getOptions?.() ?? {};
  const cwd =
    this.rootContext ?? this.context ?? path.dirname(this.resourcePath);
  const key = `${cwd}\u0000${options.tsconfig ?? ""}`;

  try {
    let project = projects.get(key);
    const current = project?.program.getSourceFile(this.resourcePath);

    if (
      project === undefined ||
      current === undefined ||
      current.text !== source
    ) {
      project = loadSemaProjectBuild(options, {
        cwd,
        ...(project === undefined ? {} : { oldProgram: project.program }),
      });
      projects.set(key, project);
    }

    if (project.configPath !== undefined) {
      this.addDependency?.(project.configPath);
    }

    const emitted = project.emit(this.resourcePath);
    // webpack and Turbopack both take the map as a parsed object.
    this.callback(
      null,
      emitted.code,
      emitted.map === undefined
        ? undefined
        : (JSON.parse(emitted.map) as Record<string, unknown>),
    );
  } catch (error) {
    if (error instanceof SemaBuildError) {
      this.callback(error);
      return;
    }

    throw error;
  }
}

export { semantscriptLoader };
