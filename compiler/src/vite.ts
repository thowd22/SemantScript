/**
 * Vite plugin. One line of `vite.config` compiles every `.sem.ts` module
 * through TypeScript with the SemantScript rewrite and writes the IR bundle
 * into Vite's `build.outDir`:
 *
 *     plugins: [semantscript()]
 *
 * The project is re-planned on the next transform after any watched change, so
 * the dev server picks up edited sites and types.
 */
import path from "node:path";

import type { Plugin } from "vite";

import {
  isSemaSourceFileName,
  loadSemaProjectBuild,
  SemaBuildError,
  type SemaBuildOptions,
  type SemaProjectBuild,
} from "./build-tools.js";

export type { SemaBuildOptions } from "./build-tools.js";

export default function semantscript(options: SemaBuildOptions = {}): Plugin {
  let cwd = process.cwd();
  let bundleDirectory: string | undefined;
  let project: SemaProjectBuild | undefined;
  let stale = true;

  return {
    name: "semantscript",
    enforce: "pre",
    configResolved(config) {
      cwd = config.root;
      bundleDirectory = path.resolve(config.root, config.build.outDir);
    },
    buildStart() {
      stale = true;
    },
    watchChange() {
      stale = true;
    },
    writeBundle() {
      // Vite empties `build.outDir` after transforms ran, so put the bundle back.
      project?.writeBundle();
    },
    transform(_code, id) {
      const fileName = id.split("?", 1)[0] ?? id;

      if (!isSemaSourceFileName(fileName) || !path.isAbsolute(fileName)) {
        return null;
      }

      try {
        if (stale || project === undefined) {
          project = loadSemaProjectBuild(options, {
            cwd,
            ...(bundleDirectory === undefined ? {} : { bundleDirectory }),
            ...(project === undefined ? {} : { oldProgram: project.program }),
          });
          stale = false;
        }

        if (project.configPath !== undefined) {
          this.addWatchFile(project.configPath);
        }

        const emitted = project.emit(fileName);
        return { code: emitted.code, map: emitted.map ?? null };
      } catch (error) {
        if (error instanceof SemaBuildError) {
          return this.error(error.message);
        }

        throw error;
      }
    },
  };
}

export { semantscript };
