import { watch } from "node:fs";
import { relative, resolve, sep } from "node:path";
import { parseArgs } from "node:util";

import { compileProject } from "./build.js";
import {
  CliUsageError,
  stringOption,
  type CliIo,
  type OptionValues,
} from "./io.js";
import { runTrain, TRAIN_OPTIONS } from "./train.js";

const SOURCE_EXTENSIONS = [".ts", ".mts", ".cts", ".tsx"];
const IGNORED_SEGMENTS = new Set(["node_modules", ".git", ".semantscript"]);
const DEFAULT_DEBOUNCE_MILLISECONDS = 300;

/**
 * `semantscript dev`: build, train, then watch the project's TypeScript
 * sources and repeat on every save. The train cache retrains only the
 * expressions whose IR changed; the trainer streams progress per expression;
 * a running application that loaded the artifact with `watch: true` swaps in
 * each published release without restarting.
 */
export async function devCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: {
      ...TRAIN_OPTIONS,
      project: { type: "string", short: "p" },
      application: { type: "string" },
      debounce: { type: "string" },
      once: { type: "boolean" },
    },
    allowPositionals: false,
  });
  const debounce = parseDebounce(stringOption(values, "debounce"));
  const project = stringOption(values, "project");
  const application = stringOption(values, "application");
  const trainValues: OptionValues = Object.fromEntries(
    Object.entries(values).filter(
      ([name]) =>
        !["project", "application", "debounce", "once"].includes(name),
    ),
  );
  let cycle = 0;
  let projectRoot = io.cwd;
  // The environment preflight runs until one training succeeds, not on every save.
  let preflightPending = true;

  const runCycle = async (reason: string): Promise<number> => {
    cycle += 1;
    io.stderr(`semantscript dev: cycle ${String(cycle)} (${reason})\n`);
    const built = await compileProject(
      {
        ...(project === undefined ? {} : { project }),
        ...(application === undefined ? {} : { application }),
        ...(stringOption(values, "bundle") === undefined
          ? {}
          : { bundle: stringOption(values, "bundle") ?? "" }),
      },
      io,
    );
    projectRoot = built.projectRoot;
    if (built.status !== 0 || built.bundlePath === undefined) {
      io.stderr(
        "semantscript dev: build failed; the previous artifact stays in service\n",
      );
      return built.status;
    }
    const status = await runTrain(
      { ...trainValues, bundle: built.bundlePath },
      io,
      { preflight: preflightPending, command: "dev" },
    );
    if (status === 0) preflightPending = false;
    io.stderr(
      status === 0
        ? "semantscript dev: release published; watched runtimes reload it\n"
        : "semantscript dev: training failed; the previous artifact stays in service\n",
    );
    return status;
  };

  const first = await runCycle("initial build");
  if (values.once === true) return first;
  if (io.signal?.aborted === true) return 0;

  io.stderr(
    `semantscript dev: watching ${relative(io.cwd, projectRoot) || "."} for changes (Ctrl-C to stop)\n`,
  );
  await watchLoop(projectRoot, debounce, runCycle, io);
  return 0;
}

function watchLoop(
  projectRoot: string,
  debounce: number,
  runCycle: (reason: string) => Promise<number>,
  io: CliIo,
): Promise<void> {
  return new Promise((finish) => {
    let timer: NodeJS.Timeout | undefined;
    let running = false;
    let pending: string | undefined;
    const watcher = watch(projectRoot, { recursive: true }, (_event, name) => {
      if (name === null) return;
      const fileName = resolve(projectRoot, name);
      if (!isRelevantSource(fileName, projectRoot)) return;
      schedule(fileName);
    });
    const stop = (): void => {
      if (timer !== undefined) clearTimeout(timer);
      watcher.close();
      finish();
    };
    const drain = (reason: string): void => {
      if (running) {
        pending = reason;
        return;
      }
      running = true;
      void runCycle(reason)
        .catch((error: unknown) => {
          io.stderr(
            `semantscript dev: ${error instanceof Error ? error.message : String(error)}\n`,
          );
          return 1;
        })
        .then(() => {
          running = false;
          if (io.signal?.aborted === true) {
            stop();
            return;
          }
          if (pending !== undefined) {
            const next = pending;
            pending = undefined;
            drain(next);
          }
        });
    };
    const schedule = (fileName: string): void => {
      if (timer !== undefined) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = undefined;
        drain(`${relative(io.cwd, fileName)} changed`);
      }, debounce);
    };
    watcher.on("error", (error) => {
      io.stderr(`semantscript dev: watcher error: ${error.message}\n`);
      stop();
    });
    io.signal?.addEventListener(
      "abort",
      () => {
        if (!running) stop();
      },
      { once: true },
    );
  });
}

function isRelevantSource(fileName: string, projectRoot: string): boolean {
  const relativePath = relative(projectRoot, fileName);
  if (relativePath.startsWith("..")) return false;
  if (
    relativePath.split(sep).some((segment) => IGNORED_SEGMENTS.has(segment))
  ) {
    return false;
  }
  if (relativePath === "tsconfig.json") return true;
  if (fileName.endsWith(".d.ts")) return false;
  return SOURCE_EXTENSIONS.some((extension) => fileName.endsWith(extension));
}

function parseDebounce(value: string | undefined): number {
  if (value === undefined) return DEFAULT_DEBOUNCE_MILLISECONDS;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new CliUsageError(
      "--debounce must be a non-negative integer of milliseconds",
    );
  }
  return parsed;
}
