import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { delimiter, dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import {
  CliUsageError,
  stringOption,
  type CliIo,
  type OptionValues,
} from "./io.js";

const REPOSITORY_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../..",
);

/** Where `init`, `train`, `test` and `run` agree the artifact lives, unless overridden. */
export const DEFAULT_ARTIFACT_PATH = ".semantscript/artifact";
export const ARTIFACT_ENVIRONMENT_VARIABLE = "SEMANTSCRIPT_ARTIFACT";
export const BUNDLE_FILE_NAME = "semantscript.ir.v1.json";
/** Teacher files looked up, in order, when `--teacher` is not passed. */
export const TEACHER_CONFIG_CANDIDATES = [
  "semantscript.teacher.toml",
  "teacher.toml",
  ".semantscript/teacher.toml",
] as const;
/**
 * Teacher keywords `--teacher` accepts in place of a TOML path when no file of that
 * name exists: `constraints` is the built-in teacher that labels inputs with the
 * expression's own constraints (no language model, no key).
 */
export const BUILT_IN_TEACHERS = ["constraints"] as const;
export const DEFAULT_TEACHER_MODEL = "claude-sonnet-5";
export const DEFAULT_TEACHER_TOML = `[teacher]
backend = "anthropic"
model = "${DEFAULT_TEACHER_MODEL}"
`;
export const PYTHON_ENVIRONMENT_VARIABLE = "SEMANTSCRIPT_PYTHON";
export const DEFAULT_TRAINER_MODULE = "semantscript_trainer.cli";

/** `--python`, else `SEMANTSCRIPT_PYTHON`, else `python3` (`python` on Windows). */
export function resolvePython(values: OptionValues, io: CliIo): string {
  return (
    stringOption(values, "python") ??
    nonEmpty(io.env[PYTHON_ENVIRONMENT_VARIABLE]) ??
    (process.platform === "win32" ? "python" : "python3")
  );
}

function nonEmpty(value: string | undefined): string | undefined {
  return value === undefined || value.length === 0 ? undefined : value;
}

/**
 * The teacher file `train` would use, without writing anything: `--teacher`, else the
 * first existing file of `TEACHER_CONFIG_CANDIDATES`, else undefined. A built-in
 * teacher keyword (`--teacher constraints`) passes through unchanged unless a file
 * of that name exists.
 */
export function findTeacherConfig(
  values: OptionValues,
  io: CliIo,
): string | undefined {
  const requested = stringOption(values, "teacher");
  if (requested !== undefined) {
    const path = resolve(io.cwd, requested);
    if (isBuiltInTeacher(requested) && !isFile(path)) return requested;
    return path;
  }
  for (const candidate of TEACHER_CONFIG_CANDIDATES) {
    const path = resolve(io.cwd, candidate);
    if (existsSync(path)) return path;
  }
  return undefined;
}

function isFile(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

export function isBuiltInTeacher(value: string): boolean {
  return (BUILT_IN_TEACHERS as readonly string[]).includes(value);
}

/** `--artifact`, else `SEMANTSCRIPT_ARTIFACT`, else `.semantscript/artifact`, resolved against cwd. */
export function resolveArtifactRoot(values: OptionValues, io: CliIo): string {
  return resolve(
    io.cwd,
    stringOption(values, "artifact") ??
      io.env[ARTIFACT_ENVIRONMENT_VARIABLE] ??
      DEFAULT_ARTIFACT_PATH,
  );
}

/**
 * `--bundle`, else the bundle the build wrote: `<tsconfig outDir>/semantscript.ir.v1.json`
 * when the project's tsconfig sets one, then the conventional output directories.
 */
export function resolveBundlePath(values: OptionValues, io: CliIo): string {
  const requested = stringOption(values, "bundle");
  if (requested !== undefined) return resolve(io.cwd, requested);
  const candidates = bundleCandidates(io.cwd);
  const found = candidates.find((candidate) => existsSync(candidate));
  if (found === undefined) {
    throw new CliUsageError(
      `--bundle is required: no ${BUNDLE_FILE_NAME} under ${candidates
        .map((candidate) => dirname(candidate))
        .join(", ")} (run the build first)`,
    );
  }
  return found;
}

export function bundleCandidates(cwd: string): readonly string[] {
  const directories = new Set<string>();
  const outDir = tsconfigOutDir(cwd);
  if (outDir !== undefined) directories.add(outDir);
  for (const conventional of [".", "dist", "out", "build"]) {
    directories.add(resolve(cwd, conventional));
  }
  return [...directories].map((directory) => join(directory, BUNDLE_FILE_NAME));
}

function tsconfigOutDir(cwd: string): string | undefined {
  const configPath = join(cwd, "tsconfig.json");
  if (!existsSync(configPath)) return undefined;
  const host: ts.ParseConfigFileHost = {
    ...ts.sys,
    onUnRecoverableConfigFileDiagnostic: () => undefined,
  };
  const parsed = ts.getParsedCommandLineOfConfigFile(configPath, {}, host);
  const outDir = parsed?.options.outDir;
  return outDir === undefined ? undefined : resolve(cwd, outDir);
}

/**
 * `--teacher`, else the first teacher file of `TEACHER_CONFIG_CANDIDATES`, else a
 * generated `.semantscript/teacher.toml` for the Anthropic backend when
 * `ANTHROPIC_API_KEY` is set (the key itself stays in the environment).
 */
export function resolveTeacherConfig(values: OptionValues, io: CliIo): string {
  const found = findTeacherConfig(values, io);
  if (found !== undefined) return found;
  const apiKey = io.env["ANTHROPIC_API_KEY"];
  if (apiKey !== undefined && apiKey.length > 0) {
    const generated = resolve(io.cwd, ".semantscript/teacher.toml");
    mkdirSync(dirname(generated), { recursive: true });
    writeFileSync(generated, DEFAULT_TEACHER_TOML);
    io.stderr(
      `semantscript train: wrote ${generated} (Anthropic backend, ${DEFAULT_TEACHER_MODEL}); edit it to change the teacher\n`,
    );
    return generated;
  }
  throw new CliUsageError(
    `--teacher is required: no ${TEACHER_CONFIG_CANDIDATES.join(", ")} found and ANTHROPIC_API_KEY is not set (set it to use the default Anthropic teacher, write a [teacher] TOML, or pass --teacher constraints when the constraints decide every input)`,
  );
}

/** The monorepo's trainer and model sources when the CLI runs from the checkout. */
export function pythonPath(existing: string | undefined): string {
  const entries: string[] = [];
  if (
    existsSync(join(REPOSITORY_ROOT, "trainer", "src", "semantscript_trainer"))
  ) {
    entries.push(
      join(REPOSITORY_ROOT, "trainer", "src"),
      join(REPOSITORY_ROOT, "model", "src"),
    );
    const localPackages = join(REPOSITORY_ROOT, ".python-packages");
    if (existsSync(localPackages)) entries.push(localPackages);
  }
  if (existing !== undefined && existing.length > 0) entries.push(existing);
  return entries.join(delimiter);
}
