import {
  existsSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";

import ts from "typescript";

import {
  createSemaProgramTransformer,
  emitSemaSourceFile,
  planSemaCompilationSync,
  type EmittedSemaSourceFile,
  type SemaCompilationPlan,
} from "./compile.js";

const DEFAULT_BUNDLE_NAME = "semantscript.ir.v1.json";
const APPLICATION_ID = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/u;
const BUILD_DIAGNOSTIC = 9140;

/** Options shared by every build-tool adapter (transformer, esbuild, Vite, loader). */
export interface SemaBuildOptions {
  /** The application's tsconfig; defaults to the nearest `tsconfig.json`. */
  readonly tsconfig?: string;
  /** Application id that names the encoder and adapter refs; default `application`. */
  readonly application?: string;
  /** Where the IR bundle is written; see each adapter for its default. */
  readonly bundlePath?: string;
  /** Depth routing per domain (see the compiler README on routed domains). */
  readonly domainDepths?: Readonly<Record<string, number>>;
  /** Route domains (one adapter each) even without depths or `@domain` headers. */
  readonly routeDomains?: boolean;
}

export interface SemaProjectBuildDefaults {
  /** Directory that relative options resolve against. */
  readonly cwd: string;
  /** The build tool's output directory, the default bundle location. */
  readonly bundleDirectory?: string;
  /** The previous build's program, reused for incremental re-planning. */
  readonly oldProgram?: ts.Program;
}

/** A planned project whose `.sem.ts` files a build tool emits one at a time. */
export interface SemaProjectBuild {
  readonly program: ts.Program;
  readonly plan: SemaCompilationPlan;
  readonly projectRoot: string;
  readonly configPath: string | undefined;
  readonly bundlePath: string;
  /** Emits one `.sem.ts` file (absolute path) as JavaScript plus its map. */
  emit(fileName: string): EmittedSemaSourceFile;
  /** Rewrites the IR bundle when its file is missing or differs (Vite empties `outDir`). */
  writeBundle(): void;
}

/** Planning or emission failed; `diagnostics` carry the TypeScript positions. */
export class SemaBuildError extends Error {
  readonly diagnostics: readonly ts.Diagnostic[];

  constructor(diagnostics: readonly ts.Diagnostic[], cwd: string) {
    super(formatSemaBuildDiagnostics(diagnostics, cwd).trimEnd());
    this.name = "SemaBuildError";
    this.diagnostics = diagnostics;
  }
}

/** Whether a file name ends in `.sem.ts` (or `.sem.mts`/`.sem.cts`), the files the adapters rewrite. */
export function isSemaSourceFileName(fileName: string): boolean {
  return fileName.endsWith(".sem.ts");
}

/** Render build diagnostics as `file:line:column: message` lines for a bundler's error output. */
export function formatSemaBuildDiagnostics(
  diagnostics: readonly ts.Diagnostic[],
  cwd: string,
): string {
  return ts.formatDiagnostics(diagnostics, {
    getCanonicalFileName: (fileName) => fileName,
    getCurrentDirectory: () => cwd,
    getNewLine: () => "\n",
  });
}

/**
 * Loads the application's tsconfig, builds a TypeScript program whose emit
 * suits a bundler (per-file JavaScript with separate source maps, no
 * declarations), plans every sema site and writes the IR bundle.
 */
export function loadSemaProjectBuild(
  options: SemaBuildOptions,
  defaults: SemaProjectBuildDefaults,
): SemaProjectBuild {
  const cwd = path.resolve(defaults.cwd);
  const configPath = resolveConfigPath(options.tsconfig, cwd);
  const parsed = parseConfig(configPath, cwd);
  const projectRoot = path.dirname(configPath);
  const program = ts.createProgram({
    rootNames: parsed.fileNames,
    options: bundlerCompilerOptions(parsed.options),
    ...(parsed.projectReferences === undefined
      ? {}
      : { projectReferences: parsed.projectReferences }),
    ...(defaults.oldProgram === undefined
      ? {}
      : { oldProgram: defaults.oldProgram }),
  });
  const bundlePath = resolveBundlePath(options, {
    cwd,
    projectRoot,
    outDir: parsed.options.outDir,
    bundleDirectory: defaults.bundleDirectory,
  });
  return planSemaProgramBuild(program, options, {
    projectRoot,
    configPath,
    bundlePath,
    cwd,
  });
}

/**
 * Plans an existing program (the ts-patch transformer receives tsc's own) and
 * writes the IR bundle. `projectRoot` defaults to the tsconfig's directory.
 */
export function planSemaProgramBuild(
  program: ts.Program,
  options: SemaBuildOptions,
  defaults: {
    readonly projectRoot?: string;
    readonly configPath?: string;
    readonly bundlePath?: string;
    readonly cwd?: string;
  } = {},
): SemaProjectBuild {
  const configPath = defaults.configPath ?? configPathOfProgram(program);
  const cwd = defaults.cwd ?? program.getCurrentDirectory();
  const projectRoot =
    defaults.projectRoot ??
    (configPath === undefined ? cwd : path.dirname(configPath));
  const application = options.application ?? "application";

  if (!APPLICATION_ID.test(application)) {
    throw new SemaBuildError(
      [
        buildDiagnostic(
          `application must be lowercase letters, digits and single dashes, not ${JSON.stringify(application)}`,
        ),
      ],
      cwd,
    );
  }

  const bundlePath =
    defaults.bundlePath ??
    resolveBundlePath(options, {
      cwd,
      projectRoot,
      outDir: program.getCompilerOptions().outDir,
    });
  const planned = planSemaCompilationSync(program, {
    projectRoot,
    encoderRef: `encoder.${application}`,
    adapterRef: `adapter.${application}`,
    ...(options.domainDepths === undefined
      ? {}
      : { domainDepths: options.domainDepths }),
    ...(options.routeDomains === true ? { routeDomains: true } : {}),
  });

  if (!planned.ok) {
    throw new SemaBuildError(planned.diagnostics, cwd);
  }

  const plan = planned.value;
  assertBundlePathIsFree(program, bundlePath, cwd);
  writeBundle(bundlePath, plan.bundleText);
  // Validates the plan against the program once, before any file is emitted.
  createSemaProgramTransformer(program, plan);

  return {
    program,
    plan,
    projectRoot,
    configPath,
    bundlePath,
    writeBundle() {
      writeBundle(bundlePath, plan.bundleText);
    },
    emit(fileName) {
      const sourceFile = program.getSourceFile(path.resolve(cwd, fileName));

      if (sourceFile === undefined) {
        throw new SemaBuildError(
          [
            buildDiagnostic(
              `${fileName} is not part of the TypeScript project${configPath === undefined ? "" : ` ${configPath}`}`,
            ),
          ],
          cwd,
        );
      }

      const emitted = emitSemaSourceFile(program, plan, sourceFile);

      if (!emitted.ok) {
        throw new SemaBuildError(emitted.diagnostics, cwd);
      }

      return emitted.value;
    },
  };
}

function resolveConfigPath(requested: string | undefined, cwd: string): string {
  if (requested !== undefined) {
    return path.resolve(cwd, requested);
  }

  const found = ts.findConfigFile(cwd, (fileName) =>
    ts.sys.fileExists(fileName),
  );

  if (found === undefined) {
    throw new SemaBuildError(
      [
        buildDiagnostic(
          `no tsconfig.json found from ${cwd}; pass the tsconfig option`,
        ),
      ],
      cwd,
    );
  }

  return path.resolve(found);
}

function parseConfig(configPath: string, cwd: string): ts.ParsedCommandLine {
  const configDiagnostics: ts.Diagnostic[] = [];
  const host: ts.ParseConfigFileHost = {
    ...ts.sys,
    onUnRecoverableConfigFileDiagnostic: (diagnostic) => {
      configDiagnostics.push(diagnostic);
    },
  };
  const parsed = ts.getParsedCommandLineOfConfigFile(configPath, {}, host);

  if (parsed === undefined || configDiagnostics.length > 0) {
    throw new SemaBuildError(
      configDiagnostics.length > 0
        ? configDiagnostics
        : [buildDiagnostic(`unable to read ${configPath}`)],
      cwd,
    );
  }

  if (parsed.errors.length > 0) {
    throw new SemaBuildError(parsed.errors, cwd);
  }

  return parsed;
}

/**
 * Bundlers consume per-file JavaScript with separate maps and do their own
 * declaration-free, whole-program work, so the emit-shape options are fixed
 * here. A script-style `module` (CommonJS and older, which TypeScript numbers
 * below ES2015) becomes ESNext when the resolution mode allows it; Node16-style
 * resolution keeps the package's own module format.
 */
function bundlerCompilerOptions(
  options: ts.CompilerOptions,
): ts.CompilerOptions {
  const kept: ts.CompilerOptions = { ...options };
  kept.composite = false;
  kept.declaration = false;
  kept.declarationMap = false;
  kept.emitDeclarationOnly = false;
  kept.incremental = false;
  kept.inlineSourceMap = false;
  kept.noEmit = false;
  kept.noEmitOnError = false;
  kept.sourceMap = true;
  kept.inlineSources = true;
  delete kept.declarationDir;
  delete kept.outFile;
  delete kept.tsBuildInfoFile;
  const moduleKind = kept.module;
  const resolution = kept.moduleResolution;
  const nodeResolution =
    resolution === ts.ModuleResolutionKind.Node16 ||
    resolution === ts.ModuleResolutionKind.NodeNext;

  if (
    moduleKind === undefined ||
    (isScriptModuleKind(moduleKind) && !nodeResolution)
  ) {
    kept.module = ts.ModuleKind.ESNext;
  }

  return kept;
}

function isScriptModuleKind(moduleKind: ts.ModuleKind): boolean {
  return moduleKind < ts.ModuleKind.ES2015;
}

function resolveBundlePath(
  options: SemaBuildOptions,
  context: {
    readonly cwd: string;
    readonly projectRoot: string;
    readonly outDir: string | undefined;
    readonly bundleDirectory?: string | undefined;
  },
): string {
  if (options.bundlePath !== undefined) {
    return path.resolve(context.cwd, options.bundlePath);
  }

  const directory =
    context.bundleDirectory ??
    (context.outDir === undefined
      ? context.projectRoot
      : path.resolve(context.projectRoot, context.outDir));
  return path.join(directory, DEFAULT_BUNDLE_NAME);
}

function configPathOfProgram(program: ts.Program): string | undefined {
  const configFilePath = program.getCompilerOptions()["configFilePath"];
  return typeof configFilePath === "string"
    ? path.resolve(configFilePath)
    : undefined;
}

function assertBundlePathIsFree(
  program: ts.Program,
  bundlePath: string,
  cwd: string,
): void {
  const target = canonicalPath(bundlePath);

  for (const sourceFile of program.getSourceFiles()) {
    if (canonicalPath(sourceFile.fileName) === target) {
      throw new SemaBuildError(
        [
          buildDiagnostic(
            `IR bundle path ${bundlePath} conflicts with a TypeScript source file`,
          ),
        ],
        cwd,
      );
    }
  }
}

function canonicalPath(fileName: string): string {
  const resolved = path.resolve(fileName);

  try {
    return realpathSync.native(resolved);
  } catch {
    return resolved;
  }
}

function writeBundle(bundlePath: string, bundleText: string): void {
  if (existsSync(bundlePath)) {
    let current: string | undefined;

    try {
      current = readFileSync(bundlePath, "utf8");
    } catch {
      current = undefined;
    }

    if (current === bundleText) {
      return;
    }
  }

  mkdirSync(path.dirname(bundlePath), { recursive: true });
  writeFileSync(bundlePath, bundleText);
}

function buildDiagnostic(messageText: string): ts.Diagnostic {
  return {
    category: ts.DiagnosticCategory.Error,
    code: BUILD_DIAGNOSTIC,
    file: undefined,
    start: undefined,
    length: undefined,
    messageText,
  };
}
