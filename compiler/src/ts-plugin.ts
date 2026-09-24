/**
 * TypeScript language-service plugin: hover and inline diagnostics at every
 * sema site, from the compiler's analysis and the latest trained artifact.
 * One tsconfig entry enables it in VS Code and any tsserver client:
 *
 *     "plugins": [{ "name": "@semantscript/compiler/ts-plugin" }]
 *
 * Optional keys: `artifact` (root, default `.semantscript/artifact` beside
 * the tsconfig), `accuracyThreshold` (default 0.95) and `eceThreshold`
 * (default 0.1, the trainer's gate).
 */
import { readFileSync, statSync } from "node:fs";
import path from "node:path";

import type ts from "typescript";

import {
  planSemaCompilationSync,
  type PlanSemaCompilationResult,
  type PlannedSemaSite,
} from "./compile.js";
import { findMalformedSemaSites, findSemaSites } from "./sema-sites.js";

export interface SemaPluginConfig {
  readonly name?: string;
  readonly artifact?: string;
  readonly accuracyThreshold?: number;
  readonly eceThreshold?: number;
}

/** Diagnostic codes the plugin adds beyond the compiler's 9100 to 9131. */
export const PLUGIN_DIAGNOSTICS = {
  noArtifact: 9150,
  notInArtifact: 9151,
  accuracyBelowThreshold: 9152,
  eceAboveThreshold: 9153,
} as const;

const DEFAULT_ARTIFACT = ".semantscript/artifact";
const DEFAULT_ACCURACY_THRESHOLD = 0.95;
const DEFAULT_ECE_THRESHOLD = 0.1;
const DIAGNOSTIC_SOURCE = "semantscript";

interface ArtifactFunctionRecord {
  readonly status: string;
  readonly accuracy: number;
  readonly ece: number;
  readonly brier: number;
  readonly pairConsistency: number;
  readonly attestedCases: number;
  readonly constraintViolations: number;
}

interface ArtifactSnapshot {
  readonly root: string;
  readonly release: string;
  readonly functions: ReadonlyMap<string, ArtifactFunctionRecord>;
}

type LoadedArtifact =
  | { readonly kind: "missing"; readonly root: string }
  | {
      readonly kind: "unreadable";
      readonly root: string;
      readonly reason: string;
    }
  | { readonly kind: "loaded"; readonly snapshot: ArtifactSnapshot };

/** The state behind one project's proxied language service. */
export class SemaEditorState {
  private readonly plans = new WeakMap<ts.Program, PlanSemaCompilationResult>();
  private artifactCache:
    { readonly key: string; readonly value: LoadedArtifact } | undefined;

  constructor(
    private readonly typescript: typeof ts,
    private readonly projectRoot: string,
    private readonly config: SemaPluginConfig,
  ) {}

  get artifactRoot(): string {
    return path.resolve(
      this.projectRoot,
      this.config.artifact ?? DEFAULT_ARTIFACT,
    );
  }

  /** The compiler's diagnostics for `fileName` plus the artifact-backed ones. */
  diagnostics(program: ts.Program, fileName: string): ts.Diagnostic[] {
    const sourceFile = program.getSourceFile(fileName);
    if (sourceFile === undefined || !fileName.endsWith(".sem.ts")) return [];
    const plan = this.plan(program);
    if (!plan.ok) {
      return plan.diagnostics
        .filter(
          (diagnostic) => diagnostic.file?.fileName === sourceFile.fileName,
        )
        .map((diagnostic) => ({ ...diagnostic, source: DIAGNOSTIC_SOURCE }));
    }
    const artifact = this.artifact();
    const diagnostics: ts.Diagnostic[] = [];
    for (const site of plan.value.sites) {
      if (site.sourceFile.fileName !== sourceFile.fileName) continue;
      const message = this.artifactMessage(site, artifact);
      if (message !== undefined) {
        diagnostics.push({
          category: this.typescript.DiagnosticCategory.Warning,
          code: message.code,
          file: sourceFile,
          start: site.start,
          length: site.end - site.start,
          messageText: message.text,
          source: DIAGNOSTIC_SOURCE,
        });
      }
    }
    return diagnostics;
  }

  /** Hover text for the sema site containing `position`, if any. */
  quickInfo(
    program: ts.Program,
    fileName: string,
    position: number,
  ): ts.QuickInfo | undefined {
    const sourceFile = program.getSourceFile(fileName);
    if (sourceFile === undefined || !fileName.endsWith(".sem.ts"))
      return undefined;
    const plan = this.plan(program);
    if (plan.ok) {
      const site = plan.value.sites.find(
        (candidate) =>
          candidate.sourceFile.fileName === sourceFile.fileName &&
          candidate.start <= position &&
          position <= candidate.end,
      );
      if (site === undefined) return undefined;
      return this.quickInfoFor(
        site.start,
        site.end,
        this.describePlanned(site, this.artifact()),
      );
    }
    const within = (candidate: {
      readonly sourceFile: ts.SourceFile;
      readonly location: { readonly start: number; readonly end: number };
    }): boolean =>
      candidate.sourceFile.fileName === sourceFile.fileName &&
      candidate.location.start <= position &&
      position <= candidate.location.end;
    const location =
      findSemaSites(program, sourceFile).find(within)?.location ??
      findMalformedSemaSites(program).find(within)?.location;
    if (location === undefined) return undefined;
    return this.quickInfoFor(
      location.start,
      location.end,
      this.describeUnplanned(sourceFile, location, plan.diagnostics),
    );
  }

  private plan(program: ts.Program): PlanSemaCompilationResult {
    let plan = this.plans.get(program);
    if (plan === undefined) {
      plan = planSemaCompilationSync(program, {
        projectRoot: this.projectRoot,
        // Editors analyse unsaved buffers, so the digest comes from the service's text.
        readSourceBytesSync: (sourceFile) =>
          new TextEncoder().encode(sourceFile.text),
      });
      this.plans.set(program, plan);
    }
    return plan;
  }

  private artifact(): LoadedArtifact {
    const root = this.artifactRoot;
    const pointerPath = path.join(root, "current.json");
    let key: string;
    try {
      const stats = statSync(pointerPath);
      key = `${pointerPath}\u0000${String(stats.mtimeMs)}\u0000${String(stats.size)}`;
    } catch {
      const missing: LoadedArtifact = { kind: "missing", root };
      this.artifactCache = { key: "missing", value: missing };
      return missing;
    }
    if (this.artifactCache?.key === key) return this.artifactCache.value;
    const value = readArtifact(root, pointerPath);
    this.artifactCache = { key, value };
    return value;
  }

  private artifactMessage(
    site: PlannedSemaSite,
    artifact: LoadedArtifact,
  ): { readonly code: number; readonly text: string } | undefined {
    const guidance =
      site.record.definition.examples.length === 0
        ? " This expression has no examples; adding a few is the first thing to try."
        : "";
    if (artifact.kind === "missing") {
      return {
        code: PLUGIN_DIAGNOSTICS.noArtifact,
        text: `sema expression is unverified: no trained artifact at ${artifact.root}. Run semantscript train.${guidance}`,
      };
    }
    if (artifact.kind === "unreadable") {
      return {
        code: PLUGIN_DIAGNOSTICS.noArtifact,
        text: `sema expression is unverified: the artifact at ${artifact.root} cannot be read (${artifact.reason}).`,
      };
    }
    const record = artifact.snapshot.functions.get(site.functionId);
    if (record === undefined) {
      return {
        code: PLUGIN_DIAGNOSTICS.notInArtifact,
        text: `sema expression is unverified: not in the latest artifact (${artifact.snapshot.release}). It changed since training or was never trained; run semantscript train.${guidance}`,
      };
    }
    const accuracyThreshold =
      this.config.accuracyThreshold ?? DEFAULT_ACCURACY_THRESHOLD;
    const eceThreshold = this.config.eceThreshold ?? DEFAULT_ECE_THRESHOLD;
    if (record.accuracy < accuracyThreshold) {
      return {
        code: PLUGIN_DIAGNOSTICS.accuracyBelowThreshold,
        text: `verified accuracy ${formatRatio(record.accuracy)} is below ${formatRatio(accuracyThreshold)}.${guidance || " Add examples for the cases it misses, or a constraint for the rule it breaks, then retrain."}`,
      };
    }
    if (record.ece > eceThreshold) {
      return {
        code: PLUGIN_DIAGNOSTICS.eceAboveThreshold,
        text: `expected calibration error ${formatRatio(record.ece)} is above ${formatRatio(eceThreshold)}; its confidence is not trustworthy. More training cases or a confidence threshold with a fallback help.`,
      };
    }
    return undefined;
  }

  private describePlanned(
    site: PlannedSemaSite,
    artifact: LoadedArtifact,
  ): string {
    const record = site.record;
    const lines = [
      `sema function ${shortId(site.functionId)} → ${record.output.tsType}`,
      `inputs ${String(record.inputs.length)} · examples ${String(record.definition.examples.length)} · constraints ${String(record.definition.constraints.length)} · confidence threshold ${record.runtime.confidenceThreshold === null ? "none" : formatRatio(record.runtime.confidenceThreshold)}`,
    ];
    if (artifact.kind === "missing") {
      lines.push(
        `artifact: none at ${artifact.root} (unverified; run semantscript train)`,
      );
    } else if (artifact.kind === "unreadable") {
      lines.push(
        `artifact: unreadable at ${artifact.root} (${artifact.reason})`,
      );
    } else {
      const trained = artifact.snapshot.functions.get(site.functionId);
      lines.push(
        trained === undefined
          ? `artifact ${artifact.snapshot.release}: this expression is not in it (unverified; run semantscript train)`
          : `artifact ${artifact.snapshot.release}: ${trained.status} · accuracy ${formatRatio(trained.accuracy)} · ECE ${formatRatio(trained.ece)} · pair consistency ${formatRatio(trained.pairConsistency)} · attested cases ${String(trained.attestedCases)} · constraint violations ${String(trained.constraintViolations)}`,
      );
    }
    const message = this.artifactMessage(site, artifact);
    if (message !== undefined) lines.push(`⚠ ${message.text}`);
    return lines.join("\n");
  }

  private describeUnplanned(
    sourceFile: ts.SourceFile,
    location: { readonly start: number; readonly end: number },
    diagnostics: readonly ts.Diagnostic[],
  ): string {
    const own = diagnostics.filter(
      (diagnostic) =>
        diagnostic.file?.fileName === sourceFile.fileName &&
        diagnostic.start !== undefined &&
        diagnostic.start >= location.start &&
        diagnostic.start <= location.end,
    );
    const lines = [
      "sema expression (not compiled: the project has sema diagnostics)",
    ];
    for (const diagnostic of own) {
      lines.push(
        `⚠ TS${String(diagnostic.code)}: ${this.typescript.flattenDiagnosticMessageText(diagnostic.messageText, "\n")}`,
      );
    }
    return lines.join("\n");
  }

  private quickInfoFor(start: number, end: number, text: string): ts.QuickInfo {
    return {
      kind: this.typescript.ScriptElementKind.functionElement,
      kindModifiers: "",
      textSpan: { start, length: end - start },
      displayParts: [{ text: "sema", kind: "keyword" }],
      documentation: [{ text, kind: "text" }],
    };
  }
}

function readArtifact(root: string, pointerPath: string): LoadedArtifact {
  try {
    const pointer = JSON.parse(readFileSync(pointerPath, "utf8")) as {
      release?: unknown;
    };
    if (typeof pointer.release !== "string")
      throw new Error("current.json has no release");
    const manifestPath = path.join(root, pointer.release, "manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
      functions?: unknown;
    };
    if (!Array.isArray(manifest.functions))
      throw new Error("manifest has no functions");
    const functions = new Map<string, ArtifactFunctionRecord>();
    for (const entry of manifest.functions as readonly unknown[]) {
      const fn = entry as {
        id?: unknown;
        verification?: Record<string, unknown>;
      };
      const verification = fn.verification;
      if (typeof fn.id !== "string" || verification === undefined) continue;
      functions.set(fn.id, {
        status:
          typeof verification["status"] === "string"
            ? verification["status"]
            : "unknown",
        accuracy: numberOr(verification["accuracy"]),
        ece: numberOr(verification["ece"]),
        brier: numberOr(verification["brier"]),
        pairConsistency: numberOr(verification["pairConsistency"]),
        attestedCases: numberOr(verification["attestedCases"]),
        constraintViolations: numberOr(verification["constraintViolations"]),
      });
    }
    return {
      kind: "loaded",
      snapshot: { root, release: pointer.release, functions },
    };
  } catch (error) {
    return {
      kind: "unreadable",
      root,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

function numberOr(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : Number.NaN;
}

function formatRatio(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}

function shortId(functionId: string): string {
  return `${functionId.slice(0, 11)}…`;
}

/** The plugin module factory tsserver calls with its own TypeScript instance. */
export function createSemaLanguageServicePlugin(mod: {
  readonly typescript: typeof ts;
}): ts.server.PluginModule {
  const typescript = mod.typescript;
  return {
    create(info: ts.server.PluginCreateInfo): ts.LanguageService {
      const service = info.languageService;
      const config = (info.config ?? {}) as SemaPluginConfig;
      const state = new SemaEditorState(
        typescript,
        info.project.getCurrentDirectory(),
        config,
      );
      const proxy: ts.LanguageService = Object.create(
        null,
      ) as ts.LanguageService;
      const proxyRecord = proxy as unknown as Record<string, unknown>;
      const serviceRecord = service as unknown as Record<string, unknown>;
      for (const key of Object.keys(serviceRecord)) {
        const member = serviceRecord[key];
        if (typeof member === "function") {
          proxyRecord[key] = (...args: unknown[]) =>
            (member as (...inner: unknown[]) => unknown).apply(service, args);
        }
      }
      proxy.getSemanticDiagnostics = (fileName) => {
        const own = service.getSemanticDiagnostics(fileName);
        const program = service.getProgram();
        if (program === undefined) return own;
        try {
          return [...own, ...state.diagnostics(program, fileName)];
        } catch (error) {
          info.project.projectService.logger.info(
            `semantscript plugin diagnostics failed: ${String(error)}`,
          );
          return own;
        }
      };
      proxy.getQuickInfoAtPosition = (fileName, position) => {
        const program = service.getProgram();
        if (program !== undefined) {
          try {
            const info = state.quickInfo(program, fileName, position);
            if (info !== undefined) return info;
          } catch (error) {
            info.project.projectService.logger.info(
              `semantscript plugin hover failed: ${String(error)}`,
            );
          }
        }
        return service.getQuickInfoAtPosition(fileName, position);
      };
      return proxy;
    },
  };
}

export default createSemaLanguageServicePlugin;
