import { Buffer } from "node:buffer";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
  copyFile,
  cp,
  lstat,
  mkdir,
  readdir,
  readFile,
  readlink,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { ARTIFACT_ENVIRONMENT_VARIABLE, BUNDLE_FILE_NAME } from "./defaults.js";
import {
  CliUsageError,
  objectOf,
  stringOption,
  type CliIo,
  type OptionValues,
} from "./io.js";
import {
  isEnoent,
  pointerBytes,
  readPointer,
  type ArtifactSummary,
} from "./manifest.js";
import {
  fileSha256,
  formatBytes,
  scanReleases,
  treeSize,
  verifyRelease,
} from "./releases.js";
import { renderTable } from "./table.js";

export const PACKAGE_KIND = "semantscript.package";
export const PACKAGE_MANIFEST = "semantscript-package.json";
export const DEFAULT_PACKAGE_PATH = ".semantscript/package";

/**
 * A packaging failure with a stable code, reported with exit status 1 (the
 * same shape as ReleaseError).
 */
export class PackageError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(`${code}: ${message}`);
    this.name = "PackageError";
    this.code = code;
  }
}

export interface PackageTarget {
  readonly name: string;
  readonly bytes: number;
  readonly description: string;
}

/**
 * Deployment size limits, checked against the providers' quota pages on
 * 2026-09-25. AWS writes MB for 1,048,576 bytes; Google's 500 MB is taken as
 * 500,000,000 bytes, the stricter reading.
 */
export const PACKAGE_TARGETS: Readonly<Record<string, PackageTarget>> = {
  "lambda-zip": {
    name: "lambda-zip",
    bytes: 250 * 1024 * 1024,
    description:
      "AWS Lambda .zip deployment package: 250 MB unzipped, including layers (docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)",
  },
  "lambda-image": {
    name: "lambda-image",
    bytes: 10 * 1024 * 1024 * 1024,
    description:
      "AWS Lambda container image: 10 GB uncompressed, including every layer and the base image (docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)",
  },
  "cloud-run-functions": {
    name: "cloud-run-functions",
    bytes: 500_000_000,
    description:
      "Cloud Run functions source deployment: 500 MB uncompressed sources plus modules (docs.cloud.google.com/functions/quotas); Cloud Run services have no image size limit",
  },
};

/**
 * Encoder sizes measured on ModernBERT-base (22 layers, float32): the depth
 * sweep (benchmarks/refund/data/results-depth-sweep-2026-09-24/results.json)
 * and the int8 derivation (data/release-int8-tolerant-2026-09-24). Levers
 * scale an encoder by these ratios.
 */
export const MEASURED_ENCODER = {
  fullDepth: 22,
  fullBytes: 596_679_464,
  depthBytes: { 4: 235_118_960, 6: 275_297_443, 12: 395_833_756 },
  int8Bytes: 150_073_785,
} as const;

export interface PackageLever {
  readonly lever:
    "depth" | "route-remaining" | "int8" | "depth+int8" | "smaller-encoder";
  readonly label: string;
  /** The projected bundle total; null for a smaller encoder, whose size depends on the model. */
  readonly projectedBytes: number | null;
  /** For a smaller encoder: the largest encoder graph that fits. */
  readonly encoderBudgetBytes: number | null;
  readonly fits: boolean;
  /** False for a lever that is a measurement only (int8): no command produces it yet. */
  readonly actionable: boolean;
  readonly how: string;
}

export interface LeverInput {
  readonly totalBytes: number;
  readonly encoderBytes: number;
  readonly limitBytes: number;
  /** The release already ships a depth-routed prefix encoder (see isDepthRouted). */
  readonly depthRouted: boolean;
  /** An encoder resource is already quantized (onnx.precision int8-dynamic). */
  readonly quantized: boolean;
  /**
   * For a mixed release (some domains routed to a prefix, some at full
   * depth): the bytes of the full-depth encoder graphs, which routing the
   * remaining domains would drop. Zero or absent otherwise.
   */
  readonly fullEncoderBytes?: number;
  /** For a mixed release: the functions still on the full-depth encoder. */
  readonly fullDepthFunctions?: readonly string[];
}

const INT8_TOLERANCE =
  "and only under a recorded tolerance: the measured int8 refund release changed 1 of 80 attested cases and 0.105% of decisions, which the strict gate refused (it was measured under attestedDisagreementTolerance 2 and argmaxDisagreementTolerance 0.01; docs/deploy.md has the figures)";
const INT8_HOW = `a measurement, not a step to run: no int8 derivation exists for applications yet (it has only been measured on the refund benchmark, through a refund-specific driver), ${INT8_TOLERANCE}`;

const DEPTH_PREFIX = /(?:^|[./])depth-\d{3}(?:$|[./])/u;

export interface DepthRouting {
  /** Any domain is routed to a prefix encoder. */
  readonly routed: boolean;
  /**
   * Encoder resources (by ref) at full depth in a routed release: a mixed
   * release keeps them for the domains left at full depth. Empty for a
   * release that is not routed or is fully routed.
   */
  readonly fullEncoderRefs: readonly string[];
  /** Functions of a mixed release that still run the full-depth encoder. */
  readonly fullDepthFunctions: readonly string[];
}

/**
 * How a release uses depth routing. A prefix encoder is one whose ref or
 * path names a depth-NNN prefix, or one a function names in its encoderRef
 * that is not model.encoderRef. A mixed release has a prefix and a
 * full-depth encoder (the default for functions with no encoderRef); a
 * fully routed one exports only the prefix graph and names it in
 * model.encoderRef.
 */
export function depthRouting(
  manifest: Readonly<Record<string, unknown>>,
): DepthRouting {
  const model = manifest["model"];
  const modelRef =
    typeof model === "object" &&
    model !== null &&
    typeof (model as Record<string, unknown>)["encoderRef"] === "string"
      ? ((model as Record<string, unknown>)["encoderRef"] as string)
      : undefined;
  const functions = listFunctions(manifest);
  const functionRefs = new Set(
    functions
      .map((fn) => fn["encoderRef"])
      .filter((ref): ref is string => typeof ref === "string"),
  );
  const isPrefix = (ref: string | undefined, path?: unknown): boolean =>
    (ref !== undefined && DEPTH_PREFIX.test(ref)) ||
    (typeof path === "string" && DEPTH_PREFIX.test(path)) ||
    (ref !== undefined && ref !== modelRef && functionRefs.has(ref));
  const encoders = listResources(manifest).filter(
    (resource) => resource["role"] === "encoder",
  );
  const prefixRefs = new Set<string>();
  const fullRefs: string[] = [];
  for (const encoder of encoders) {
    const ref = typeof encoder["ref"] === "string" ? encoder["ref"] : undefined;
    if (isPrefix(ref, encoder["path"])) {
      if (ref !== undefined) prefixRefs.add(ref);
    } else if (ref !== undefined) {
      fullRefs.push(ref);
    }
  }
  const routed =
    prefixRefs.size > 0 ||
    [...functionRefs].some((ref) => ref !== modelRef) ||
    (modelRef !== undefined && DEPTH_PREFIX.test(modelRef)) ||
    encoders.some((encoder) => isPrefix(undefined, encoder["path"]));
  if (!routed) {
    return { routed: false, fullEncoderRefs: [], fullDepthFunctions: [] };
  }
  const fullSet = new Set(fullRefs);
  const fullDepthFunctions = functions
    .filter((fn) => {
      const ref =
        typeof fn["encoderRef"] === "string" ? fn["encoderRef"] : modelRef;
      return ref !== undefined && fullSet.has(ref);
    })
    .map((fn) => (typeof fn["id"] === "string" ? fn["id"] : "?"));
  return { routed: true, fullEncoderRefs: fullRefs, fullDepthFunctions };
}

/** Whether a release already uses depth routing, mixed or full (see depthRouting). */
export function isDepthRouted(
  manifest: Readonly<Record<string, unknown>>,
): boolean {
  return depthRouting(manifest).routed;
}

/**
 * The size levers for a bundle over its limit, each with the projected
 * bundle total, skipping any the release already uses. Encoder sizes scale
 * by the ModernBERT-base measurements, so for another encoder they are
 * estimates. Training writes a float32 graph, so for an already quantized
 * encoder the depth levers scale its float32 size, not the int8 one.
 */
export function packageLevers(input: LeverInput): readonly PackageLever[] {
  const rest = input.totalBytes - input.encoderBytes;
  const int8Ratio = MEASURED_ENCODER.int8Bytes / MEASURED_ENCODER.fullBytes;
  const float32Encoder = input.quantized
    ? input.encoderBytes / int8Ratio
    : input.encoderBytes;
  const scaled = (bytes: number): number =>
    Math.round(float32Encoder * (bytes / MEASURED_ENCODER.fullBytes));
  const lever = (
    kind: PackageLever["lever"],
    label: string,
    encoder: number,
    how: string,
    actionable = true,
  ): PackageLever => ({
    lever: kind,
    label,
    projectedBytes: rest + encoder,
    encoderBudgetBytes: null,
    fits: rest + encoder <= input.limitBytes,
    actionable,
    how,
  });
  const levers: PackageLever[] = [];
  const budget = input.limitBytes - rest;
  const smaller: PackageLever = {
    lever: "smaller-encoder",
    label: "a smaller encoder",
    projectedBytes: null,
    encoderBudgetBytes: Math.max(budget, 0),
    fits: budget > 0,
    actionable: true,
    how:
      budget > 0
        ? `semantscript train --encoder-name <model> with an encoder whose ONNX graph is at most ${formatBytes(budget)} (${String(budget)} bytes)`
        : `nothing fits: the bundle without its encoder is already ${formatBytes(rest)}, so no encoder lever helps; the dependencies, the compiled output or the includes are what is over`,
  };
  // When the rest of the bundle is already over, no encoder lever can help.
  if (budget <= 0) return [smaller];
  const depths = [12, 6, 4] as const;
  const trained = input.quantized
    ? "then train (training writes a float32 prefix, so the int8 encoder is replaced by a larger graph; the size shown is that float32 prefix)"
    : "then train";
  if (!input.depthRouted) {
    for (const depth of depths) {
      levers.push(
        lever(
          "depth",
          `depth routing to ${String(depth)} layers`,
          scaled(MEASURED_ENCODER.depthBytes[depth]),
          `give every domain that depth (semantscript build --domain-depth <domain>=${String(depth)} once per domain, or domainDepths in a tspc plugin entry), ${trained}; a domain left at full depth keeps the full encoder in the bundle beside the prefix (the refund policy kept 160/160 on its final set at 4, 6 and 12 layers)`,
        ),
      );
    }
  }
  const fullEncoderBytes = input.fullEncoderBytes ?? 0;
  if (input.depthRouted && fullEncoderBytes > 0) {
    const remaining = input.fullDepthFunctions ?? [];
    const named =
      remaining.length === 0
        ? ""
        : ` (still at full depth: ${remaining.slice(0, 5).join(", ")}${remaining.length > 5 ? `, and ${String(remaining.length - 5)} more` : ""})`;
    levers.push(
      lever(
        "route-remaining",
        "route the remaining domains",
        input.encoderBytes - fullEncoderBytes,
        `give the domains left at full depth the depth the others use${named} (semantscript build --domain-depth <domain>=<depth>, or domainDepths in a tspc plugin entry), then train; the release then exports only the prefix and drops the ${formatBytes(fullEncoderBytes)} full-depth encoder`,
      ),
    );
  }
  if (!input.quantized) {
    levers.push(
      lever(
        "int8",
        "int8 dynamic quantization",
        Math.round(input.encoderBytes * int8Ratio),
        INT8_HOW,
        false,
      ),
    );
    if (!input.depthRouted) {
      for (const depth of [6, 4] as const) {
        levers.push(
          lever(
            "depth+int8",
            `depth ${String(depth)} and int8`,
            Math.round(scaled(MEASURED_ENCODER.depthBytes[depth]) * int8Ratio),
            `give every domain depth ${String(depth)} (semantscript build --domain-depth <domain>=${String(depth)} once per domain), then train; the int8 half is a measurement only, as for int8 above`,
            false,
          ),
        );
      }
    }
  }
  levers.push(smaller);
  return levers;
}

export interface PruneResult {
  readonly removed: readonly string[];
  readonly bytes: number;
}

/**
 * Remove the native binaries of other platforms from a production
 * node_modules: onnxruntime-node's bin/napi-v*\/<os>/<arch> directories that
 * are not the target and its CUDA and TensorRT provider libraries (the
 * runtime runs the CPU provider only), and tokenizers' prebuilt .node files
 * for other triples (both libc variants of the target are kept). A package
 * left without a binding for the target is a PACKAGE_NO_BINDING error.
 */
export async function pruneNativeBindings(
  nodeModules: string,
  platform: string,
  arch: string,
): Promise<PruneResult> {
  const removed: string[] = [];
  let bytes = 0;
  const drop = async (path: string): Promise<void> => {
    bytes += await treeSize(path);
    removed.push(relative(nodeModules, path).split(sep).join("/"));
    await rm(path, { recursive: true, force: true });
  };
  for (const pkg of await findPackages(nodeModules, "onnxruntime-node")) {
    const bin = join(pkg, "bin");
    let kept = 0;
    for (const napi of await listDirectory(bin)) {
      if (!napi.startsWith("napi-v")) continue;
      for (const os of await listDirectory(join(bin, napi))) {
        if (os !== platform) {
          await drop(join(bin, napi, os));
          continue;
        }
        for (const cpu of await listDirectory(join(bin, napi, os))) {
          const directory = join(bin, napi, os, cpu);
          if (cpu !== arch) {
            await drop(directory);
            continue;
          }
          for (const file of await listDirectory(directory)) {
            if (/providers_(?:cuda|tensorrt)/u.test(file)) {
              await drop(join(directory, file));
            }
          }
          kept += 1;
        }
      }
    }
    if (kept === 0) {
      throw new PackageError(
        "PACKAGE_NO_BINDING",
        `${relative(nodeModules, pkg)} ships no binding for ${platform}/${arch}`,
      );
    }
  }
  for (const pkg of await findPackages(nodeModules, "tokenizers")) {
    const nodes = (await listDirectory(pkg)).filter((file) =>
      /^tokenizers\..+\.node$/u.test(file),
    );
    const keep = (file: string): boolean => {
      const triple = file.slice("tokenizers.".length, -".node".length);
      return (
        triple === `${platform}-${arch}` ||
        triple.startsWith(`${platform}-${arch}-`) ||
        (platform === "darwin" && triple === "darwin-universal")
      );
    };
    if (nodes.length > 0 && !nodes.some(keep)) {
      throw new PackageError(
        "PACKAGE_NO_BINDING",
        `${relative(nodeModules, pkg)} ships no binding for ${platform}/${arch}`,
      );
    }
    for (const file of nodes) {
      if (!keep(file)) await drop(join(pkg, file));
    }
  }
  return { removed: removed.sort(), bytes };
}

/** Every installed copy of a package under node_modules, nested ones included. */
async function findPackages(
  nodeModules: string,
  name: string,
): Promise<readonly string[]> {
  const found: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await listDirectory(directory)) {
      if (entry.startsWith(".")) continue;
      const path = join(directory, entry);
      if (!(await isRealDirectory(path))) continue;
      if (entry.startsWith("@")) {
        for (const scoped of await listDirectory(path)) {
          const scopedPath = join(path, scoped);
          if (await isRealDirectory(scopedPath)) {
            await visitPackage(`${entry}/${scoped}`, scopedPath);
          }
        }
        continue;
      }
      await visitPackage(entry, path);
    }
  };
  const visitPackage = async (
    packageName: string,
    path: string,
  ): Promise<void> => {
    if (packageName === name) found.push(path);
    const nested = join(path, "node_modules");
    if (await isRealDirectory(nested)) await visit(nested);
  };
  if (await isRealDirectory(nodeModules)) await visit(nodeModules);
  return found.sort();
}

async function listDirectory(directory: string): Promise<readonly string[]> {
  try {
    return (await readdir(directory)).sort();
  } catch (error: unknown) {
    if (isEnoent(error) || isNotDirectory(error)) return [];
    throw error;
  }
}

async function isRealDirectory(path: string): Promise<boolean> {
  try {
    const stats = await lstat(path);
    return stats.isDirectory();
  } catch (error: unknown) {
    if (isEnoent(error)) return false;
    throw error;
  }
}

function isNotDirectory(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOTDIR";
}

interface PackageFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256?: string;
  readonly link?: string;
}

interface PartSize {
  readonly part: string;
  readonly bytes: number;
  readonly children?: readonly {
    readonly part: string;
    readonly bytes: number;
  }[];
}

/**
 * `semantscript package`: write a self-contained deployable directory (the
 * compiled output without the IR bundle, the production dependencies pruned
 * to the target platform's native bindings, the current artifact release
 * only, any --include files and a manifest of every file's digest), print
 * its size by part, and check it against a named deployment limit.
 */
export async function packageCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values, positionals } = parseArgs({
    args: [...args],
    options: {
      project: { type: "string" },
      dist: { type: "string" },
      artifact: { type: "string" },
      out: { type: "string" },
      include: { type: "string", multiple: true },
      target: { type: "string" },
      "max-bytes": { type: "string" },
      platform: { type: "string" },
      arch: { type: "string" },
      force: { type: "boolean" },
      json: { type: "boolean" },
    },
    allowPositionals: true,
  });
  if (positionals.length > 0) {
    throw new CliUsageError(
      `package takes no positional arguments, got ${positionals.join(" ")}`,
    );
  }
  const scalar: OptionValues = { ...values, include: undefined };
  const target = resolveTarget(
    stringOption(scalar, "target"),
    stringOption(scalar, "max-bytes"),
  );
  const project = resolve(io.cwd, stringOption(scalar, "project") ?? ".");
  const dist = resolve(project, stringOption(scalar, "dist") ?? "dist");
  const artifactOption =
    stringOption(scalar, "artifact") ??
    io.env[ARTIFACT_ENVIRONMENT_VARIABLE] ??
    undefined;
  const artifactRoot =
    artifactOption !== undefined && artifactOption.length > 0
      ? resolve(io.cwd, artifactOption)
      : join(project, ".semantscript", "artifact");
  const out = resolve(
    io.cwd,
    stringOption(scalar, "out") ?? join(project, DEFAULT_PACKAGE_PATH),
  );
  const platform = stringOption(scalar, "platform") ?? process.platform;
  const arch = stringOption(scalar, "arch") ?? process.arch;
  const includes = values.include ?? [];

  // The compiled output keeps its path relative to the project, so the
  // deployed package.json's main and scripts still name real files.
  const distPath = relative(project, dist).split(sep).join("/");
  if (!isInside(project, dist)) {
    throw new CliUsageError(
      `--dist ${dist} must be a directory inside the project ${project}, so the deployed package.json's entry points still resolve`,
    );
  }
  const distFirst = distPath.split("/")[0] ?? "";
  if (RESERVED_INCLUDES.has(distFirst)) {
    throw new CliUsageError(
      `--dist ${dist}: ${distFirst} is written by package itself`,
    );
  }
  const packageJson = await readProjectPackage(project);
  if (!(await isRealDirectory(dist))) {
    throw new PackageError(
      "PACKAGE_NO_DIST",
      `${dist} is not a directory: build the project first (or pass --dist)`,
    );
  }
  for (const guarded of [project, dist, artifactRoot]) {
    if (guarded === out || isInside(out, guarded)) {
      throw new CliUsageError(
        `--out ${out} would replace ${guarded}; choose a directory of its own`,
      );
    }
  }
  if (isInside(dist, out) || isInside(artifactRoot, out)) {
    throw new CliUsageError(
      `--out ${out} is inside the compiled output or the artifact; choose a directory of its own`,
    );
  }
  const reserved = new Set([...RESERVED_INCLUDES, distFirst]);
  const includeSources: IncludeSource[] = [];
  for (const spec of includes) {
    includeSources.push(await checkInclude(project, out, reserved, spec));
  }
  await checkOut(out, values.force === true);

  const pointer = await readPointer(artifactRoot);
  if (pointer === undefined) {
    throw new PackageError(
      "PACKAGE_NO_RELEASE",
      `${join(artifactRoot, "current.json")} does not exist: train the project first (or pass --artifact)`,
    );
  }
  const release = (await scanReleases(artifactRoot)).find(
    (entry) => entry.digest === pointer.manifestSha256,
  );
  if (release === undefined) {
    throw new PackageError(
      "PACKAGE_NO_RELEASE",
      `${join(artifactRoot, "current.json")} names ${pointer.release}, which is not on disk`,
    );
  }
  await verifyRelease(release);
  const summary = release.summary as ArtifactSummary;
  const manifest = release.manifest as Readonly<Record<string, unknown>>;

  const staging = join(
    dirname(out),
    `.package-staging-${randomBytes(6).toString("hex")}`,
  );
  await mkdir(staging, { recursive: true });
  try {
    // Compiled output without the IR bundle, which holds the prompt text.
    await cp(dist, join(staging, distPath), {
      recursive: true,
      verbatimSymlinks: true,
      filter: (source) => !source.endsWith(`${sep}${BUNDLE_FILE_NAME}`),
    });
    const artifactOut = join(staging, ".semantscript", "artifact");
    await mkdir(join(artifactOut, "releases"), { recursive: true });
    await cp(release.directory, join(artifactOut, "releases", release.name), {
      recursive: true,
    });
    await writeFile(
      join(artifactOut, "current.json"),
      pointerBytes(release.digest),
    );
    const included: string[] = [];
    for (const include of includeSources) {
      included.push(await copyInclude(staging, include));
    }

    const install = await installDependencies(
      project,
      packageJson,
      staging,
      io,
    );
    const pruned = await pruneNativeBindings(
      join(staging, "node_modules"),
      platform,
      arch,
    );
    const loadCheck =
      platform === process.platform && arch === process.arch
        ? await checkBindingsLoad(staging)
        : `not run: the bundle targets ${platform}/${arch} and this machine is ${process.platform}/${process.arch}`;

    const files = await listFiles(staging);
    const resources = resourceRoles(manifest, `releases/${release.name}`);
    const parts = partSizes(files, resources, included, distPath);
    const filesBytes = files.reduce((total, file) => total + file.bytes, 0);
    const encoderBytes = files
      .filter((file) => resources.get(file.path) === "encoder")
      .reduce((total, file) => total + file.bytes, 0);
    const document = {
      kind: PACKAGE_KIND,
      version: 1,
      createdAt: new Date().toISOString(),
      release: {
        release: `releases/${release.name}`,
        manifestSha256: release.digest,
        application: {
          id: summary.applicationId,
          version: summary.applicationVersion,
        },
        createdAt: summary.createdAt,
      },
      platform,
      arch,
      node: process.version,
      install,
      prunedBindings: pruned,
      parts,
      filesBytes,
      files,
    };
    const manifestText = `${JSON.stringify(document, null, 2)}\n`;
    await writeFile(join(staging, PACKAGE_MANIFEST), manifestText);
    const manifestBytes = Buffer.byteLength(manifestText);
    const totalBytes = filesBytes + manifestBytes;

    await rm(out, { recursive: true, force: true });
    await rename(staging, out);

    const quantized = listResources(manifest).some(
      (resource) =>
        resource["role"] === "encoder" &&
        typeof resource["onnx"] === "object" &&
        resource["onnx"] !== null &&
        (resource["onnx"] as Record<string, unknown>)["precision"] !==
          undefined &&
        (resource["onnx"] as Record<string, unknown>)["precision"] !==
          "float32",
    );
    const routing = depthRouting(manifest);
    const depthRouted = routing.routed;
    const fullEncoderPaths = new Set(
      listResources(manifest)
        .filter(
          (resource) =>
            resource["role"] === "encoder" &&
            typeof resource["ref"] === "string" &&
            routing.fullEncoderRefs.includes(resource["ref"]) &&
            typeof resource["path"] === "string",
        )
        .map(
          (resource) =>
            `.semantscript/artifact/releases/${release.name}/${resource["path"] as string}`,
        ),
    );
    const fullEncoderBytes = files
      .filter((file) => fullEncoderPaths.has(file.path))
      .reduce((total, file) => total + file.bytes, 0);
    const over = target !== undefined && totalBytes > target.bytes;
    const levers =
      target !== undefined && over
        ? packageLevers({
            totalBytes,
            encoderBytes,
            limitBytes: target.bytes,
            depthRouted,
            quantized,
            fullEncoderBytes,
            fullDepthFunctions: routing.fullDepthFunctions,
          })
        : [];

    if (values.json === true) {
      io.stdout(
        `${JSON.stringify(
          {
            kind: `${PACKAGE_KIND}.report`,
            version: 1,
            out,
            release: document.release,
            platform,
            arch,
            parts,
            manifestBytes,
            totalBytes,
            prunedBindings: {
              files: pruned.removed.length,
              bytes: pruned.bytes,
            },
            bindingsLoad: loadCheck,
            target:
              target === undefined
                ? null
                : {
                    ...target,
                    fits: !over,
                    overBytes: over ? totalBytes - target.bytes : 0,
                  },
            encoder: {
              bytes: encoderBytes,
              depthRouted,
              routing: !depthRouted
                ? "none"
                : fullEncoderBytes > 0
                  ? "mixed"
                  : "full",
              fullDepthEncoderBytes: fullEncoderBytes,
              quantized,
            },
            levers,
          },
          null,
          2,
        )}\n`,
      );
    } else {
      io.stdout(
        renderReport({
          out,
          summary,
          releaseName: release.name,
          platform,
          arch,
          parts,
          manifestBytes,
          totalBytes,
          pruned,
          loadCheck,
          install,
        }),
      );
      if (target !== undefined) {
        io.stdout(renderTarget(target, totalBytes, levers));
      }
    }
    if (over) {
      io.stderr(
        `PACKAGE_OVER_TARGET: the bundle is ${formatBytes(totalBytes)} (${String(totalBytes)} bytes), over ${target.name}'s ${formatBytes(target.bytes)}; it is written to ${out} anyway\n`,
      );
      return 1;
    }
    return 0;
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

function resolveTarget(
  name: string | undefined,
  maxBytes: string | undefined,
): PackageTarget | undefined {
  if (name !== undefined && maxBytes !== undefined) {
    throw new CliUsageError("give --target or --max-bytes, not both");
  }
  if (maxBytes !== undefined) {
    if (!/^[1-9][0-9]*$/u.test(maxBytes)) {
      throw new CliUsageError(
        `--max-bytes must be a positive whole number of bytes, got ${maxBytes}`,
      );
    }
    return {
      name: "max-bytes",
      bytes: Number(maxBytes),
      description: "the --max-bytes limit",
    };
  }
  if (name === undefined) return undefined;
  const target = Object.hasOwn(PACKAGE_TARGETS, name)
    ? PACKAGE_TARGETS[name]
    : undefined;
  if (target === undefined) {
    throw new CliUsageError(
      `unknown --target ${name}: expected ${Object.keys(PACKAGE_TARGETS).join(", ")} (or --max-bytes <n>)`,
    );
  }
  return target;
}

async function readProjectPackage(
  project: string,
): Promise<Readonly<Record<string, unknown>>> {
  const file = join(project, "package.json");
  let text: string;
  try {
    text = await readFile(file, "utf8");
  } catch (error: unknown) {
    if (isEnoent(error)) {
      throw new PackageError(
        "PACKAGE_NO_PROJECT",
        `${file} does not exist: run package from the application's directory (or pass --project)`,
      );
    }
    throw error;
  }
  try {
    return objectOf(JSON.parse(text) as unknown, "package.json");
  } catch (error: unknown) {
    throw new PackageError(
      "PACKAGE_NO_PROJECT",
      `${file} is not a JSON object: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

/** `out` must be absent, empty, or (with --force) a previous package. */
async function checkOut(out: string, force: boolean): Promise<void> {
  let names: readonly string[];
  try {
    const stats = await lstat(out);
    if (!stats.isDirectory()) {
      throw new PackageError(
        "PACKAGE_OUT_EXISTS",
        `${out} exists and is not a directory`,
      );
    }
    names = await readdir(out);
  } catch (error: unknown) {
    if (isEnoent(error)) return;
    throw error;
  }
  if (names.length === 0) return;
  if (!names.includes(PACKAGE_MANIFEST)) {
    throw new PackageError(
      "PACKAGE_OUT_EXISTS",
      `${out} is not empty and holds no ${PACKAGE_MANIFEST}; package never replaces a directory it did not write`,
    );
  }
  if (!force) {
    throw new PackageError(
      "PACKAGE_OUT_EXISTS",
      `${out} holds an earlier package; pass --force to replace it`,
    );
  }
}

function isInside(parent: string, child: string): boolean {
  const path = relative(parent, child);
  return path.length > 0 && !path.startsWith("..") && !isAbsolute(path);
}

const RESERVED_INCLUDES = new Set([
  "node_modules",
  ".semantscript",
  "package.json",
  PACKAGE_MANIFEST,
]);

interface IncludeSource {
  readonly source: string;
  /** Project-relative POSIX path, which is also its path in the bundle. */
  readonly path: string;
}

/**
 * Validate an --include before anything is written: it must name an
 * existing file or directory inside the project, outside the paths package
 * writes itself (the compiled output, which is copied without its IR bundle,
 * node_modules, the artifact, package.json and the manifest), and it must
 * not contain --out.
 */
async function checkInclude(
  project: string,
  out: string,
  reserved: ReadonlySet<string>,
  spec: string,
): Promise<IncludeSource> {
  const source = resolve(project, spec);
  const path = relative(project, source);
  if (path.length === 0 || !isInside(project, source)) {
    throw new CliUsageError(
      `--include ${spec} must name a file or directory inside the project ${project}`,
    );
  }
  const posix = path.split(sep).join("/");
  const first = posix.split("/")[0] ?? "";
  if (reserved.has(first)) {
    throw new CliUsageError(
      `--include ${spec}: ${first} is written by package itself`,
    );
  }
  if (source === out || isInside(source, out)) {
    throw new CliUsageError(
      `--include ${spec} contains --out ${out}; write the package somewhere else`,
    );
  }
  try {
    await stat(source);
  } catch (error: unknown) {
    if (isEnoent(error)) {
      throw new PackageError(
        "PACKAGE_INCLUDE_MISSING",
        `--include ${spec}: ${source} does not exist`,
      );
    }
    throw error;
  }
  return { source, path: posix };
}

async function copyInclude(
  staging: string,
  include: IncludeSource,
): Promise<string> {
  const destination = join(staging, ...include.path.split("/"));
  await mkdir(dirname(destination), { recursive: true });
  await cp(include.source, destination, { recursive: true, dereference: true });
  return include.path;
}

interface InstallRecord {
  readonly command: string | null;
  readonly dependencies: number;
}

/**
 * Install the production dependencies into the staging directory: `npm ci
 * --omit=dev` from the project's lockfile when every dependency comes from a
 * registry, else `npm install --omit=dev --install-links` with each file:
 * dependency rewritten to its absolute path (npm packs it into a real copy,
 * so the bundle needs no repository layout and holds no symlinks).
 */
async function installDependencies(
  project: string,
  packageJson: Readonly<Record<string, unknown>>,
  staging: string,
  io: CliIo,
): Promise<InstallRecord> {
  const production: Record<string, string> = {};
  let local = false;
  for (const field of ["dependencies", "optionalDependencies"] as const) {
    const value = packageJson[field];
    if (value === undefined) continue;
    for (const [name, spec] of Object.entries(objectOf(value, field))) {
      if (typeof spec !== "string") continue;
      if (spec.startsWith("file:") || spec.startsWith("link:")) {
        local = true;
        const target = resolve(project, spec.replace(/^(?:file|link):/u, ""));
        production[name] = `file:${target}`;
      } else {
        production[name] = spec;
      }
    }
  }
  // The deployed package.json keeps "type" and the scripts; devDependencies go.
  const deployed: Record<string, unknown> = { ...packageJson };
  delete deployed["devDependencies"];
  const deployedText = `${JSON.stringify(deployed, null, 2)}\n`;
  const count = Object.keys(production).length;
  if (count === 0) {
    await writeFile(join(staging, "package.json"), deployedText);
    return { command: null, dependencies: 0 };
  }
  const installManifest = {
    name: typeof packageJson["name"] === "string" ? packageJson["name"] : "app",
    version:
      typeof packageJson["version"] === "string"
        ? packageJson["version"]
        : "0.0.0",
    private: true,
    dependencies: production,
    // npm ci checks the lockfile against these too, so they must stay.
    ...(packageJson["overrides"] === undefined
      ? {}
      : { overrides: packageJson["overrides"] }),
  };
  await writeFile(
    join(staging, "package.json"),
    `${JSON.stringify(installManifest, null, 2)}\n`,
  );
  const lockfile = join(project, "package-lock.json");
  const useCi = !local && (await exists(lockfile));
  const args = useCi
    ? ["ci", "--omit=dev"]
    : ["install", "--omit=dev", "--install-links", "--no-package-lock"];
  if (useCi) {
    await copyFile(lockfile, join(staging, "package-lock.json"));
  }
  args.push("--no-audit", "--no-fund", "--loglevel=error");
  const command = `npm ${args.join(" ")}`;
  await runNpm(args, staging, io);
  await rm(join(staging, "package-lock.json"), { force: true });
  // npm's hidden lockfile records the build machine's absolute paths.
  await rm(join(staging, "node_modules", ".package-lock.json"), {
    force: true,
  });
  await writeFile(join(staging, "package.json"), deployedText);
  return { command, dependencies: count };
}

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error: unknown) {
    if (isEnoent(error)) return false;
    throw error;
  }
}

function runNpm(
  args: readonly string[],
  cwd: string,
  io: CliIo,
): Promise<void> {
  const npm = io.env["SEMANTSCRIPT_NPM"] ?? "npm";
  return new Promise((resolvePromise, reject) => {
    const child = spawn(npm, [...args], {
      cwd,
      env: {
        ...io.env,
        // The runtime runs ONNX Runtime's CPU provider only.
        ONNXRUNTIME_NODE_INSTALL: "skip",
        npm_config_update_notifier: "false",
      },
      stdio: ["ignore", "pipe", "pipe"],
      shell: process.platform === "win32",
    });
    const output: string[] = [];
    child.stdout.on("data", (chunk: Buffer) => output.push(chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => output.push(chunk.toString()));
    child.on("error", (error) => {
      reject(
        new PackageError(
          "PACKAGE_INSTALL_FAILED",
          `${npm} did not start: ${error.message}`,
        ),
      );
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolvePromise();
        return;
      }
      // npm states the cause first and may follow it with its usage text.
      const lines = output.join("").trim().split("\n");
      const tail =
        lines.length <= 30
          ? lines.join("\n")
          : [...lines.slice(0, 20), "...", ...lines.slice(-8)].join("\n");
      reject(
        new PackageError(
          "PACKAGE_INSTALL_FAILED",
          `${npm} ${args.join(" ")} exited with ${String(code)} in ${cwd}\n${tail}`,
        ),
      );
    });
  });
}

/**
 * Load the pruned native bindings from the bundle in a child Node process,
 * so a prune that removed what the runtime needs fails here, not at deploy.
 */
async function checkBindingsLoad(bundle: string): Promise<string> {
  const present: string[] = [];
  for (const name of ["onnxruntime-node", "tokenizers"]) {
    if ((await findPackages(join(bundle, "node_modules"), name)).length > 0) {
      present.push(name);
    }
  }
  if (present.length === 0) return "not run: no native bindings installed";
  const script = [
    `const { createRequire } = require("node:module");`,
    `const load = createRequire(${JSON.stringify(pathToFileURL(join(bundle, "package.json")).href)});`,
    ...present.map((name) => `load(${JSON.stringify(name)});`),
  ].join("\n");
  await new Promise<void>((resolvePromise, reject) => {
    const child = spawn(process.execPath, ["-e", script], {
      cwd: bundle,
      stdio: ["ignore", "ignore", "pipe"],
    });
    const errors: string[] = [];
    child.stderr.on("data", (chunk: Buffer) => errors.push(chunk.toString()));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolvePromise();
        return;
      }
      reject(
        new PackageError(
          "PACKAGE_BINDING_LOAD_FAILED",
          `${present.join(" and ")} did not load from the pruned bundle: ${errors.join("").trim().split("\n").slice(0, 5).join("\n")}`,
        ),
      );
    });
  });
  return `${present.join(" and ")} loaded from the bundle`;
}

/** Every file under the bundle with its size and sha256 (symlinks by target). */
async function listFiles(root: string): Promise<readonly PackageFile[]> {
  const files: PackageFile[] = [];
  const visit = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      const path = join(directory, entry.name);
      const posix = relative(root, path).split(sep).join("/");
      if (entry.isSymbolicLink()) {
        files.push({ path: posix, bytes: 0, link: await readlink(path) });
      } else if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile()) {
        const stats = await lstat(path);
        files.push({
          path: posix,
          bytes: stats.size,
          sha256: await fileSha256(path),
        });
      }
    }
  };
  await visit(root);
  return files.sort((left, right) =>
    left.path < right.path ? -1 : left.path > right.path ? 1 : 0,
  );
}

function listResources(
  manifest: Readonly<Record<string, unknown>>,
): readonly Readonly<Record<string, unknown>>[] {
  const resources = manifest["resources"];
  return Array.isArray(resources)
    ? resources.filter(
        (value): value is Readonly<Record<string, unknown>> =>
          value !== null && typeof value === "object",
      )
    : [];
}

function listFunctions(
  manifest: Readonly<Record<string, unknown>>,
): readonly Readonly<Record<string, unknown>>[] {
  const functions = manifest["functions"];
  return Array.isArray(functions)
    ? functions.filter(
        (value): value is Readonly<Record<string, unknown>> =>
          value !== null && typeof value === "object",
      )
    : [];
}

/** Bundle path -> resource role (encoder, adapter, head, tokenizer) for the release's files. */
function resourceRoles(
  manifest: Readonly<Record<string, unknown>>,
  release: string,
): ReadonlyMap<string, string> {
  const roles = new Map<string, string>();
  for (const resource of listResources(manifest)) {
    const path = resource["path"];
    const role = resource["role"];
    if (typeof path === "string" && typeof role === "string") {
      roles.set(`.semantscript/artifact/${release}/${path}`, role);
    }
  }
  return roles;
}

const ARTIFACT_PARTS = [
  ["encoder", "encoder"],
  ["adapter", "adapter"],
  ["head", "heads"],
  ["tokenizer", "tokenizer"],
] as const;

function partSizes(
  files: readonly PackageFile[],
  roles: ReadonlyMap<string, string>,
  included: readonly string[],
  distPath: string,
): readonly PartSize[] {
  const sum = (predicate: (file: PackageFile) => boolean): number =>
    files.filter(predicate).reduce((total, file) => total + file.bytes, 0);
  const under =
    (prefix: string) =>
    (file: PackageFile): boolean =>
      file.path === prefix || file.path.startsWith(`${prefix}/`);

  const byPackage = new Map<string, number>();
  for (const file of files) {
    if (!file.path.startsWith("node_modules/")) continue;
    const segments = file.path.split("/");
    const name =
      segments[1]?.startsWith("@") === true
        ? `${segments[1]}/${segments[2] ?? ""}`
        : (segments[1] ?? "");
    if (name.startsWith(".")) continue;
    byPackage.set(name, (byPackage.get(name) ?? 0) + file.bytes);
  }
  const largest = [...byPackage.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .map(([part, bytes]) => ({ part, bytes }));

  const artifact = files.filter(under(".semantscript/artifact"));
  const artifactChildren: { part: string; bytes: number }[] =
    ARTIFACT_PARTS.map(([role, part]) => ({
      part,
      bytes: artifact
        .filter((file) => roles.get(file.path) === role)
        .reduce((total, file) => total + file.bytes, 0),
    }));
  artifactChildren.push({
    part: "manifest and pointer",
    bytes: artifact
      .filter((file) => !roles.has(file.path))
      .reduce((total, file) => total + file.bytes, 0),
  });

  const includedSet = included.map((path) => under(path));
  return [
    { part: distPath, bytes: sum(under(distPath)) },
    {
      part: "node_modules",
      bytes: sum(under("node_modules")),
      children: largest,
    },
    {
      part: "artifact",
      bytes: artifact.reduce((total, file) => total + file.bytes, 0),
      children: artifactChildren,
    },
    {
      part: "included",
      bytes: sum((file) => includedSet.some((match) => match(file))),
      children: included.map((path) => ({
        part: path,
        bytes: sum(under(path)),
      })),
    },
    {
      part: "package.json",
      bytes: sum((file) => file.path === "package.json"),
    },
  ];
}

interface ReportInput {
  readonly out: string;
  readonly summary: ArtifactSummary;
  readonly releaseName: string;
  readonly platform: string;
  readonly arch: string;
  readonly parts: readonly PartSize[];
  readonly manifestBytes: number;
  readonly totalBytes: number;
  readonly pruned: PruneResult;
  readonly loadCheck: string;
  readonly install: InstallRecord;
}

function renderReport(input: ReportInput): string {
  const rows: string[][] = [];
  for (const part of input.parts) {
    rows.push([part.part, String(part.bytes), formatBytes(part.bytes)]);
    for (const child of part.children ?? []) {
      rows.push([
        `  ${child.part}`,
        String(child.bytes),
        formatBytes(child.bytes),
      ]);
    }
  }
  rows.push([
    PACKAGE_MANIFEST,
    String(input.manifestBytes),
    formatBytes(input.manifestBytes),
  ]);
  rows.push(["total", String(input.totalBytes), formatBytes(input.totalBytes)]);
  const summary = input.summary;
  return `${[
    `package  ${input.out}`,
    `release  releases/${input.releaseName} (${summary.applicationId}@${summary.applicationVersion}, created ${summary.createdAt})`,
    `platform ${input.platform}/${input.arch}, node ${process.version}`,
    `install  ${input.install.command ?? "no production dependencies"}`,
    `bindings ${input.loadCheck}; pruned ${String(input.pruned.removed.length)} other-platform or GPU-provider path${input.pruned.removed.length === 1 ? "" : "s"} (${formatBytes(input.pruned.bytes)})`,
    "",
    renderTable(["part", "bytes", "size"], rows).trimEnd(),
  ].join("\n")}\n`;
}

function renderTarget(
  target: PackageTarget,
  totalBytes: number,
  levers: readonly PackageLever[],
): string {
  const head = `\ntarget   ${target.name}: ${formatBytes(target.bytes)} (${String(target.bytes)} bytes), ${target.description}`;
  if (totalBytes <= target.bytes) {
    return `${head}\n         fits, ${formatBytes(target.bytes - totalBytes)} to spare\n`;
  }
  const rows = levers.map((lever) => [
    lever.label,
    lever.projectedBytes === null
      ? `encoder <= ${formatBytes(lever.encoderBudgetBytes ?? 0)}`
      : `~${formatBytes(lever.projectedBytes)}`,
    lever.fits
      ? lever.actionable
        ? "fits"
        : "fits (measurement only)"
      : "does not fit",
  ]);
  return `${[
    head,
    `         over by ${formatBytes(totalBytes - target.bytes)}; levers (estimates scaled from ModernBERT-base measurements, docs/deploy.md):`,
    "",
    renderTable(["lever", "bundle", "target"], rows).trimEnd(),
    "",
    ...levers.map((lever) => `${lever.label}: ${lever.how}`),
  ].join("\n")}\n`;
}
