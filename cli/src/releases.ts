import { createHash, randomBytes } from "node:crypto";
import { createReadStream, type Dirent } from "node:fs";
import { lstat, readdir, readFile, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { resolveArtifactRoot } from "./defaults.js";
import {
  CliUsageError,
  listOf,
  numberOf,
  objectOf,
  stringOf,
  type CliIo,
} from "./io.js";
import {
  isEnoent,
  readPointer,
  ReleaseError,
  summarizeManifest,
  writePointer,
  type ArtifactPointer,
  type ArtifactSummary,
  type ManifestFunctionSummary,
} from "./manifest.js";
import {
  formatRatio,
  renderTable,
  VERIFICATION_HEADERS,
  verificationCells,
} from "./table.js";

export const RELEASES_KIND = "semantscript.releases";

const RELEASE_NAME = /^sha256-([a-f0-9]{64})$/u;
const PORTABLE_PATH =
  /^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*(?:\/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)*$/u;
const PRUNING_PREFIX = ".pruning-";
const DURATION = /^([0-9]+)([dhm])$/u;
const DURATION_UNIT_MS: Readonly<Record<string, number>> = {
  d: 86_400_000,
  h: 3_600_000,
  m: 60_000,
};

/** One `releases/sha256-<digest>` directory, read without loading its models. */
export interface ReleaseEntry {
  readonly name: string;
  readonly digest: string;
  readonly directory: string;
  readonly summary?: ArtifactSummary;
  readonly createdAtMs?: number;
  readonly manifest?: Readonly<Record<string, unknown>>;
  /** Why the release is unusable: not a directory, a manifest that does not hash to its name, … */
  readonly error?: string;
}

interface ReleaseTotals {
  readonly passed: number;
  readonly total: number;
  readonly minAccuracy: number | null;
  readonly maxEce: number | null;
  readonly constraintViolations: number;
}

/**
 * `semantscript releases`: list the immutable releases under the artifact
 * root, inspect one, point `current.json` at another (rollback or promote),
 * and prune old ones. No subcommand means `list`.
 */
export async function releasesCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const at = subcommandIndex(args);
  if (at === undefined) {
    return listReleases(args, io);
  }
  const first = args[at] as string;
  const rest = [...args.slice(0, at), ...args.slice(at + 1)];
  switch (first) {
    case "list":
      return listReleases(rest, io);
    case "show":
      return showRelease(rest, io);
    case "rollback":
      return switchRelease(rest, io, "rollback");
    case "promote":
      return switchRelease(rest, io, "promote");
    case "prune":
      return pruneReleases(rest, io);
    default:
      throw new CliUsageError(
        `unknown releases subcommand ${first}: expected list, show, rollback, promote or prune`,
      );
  }
}

/** Options of any releases subcommand that take a value. */
const VALUE_OPTIONS = new Set(["--artifact", "--keep", "--older-than"]);

/**
 * The position of the subcommand, which may follow flags
 * (`releases --artifact <root> show <release>`); undefined means `list`.
 */
function subcommandIndex(args: readonly string[]): number | undefined {
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index] as string;
    if (arg === "--") return undefined;
    if (!arg.startsWith("-")) return index;
    if (VALUE_OPTIONS.has(arg)) index += 1;
  }
  return undefined;
}

async function listReleases(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: { artifact: { type: "string" }, json: { type: "boolean" } },
    allowPositionals: false,
  });
  const root = resolveArtifactRoot(values, io);
  const { pointer, pointerError } = await pointerState(root);
  const entries = await scanReleases(root);
  const currentDigest = pointer?.manifestSha256;

  if (values.json === true) {
    io.stdout(
      `${JSON.stringify(
        {
          kind: RELEASES_KIND,
          version: 1,
          root,
          current: pointer ?? null,
          pointerError: pointerError ?? null,
          releases: entries.map((entry) =>
            releaseJson(entry, entry.digest === currentDigest),
          ),
        },
        null,
        2,
      )}\n`,
    );
    return 0;
  }

  const lines = [
    `artifact ${root}`,
    `current  ${currentLine(pointer, pointerError)}`,
  ];
  if (entries.length === 0) {
    lines.push(`no releases under ${join(root, "releases")}`);
    io.stdout(`${lines.join("\n")}\n`);
    return 0;
  }
  const rows = entries.map((entry) => {
    const marker = entry.digest === currentDigest ? "*" : "";
    const summary = entry.summary;
    if (summary === undefined || entry.error !== undefined) {
      return [
        marker,
        "-",
        shortDigest(entry.digest),
        "-",
        "invalid",
        "-",
        "-",
        "-",
      ];
    }
    const totals = releaseTotals(summary.functions);
    return [
      marker,
      summary.createdAt,
      shortDigest(entry.digest),
      `${summary.applicationId}@${summary.applicationVersion}`,
      `${String(totals.passed)}/${String(totals.total)} passed`,
      totals.minAccuracy === null ? "-" : formatRatio(totals.minAccuracy),
      totals.maxEce === null ? "-" : formatRatio(totals.maxEce),
      String(totals.constraintViolations),
    ];
  });
  lines.push(
    renderTable(
      [
        "",
        "created",
        "release",
        "application",
        "functions",
        "min accuracy",
        "max ece",
        "violations",
      ],
      rows,
    ).trimEnd(),
  );
  for (const entry of entries) {
    if (entry.error !== undefined) {
      lines.push(`${entry.name}: invalid: ${entry.error}`);
    }
  }
  if (
    currentDigest !== undefined &&
    !entries.some((entry) => entry.digest === currentDigest)
  ) {
    lines.push(
      `current.json names releases/sha256-${currentDigest}, which is not on disk`,
    );
  }
  io.stdout(`${lines.join("\n")}\n`);
  return 0;
}

async function showRelease(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values, positionals } = parseArgs({
    args: [...args],
    options: { artifact: { type: "string" }, json: { type: "boolean" } },
    allowPositionals: true,
  });
  const [spec, ...extra] = positionals;
  if (spec === undefined || extra.length > 0) {
    throw new CliUsageError("releases show needs exactly one release");
  }
  const root = resolveArtifactRoot(values, io);
  const { pointer } = await pointerState(root);
  const entry = resolveRelease(await scanReleases(root), spec);
  const current = entry.digest === pointer?.manifestSha256;
  if (entry.summary === undefined || entry.error !== undefined) {
    throw new ReleaseError(
      "RELEASE_INVALID",
      `${entry.name}: ${entry.error ?? "unreadable"}`,
    );
  }
  if (values.json === true) {
    io.stdout(`${JSON.stringify(releaseJson(entry, current), null, 2)}\n`);
    return 0;
  }
  const summary = entry.summary;
  io.stdout(
    `${[
      `release      releases/${entry.name}${current ? " (current)" : ""}`,
      `created      ${summary.createdAt}`,
      `application  ${summary.applicationId}@${summary.applicationVersion}`,
      renderTable(
        VERIFICATION_HEADERS,
        summary.functions.map(verificationCells),
      ).trimEnd(),
    ].join("\n")}\n`,
  );
  return 0;
}

/**
 * `rollback [<release>]` and `promote <release>`: verify the target release
 * the way the runtime will, then rewrite `current.json` atomically. Without a
 * name, rollback targets the newest release created before the current one.
 * A running process that loaded the root with `watch: true` swaps to it.
 */
async function switchRelease(
  args: readonly string[],
  io: CliIo,
  verb: "rollback" | "promote",
): Promise<number> {
  const { values, positionals } = parseArgs({
    args: [...args],
    options: {
      artifact: { type: "string" },
      json: { type: "boolean" },
      "dry-run": { type: "boolean" },
    },
    allowPositionals: true,
  });
  const [spec, ...extra] = positionals;
  if (extra.length > 0) {
    throw new CliUsageError(`releases ${verb} takes at most one release`);
  }
  if (verb === "promote" && spec === undefined) {
    throw new CliUsageError("releases promote needs the release to promote");
  }
  const root = resolveArtifactRoot(values, io);
  const { pointer, pointerError } = await pointerState(root);
  if (spec === undefined && pointer === undefined) {
    throw new ReleaseError(
      "POINTER_INVALID",
      pointerError === undefined
        ? `${join(root, "current.json")} does not exist; name the release to switch to`
        : `${pointerError}; name the release to switch to, which also rewrites current.json`,
    );
  }
  const entries = await scanReleases(root);
  const target =
    spec === undefined
      ? previousRelease(entries, pointer?.manifestSha256 ?? "")
      : resolveRelease(entries, spec);
  if (target.digest === pointer?.manifestSha256) {
    throw new ReleaseError(
      "RELEASE_ALREADY_CURRENT",
      `releases/${target.name} is already current`,
    );
  }
  await verifyRelease(target);
  const summary = target.summary as ArtifactSummary;
  const dryRun = values["dry-run"] === true;
  if (!dryRun) {
    await writePointer(root, target.digest);
  }
  if (values.json === true) {
    io.stdout(
      `${JSON.stringify(
        {
          kind: `${RELEASES_KIND}.${verb}`,
          version: 1,
          root,
          from: pointer?.manifestSha256 ?? null,
          to: target.digest,
          createdAt: summary.createdAt,
          application: {
            id: summary.applicationId,
            version: summary.applicationVersion,
          },
          dryRun,
        },
        null,
        2,
      )}\n`,
    );
    return 0;
  }
  const from = pointer === undefined ? "(no valid pointer)" : pointer.release;
  io.stdout(
    `${[
      `${dryRun ? "would point" : "pointed"} ${join(root, "current.json")}`,
      `  from ${from}`,
      `  to   releases/${target.name} (${summary.applicationId}@${summary.applicationVersion}, created ${summary.createdAt})`,
      dryRun
        ? "dry run: current.json is unchanged"
        : "processes that loaded this root with watch: true reload it now; others load it on their next start",
    ].join("\n")}\n`,
  );
  return 0;
}

/**
 * `prune --keep <n> | --older-than <duration>`: remove releases that are
 * neither current nor kept. The pointer is re-read before every removal, a
 * release is renamed out of `releases/sha256-*` before it is deleted (so a
 * half-deleted directory never looks like a release), and invalid or
 * in-progress (`.staging-*`) directories are left alone.
 */
async function pruneReleases(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: {
      artifact: { type: "string" },
      keep: { type: "string" },
      "older-than": { type: "string" },
      "dry-run": { type: "boolean" },
      json: { type: "boolean" },
    },
    allowPositionals: false,
  });
  const keep = values.keep === undefined ? undefined : parseKeep(values.keep);
  const olderThanMs =
    values["older-than"] === undefined
      ? undefined
      : parseDuration(values["older-than"]);
  if (keep === undefined && olderThanMs === undefined) {
    throw new CliUsageError(
      "releases prune needs --keep <n>, --older-than <duration> or both",
    );
  }
  const dryRun = values["dry-run"] === true;
  const root = resolveArtifactRoot(values, io);
  const pointer = await readPointer(root);
  if (pointer === undefined) {
    throw new ReleaseError(
      "POINTER_INVALID",
      `${join(root, "current.json")} does not exist; prune never guesses which release is current`,
    );
  }
  const entries = await scanReleases(root);
  const valid = entries.filter((entry) => entry.error === undefined);
  if (!valid.some((entry) => entry.digest === pointer.manifestSha256)) {
    throw new ReleaseError(
      "POINTER_INVALID",
      `${join(root, "current.json")} names ${pointer.release}, which is missing or invalid; prune never guesses which release is current (repair the pointer with releases rollback <release> first)`,
    );
  }
  const newest = new Set(
    keep === undefined ? [] : valid.slice(0, keep).map((entry) => entry.digest),
  );
  const cutoff =
    olderThanMs === undefined ? undefined : Date.now() - olderThanMs;
  const candidates = valid.filter(
    (entry) =>
      entry.digest !== pointer.manifestSha256 &&
      (keep === undefined || !newest.has(entry.digest)) &&
      (cutoff === undefined || (entry.createdAtMs ?? Infinity) < cutoff),
  );

  const removed: { entry: ReleaseEntry; bytes: number }[] = [];
  const skipped: string[] = [];
  for (const entry of candidates) {
    const bytes = await treeSize(entry.directory);
    if (!dryRun) {
      // A train or rollback may have made this release current meanwhile.
      const now = await readPointer(root);
      if (now === undefined) {
        throw new ReleaseError(
          "POINTER_INVALID",
          `${join(root, "current.json")} disappeared during prune; stopped`,
        );
      }
      if (now.manifestSha256 === entry.digest) {
        skipped.push(entry.name);
        continue;
      }
      const doomed = join(
        root,
        "releases",
        `${PRUNING_PREFIX}${randomBytes(8).toString("hex")}`,
      );
      await rename(entry.directory, doomed);
      // A rollback may have pointed at it between the read and the rename.
      let after: ArtifactPointer | undefined;
      try {
        after = await readPointer(root);
      } catch (error: unknown) {
        await rename(doomed, entry.directory);
        throw error;
      }
      if (after === undefined || after.manifestSha256 === entry.digest) {
        await rename(doomed, entry.directory);
        skipped.push(entry.name);
        continue;
      }
      await rm(doomed, { recursive: true, force: true });
    }
    removed.push({ entry, bytes });
  }
  const leftovers = await pruningLeftovers(root);
  let leftoverBytes = 0;
  for (const leftover of leftovers) {
    leftoverBytes += await treeSize(leftover);
    if (!dryRun) await rm(leftover, { recursive: true, force: true });
  }
  const freed =
    removed.reduce((total, item) => total + item.bytes, 0) + leftoverBytes;
  const removedNames = new Set(removed.map((item) => item.entry.name));
  const kept = valid.filter((entry) => !removedNames.has(entry.name));
  const invalid = entries.filter((entry) => entry.error !== undefined);

  if (values.json === true) {
    io.stdout(
      `${JSON.stringify(
        {
          kind: `${RELEASES_KIND}.prune`,
          version: 1,
          root,
          current: pointer,
          dryRun,
          removed: removed.map((item) => ({
            manifestSha256: item.entry.digest,
            createdAt: item.entry.summary?.createdAt ?? null,
            bytes: item.bytes,
          })),
          kept: kept.map((entry) => entry.digest),
          skippedBecameCurrent: skipped,
          invalid: invalid.map((entry) => ({
            manifestSha256: entry.digest,
            error: entry.error,
          })),
          leftoversRemoved: leftovers.length,
          bytesFreed: freed,
        },
        null,
        2,
      )}\n`,
    );
    return 0;
  }
  const lines = [`artifact ${root}`, `current  ${pointer.release}`];
  if (removed.length === 0) {
    lines.push("no release matches; nothing removed");
  } else {
    lines.push(
      renderTable(
        [dryRun ? "would remove" : "removed", "created", "size"],
        removed.map((item) => [
          shortDigest(item.entry.digest),
          item.entry.summary?.createdAt ?? "-",
          formatBytes(item.bytes),
        ]),
      ).trimEnd(),
    );
  }
  for (const name of skipped) {
    lines.push(`${name}: became current during prune; kept`);
  }
  for (const entry of invalid) {
    lines.push(`${entry.name}: invalid, left in place: ${entry.error ?? ""}`);
  }
  if (leftovers.length > 0) {
    lines.push(
      `${dryRun ? "would remove" : "removed"} ${String(leftovers.length)} leftover ${PRUNING_PREFIX}* director${leftovers.length === 1 ? "y" : "ies"} from an interrupted prune`,
    );
  }
  lines.push(
    `kept ${String(kept.length)} release${kept.length === 1 ? "" : "s"}; ${dryRun ? "would free" : "freed"} ${formatBytes(freed)}`,
  );
  io.stdout(`${lines.join("\n")}\n`);
  return 0;
}

async function pointerState(
  root: string,
): Promise<{ pointer?: ArtifactPointer; pointerError?: string }> {
  try {
    const pointer = await readPointer(root);
    return pointer === undefined ? {} : { pointer };
  } catch (error: unknown) {
    if (error instanceof ReleaseError) {
      const prefix = `${error.code}: `;
      return {
        pointerError: error.message.startsWith(prefix)
          ? error.message.slice(prefix.length)
          : error.message,
      };
    }
    throw error;
  }
}

/**
 * Every `releases/sha256-<digest>` entry, newest first by the manifest's
 * `build.createdAt` (then by digest), with the invalid ones last. Staging and
 * pruning directories are not releases and are skipped.
 */
export async function scanReleases(
  root: string,
): Promise<readonly ReleaseEntry[]> {
  const directory = join(root, "releases");
  let names: Dirent[];
  try {
    names = await readdir(directory, { withFileTypes: true });
  } catch (error: unknown) {
    if (isEnoent(error)) return [];
    throw error;
  }
  const entries = await Promise.all(
    names
      .map((dirent) => dirent.name)
      .filter((name) => RELEASE_NAME.test(name))
      .map((name) => inspectRelease(root, name)),
  );
  return [...entries].sort(compareNewestFirst);
}

async function inspectRelease(
  root: string,
  name: string,
): Promise<ReleaseEntry> {
  const digest = name.slice("sha256-".length);
  const directory = join(root, "releases", name);
  const base = { name, digest, directory };
  try {
    const stats = await lstat(directory);
    if (stats.isSymbolicLink() || !stats.isDirectory()) {
      return { ...base, error: "not a non-symlink directory" };
    }
    const manifestPath = join(directory, "manifest.json");
    const manifestStats = await lstat(manifestPath);
    if (manifestStats.isSymbolicLink() || !manifestStats.isFile()) {
      return {
        ...base,
        error: "manifest.json is not a non-symlink regular file",
      };
    }
    const bytes = await readFile(manifestPath);
    const actual = createHash("sha256").update(bytes).digest("hex");
    if (actual !== digest) {
      return {
        ...base,
        error: `manifest.json hashes to ${actual}, not to the release name`,
      };
    }
    const manifest = objectOf(
      JSON.parse(bytes.toString("utf8")) as unknown,
      "manifest",
    );
    const summary = summarizeManifest(
      root,
      `releases/${name}`,
      digest,
      manifest,
    );
    const createdAtMs = Date.parse(summary.createdAt);
    if (Number.isNaN(createdAtMs)) {
      return {
        ...base,
        summary,
        error: `manifest.build.createdAt ${summary.createdAt} is not a date`,
      };
    }
    return { ...base, summary, manifest, createdAtMs };
  } catch (error: unknown) {
    return {
      ...base,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function compareNewestFirst(left: ReleaseEntry, right: ReleaseEntry): number {
  const leftValid = left.error === undefined;
  const rightValid = right.error === undefined;
  if (leftValid !== rightValid) return leftValid ? -1 : 1;
  if (!leftValid) {
    return left.name < right.name ? -1 : left.name > right.name ? 1 : 0;
  }
  const byDate = (right.createdAtMs ?? 0) - (left.createdAtMs ?? 0);
  if (byDate !== 0) return byDate;
  return right.digest < left.digest ? -1 : right.digest > left.digest ? 1 : 0;
}

/** A full digest, `sha256-<digest>`, `releases/sha256-<digest>` or a unique prefix of at least 7 hex characters. */
export function resolveRelease(
  entries: readonly ReleaseEntry[],
  spec: string,
): ReleaseEntry {
  const hex = spec
    .replace(/\/+$/u, "")
    .replace(/^releases\//u, "")
    .replace(/^sha256-/u, "");
  if (!/^[a-f0-9]{7,64}$/u.test(hex)) {
    throw new CliUsageError(
      `${spec} is not a release: give its manifest digest (at least 7 lowercase hex characters), sha256-<digest> or releases/sha256-<digest>`,
    );
  }
  const matches = entries.filter((entry) => entry.digest.startsWith(hex));
  const [only, ...others] = matches;
  if (only === undefined) {
    throw new ReleaseError(
      "RELEASE_NOT_FOUND",
      `no release under releases/ matches ${spec}`,
    );
  }
  if (others.length > 0) {
    throw new ReleaseError(
      "RELEASE_AMBIGUOUS",
      `${spec} matches ${matches.map((entry) => entry.name).join(", ")}; give more of the digest`,
    );
  }
  return only;
}

function previousRelease(
  entries: readonly ReleaseEntry[],
  currentDigest: string,
): ReleaseEntry {
  const current = entries.find((entry) => entry.digest === currentDigest);
  if (current === undefined || current.error !== undefined) {
    throw new ReleaseError(
      "RELEASE_NO_PREVIOUS",
      `the current release sha256-${currentDigest} is missing or invalid, so there is no "previous" one; name the release to switch to`,
    );
  }
  const index = entries.indexOf(current);
  const previous = entries
    .slice(index + 1)
    .find((entry) => entry.error === undefined);
  if (previous === undefined) {
    throw new ReleaseError(
      "RELEASE_NO_PREVIOUS",
      `no valid release was created before the current one (${current.summary?.createdAt ?? "?"})`,
    );
  }
  return previous;
}

/**
 * Check a release the way the runtime will load it: the manifest hashes to the
 * directory name, every resource is a regular non-symlink file inside the
 * release with its recorded size and digest, every function passed
 * verification, and then the runtime's own loader checks (`checkSemaArtifact`
 * from `@semantscript/core` with its default options: manifest schema, runtime
 * and model ABI compatibility, tensor names and shapes, opsets and the
 * encoder-adapter-head chain) accept it. Only ONNX session start-up and the
 * application's fallback registrations are left unchecked.
 */
async function verifyRelease(entry: ReleaseEntry): Promise<void> {
  if (
    entry.error !== undefined ||
    entry.manifest === undefined ||
    entry.summary === undefined
  ) {
    throw new ReleaseError(
      "RELEASE_INTEGRITY",
      `${entry.name}: ${entry.error ?? "unreadable"}`,
    );
  }
  const resources = listOf(entry.manifest["resources"], "manifest.resources");
  for (const [index, value] of resources.entries()) {
    const path = `manifest.resources[${String(index)}]`;
    try {
      const resource = objectOf(value, path);
      const relativePath = stringOf(resource["path"], `${path}.path`);
      const byteLength = numberOf(resource["byteLength"], `${path}.byteLength`);
      const digest = stringOf(resource["sha256"], `${path}.sha256`);
      if (!PORTABLE_PATH.test(relativePath) || relativePath.length > 500) {
        throw new Error(
          `${relativePath} is not a portable path inside the release`,
        );
      }
      const segments = relativePath.split("/");
      let current = entry.directory;
      for (const [position, segment] of segments.entries()) {
        current = join(current, segment);
        const stats = await lstat(current);
        const last = position === segments.length - 1;
        if (
          stats.isSymbolicLink() ||
          (last ? !stats.isFile() : !stats.isDirectory())
        ) {
          throw new Error(
            `${relativePath} crosses a symlink or has the wrong file type`,
          );
        }
        if (last && stats.size !== byteLength) {
          throw new Error(
            `${relativePath} is ${String(stats.size)} bytes, the manifest records ${String(byteLength)}`,
          );
        }
      }
      const actual = await fileSha256(current);
      if (actual !== digest) {
        throw new Error(
          `${relativePath} hashes to ${actual}, the manifest records ${digest}`,
        );
      }
    } catch (error: unknown) {
      throw new ReleaseError(
        "RELEASE_INTEGRITY",
        `${entry.name}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  const unverified = entry.summary.functions.filter(
    (fn) => fn.status !== "passed",
  );
  if (unverified.length > 0) {
    throw new ReleaseError(
      "RELEASE_UNVERIFIED",
      `${entry.name}: ${unverified.map((fn) => `${fn.id} is ${fn.status}`).join(", ")}; the runtime refuses a release whose functions did not pass verification`,
    );
  }
  const { checkSemaArtifact } = await import("@semantscript/core");
  try {
    await checkSemaArtifact(entry.directory);
  } catch (error: unknown) {
    const code =
      typeof error === "object" && error !== null && "code" in error
        ? String(error.code)
        : undefined;
    // An ArtifactLoadError's detail leaves out its remedy, which names this
    // very command (releases rollback) for a corrupt release.
    const detail =
      typeof error === "object" &&
      error !== null &&
      "detail" in error &&
      typeof error.detail === "string"
        ? error.detail
        : error instanceof Error
          ? error.message
          : String(error);
    throw new ReleaseError(
      "RELEASE_REJECTED",
      `${entry.name}: the runtime refuses to load it (${code === undefined ? detail : `${code}: ${detail}`})`,
    );
  }
}

async function fileSha256(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) {
    hash.update(chunk as Uint8Array);
  }
  return hash.digest("hex");
}

async function treeSize(path: string): Promise<number> {
  const stats = await lstat(path);
  if (!stats.isDirectory()) return stats.isFile() ? stats.size : 0;
  let total = 0;
  for (const name of await readdir(path)) {
    total += await treeSize(join(path, name));
  }
  return total;
}

async function pruningLeftovers(root: string): Promise<readonly string[]> {
  const directory = join(root, "releases");
  try {
    return (await readdir(directory))
      .filter((name) => name.startsWith(PRUNING_PREFIX))
      .map((name) => join(directory, name));
  } catch (error: unknown) {
    if (isEnoent(error)) return [];
    throw error;
  }
}

function releaseJson(entry: ReleaseEntry, current: boolean): unknown {
  const summary = entry.summary;
  return {
    release: `releases/${entry.name}`,
    manifestSha256: entry.digest,
    current,
    valid: entry.error === undefined,
    error: entry.error ?? null,
    createdAt: summary?.createdAt ?? null,
    application:
      summary === undefined
        ? null
        : { id: summary.applicationId, version: summary.applicationVersion },
    verification:
      summary === undefined ? null : releaseTotals(summary.functions),
    functions: summary?.functions ?? null,
  };
}

function releaseTotals(
  functions: readonly ManifestFunctionSummary[],
): ReleaseTotals {
  return {
    passed: functions.filter((fn) => fn.status === "passed").length,
    total: functions.length,
    minAccuracy:
      functions.length === 0
        ? null
        : Math.min(...functions.map((fn) => fn.accuracy)),
    maxEce:
      functions.length === 0
        ? null
        : Math.max(...functions.map((fn) => fn.ece)),
    constraintViolations: functions.reduce(
      (total, fn) => total + fn.constraintViolations,
      0,
    ),
  };
}

function currentLine(
  pointer: ArtifactPointer | undefined,
  pointerError: string | undefined,
): string {
  if (pointer !== undefined) return pointer.release;
  return pointerError === undefined
    ? "none (no current.json)"
    : `invalid pointer: ${pointerError}`;
}

function shortDigest(digest: string): string {
  return digest.slice(0, 12);
}

function parseKeep(text: string): number {
  if (!/^[0-9]+$/u.test(text)) {
    throw new CliUsageError(`--keep must be a whole number, got ${text}`);
  }
  return Number(text);
}

function parseDuration(text: string): number {
  const match = DURATION.exec(text);
  const unit = match?.[2];
  if (match === null || unit === undefined) {
    throw new CliUsageError(
      `--older-than must be a whole number of days, hours or minutes such as 30d, 12h or 90m, got ${text}`,
    );
  }
  return Number(match[1]) * (DURATION_UNIT_MS[unit] ?? 0);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${String(bytes)} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit] ?? "TiB"}`;
}
