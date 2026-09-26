import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";

/** One labelled training case and where its label came from. */
export interface TrainingCase {
  readonly inputs: unknown;
  readonly output: unknown;
  /** `gold`, `synthetic` or `adversarial`. */
  readonly origin: "gold" | "synthetic" | "adversarial";
  /** Who labelled it: the teacher for synthetic cases, the adversarial tag for adversarial ones. */
  readonly detail: string | null;
}

export interface TrainingCases {
  /** The dataset file's SHA-256 (what the manifest records as `datasetSha256`). */
  readonly datasetSha256: string | null;
  readonly datasetPath: string | null;
  /** True when this is the dataset the loaded release was trained on. */
  readonly release: boolean;
  readonly teacher: string | null;
  readonly adversarialPath: string | null;
  readonly cases: readonly TrainingCase[];
  /** Other cached datasets for the same function (other teachers, sizes or revisions). */
  readonly otherDatasets: number;
}

interface CachedDocument {
  readonly path: string;
  readonly sha256: string;
  readonly modified: number;
  readonly payload: Readonly<Record<string, unknown>>;
}

/**
 * The training cases a function was trained on, from the training cache
 * (`datasets/v1/**.json` and the `adversarial-datasets/v1/**.json` sidecars).
 * The release's dataset is the file whose SHA-256 equals the manifest's
 * `trainingProvenance.datasetSha256`; when it is gone (the cache was cleared
 * or training ran elsewhere) the newest dataset for the function id is used
 * and marked as not the release's. Undefined when the cache has none.
 */
export async function findTrainingCases(
  cacheDir: string,
  functionId: string,
  releaseDatasetSha256: string | null,
): Promise<TrainingCases | undefined> {
  const datasets = (
    await readDocuments(
      join(cacheDir, "datasets", "v1"),
      "semantscript.training-dataset",
    )
  ).filter((document) => functionIdOf(document.payload) === functionId);
  const releaseDataset = datasets.find(
    (document) => document.sha256 === releaseDatasetSha256,
  );
  const chosen =
    releaseDataset ??
    [...datasets].sort((left, right) => right.modified - left.modified)[0];
  const baseSha256 = chosen?.sha256 ?? releaseDatasetSha256;
  const sidecar =
    baseSha256 === null
      ? undefined
      : (
          await readDocuments(
            join(cacheDir, "adversarial-datasets", "v1"),
            "semantscript.adversarial-dataset",
          )
        ).find(
          (document) =>
            functionIdOf(document.payload) === functionId &&
            baseDatasetSha256(document.payload) === baseSha256,
        );
  if (chosen === undefined && sidecar === undefined) return undefined;

  const cases: TrainingCase[] = [];
  const teacher = chosen === undefined ? null : teacherOf(chosen.payload);
  for (const raw of listOf(chosen?.payload["cases"])) {
    const record = recordOf(raw);
    const origin = record["origin"] === "gold" ? "gold" : "synthetic";
    cases.push({
      inputs: record["inputs"],
      output: record["output"],
      origin,
      detail: origin === "synthetic" ? teacher : null,
    });
  }
  for (const raw of listOf(sidecar?.payload["cases"])) {
    const record = recordOf(raw);
    cases.push({
      inputs: record["inputs"],
      output: record["output"],
      origin: "adversarial",
      detail: typeof record["tag"] === "string" ? record["tag"] : null,
    });
  }
  return {
    datasetSha256: chosen?.sha256 ?? null,
    datasetPath: chosen?.path ?? null,
    release:
      releaseDataset !== undefined ||
      (chosen === undefined && releaseDatasetSha256 !== null),
    teacher,
    adversarialPath: sidecar?.path ?? null,
    cases,
    otherDatasets: datasets.length - (chosen === undefined ? 0 : 1),
  };
}

async function readDocuments(
  directory: string,
  kind: string,
): Promise<readonly CachedDocument[]> {
  const documents: CachedDocument[] = [];
  for (const shard of await entries(directory)) {
    const shardPath = join(directory, shard);
    for (const name of await entries(shardPath)) {
      if (!name.endsWith(".json")) continue;
      const path = join(shardPath, name);
      try {
        const bytes = await readFile(path);
        const document = recordOf(JSON.parse(bytes.toString("utf8")));
        if (document["kind"] !== kind) continue;
        documents.push({
          path,
          sha256: createHash("sha256").update(bytes).digest("hex"),
          modified: (await stat(path)).mtimeMs,
          payload: recordOf(document["payload"]),
        });
      } catch {
        // A torn or foreign file is not a training dataset; the trainer rebuilds it.
      }
    }
  }
  return documents;
}

async function entries(directory: string): Promise<readonly string[]> {
  try {
    return (await readdir(directory)).sort();
  } catch {
    return [];
  }
}

function functionIdOf(payload: Readonly<Record<string, unknown>>): unknown {
  return recordOf(payload["function"])["id"];
}

function baseDatasetSha256(
  payload: Readonly<Record<string, unknown>>,
): unknown {
  return recordOf(payload["baseDataset"])["datasetSha256"];
}

function teacherOf(payload: Readonly<Record<string, unknown>>): string | null {
  const teacher = recordOf(payload["teacher"]);
  const provider = teacher["provider"];
  const model = teacher["model"];
  if (typeof provider !== "string") return null;
  if (typeof model !== "string" || model.length === 0) return provider;
  return model.startsWith(`${provider}/`) ? model : `${provider}/${model}`;
}

function recordOf(value: unknown): Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : {};
}

function listOf(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? (value as readonly unknown[]) : [];
}
