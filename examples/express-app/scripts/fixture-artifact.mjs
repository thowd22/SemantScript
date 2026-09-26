// Writes a fixture artifact keyed to this app's compiled bundle, so the server,
// the tests and the Docker image run without a trained model. The fixture's
// tiny ONNX graphs (runtime/test/fixtures) answer the third support value of
// every function: "review" for decideRefund and "urgent" for triage. It proves
// the wiring (artifact load, canonical input, inference worker, routes), not
// the policy.
//
//   npm run build && npm run fixture-artifact            # .semantscript/artifact
//   node scripts/fixture-artifact.mjs <dir> [--force]    # another root
//   npm run fixture-artifact -- --quantizable [--cache-dir <dir>]
//
// An existing artifact (for example a trained release) is kept unless --force
// is given (`npm run fixture-artifact -- --force`). A non-empty directory that
// is not an artifact root is never touched.
//
// --quantizable uses an encoder with a constant MatMul (the plain fixture has
// none for dynamic quantization to convert), so `semantscript releases derive
// --int8` can derive an int8 release from it, and writes the records the
// derivation verifies on into the build cache beside the artifact
// (<artifact>/../cache, or --cache-dir <dir>): one training dataset per
// function with the bundle's gold examples and synthetic rows labelled with
// the fixture's own answer, teacher "fixture". The manifest records each
// dataset's digest as the function's trainingProvenance.datasetSha256, as a
// trained release does. The CI package job runs the derivation on it.
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  createFixtureArtifact,
  semanticSha,
} from "../../../runtime/test/fixtures/artifact.mjs";

const appRoot = dirname(dirname(fileURLToPath(import.meta.url)));

/**
 * Creates a fixture artifact at `root` for every function in the IR bundle.
 * With `options.cacheDir`, the encoder is the quantizable one and each
 * function's fixture training dataset is written to that build cache.
 */
export async function writeFixtureArtifact(root, ir, options = {}) {
  const functions = ir.functions;
  const datasets =
    options.cacheDir === undefined
      ? undefined
      : await writeFixtureDatasets(options.cacheDir, functions);
  return createFixtureArtifact(root, {
    ...(datasets === undefined
      ? {}
      : { encoderSource: "quantizable-encoder.onnx" }),
    extraFunctions: functions.slice(1).map((fn) => ({
      id: fn.id,
      headRef: `head.${fn.id.slice(3, 11)}.value`,
    })),
    transformManifest: (manifest) => {
      manifest.functions = manifest.functions.map((entry, index) => {
        const fn = functions[index];
        return {
          ...entry,
          id: fn.id,
          inputs: fn.inputs,
          inputSchemaSha256: semanticSha(fn.inputs),
          heads: [
            {
              ...entry.heads[0],
              type: { kind: "nominal-string", support: fn.output.head.support },
            },
          ],
          ...(datasets === undefined
            ? {}
            : {
                trainingProvenance: {
                  ...entry.trainingProvenance,
                  datasetSha256: datasets[index],
                },
              }),
        };
      });
    },
  });
}

/** Synthetic rows per function: enough that the gold misses keep the ECE low. */
const FIXTURE_SYNTHETIC_ROWS = 61;

/**
 * One `semantscript.training-dataset` per function under
 * `<cacheDir>/datasets/v1`: the bundle's gold examples plus synthetic rows
 * labelled with the fixture's answer (the third support value), teacher
 * "fixture". Returns each file's SHA-256, in function order.
 */
export async function writeFixtureDatasets(cacheDir, functions) {
  const digests = [];
  for (const fn of functions) {
    const answer = fn.output.head.support[2];
    const gold = fn.definition.examples.map((example) => ({
      inputs: example.inputs,
      origin: "gold",
      output: example.output,
    }));
    const synthetic = Array.from(
      { length: FIXTURE_SYNTHETIC_ROWS },
      (_, index) => ({
        inputs: Object.fromEntries(
          fn.inputs.map((input) => [
            input.name,
            sampleValue(input.type, index + input.index),
          ]),
        ),
        origin: "synthetic",
        output: answer,
      }),
    );
    const cases = [...gold, ...synthetic];
    const requestSha256 = sha256(`fixture-dataset:${fn.id}`);
    const payload = {
      cases,
      counts: {
        gold: gold.length,
        synthetic: synthetic.length,
        total: cases.length,
      },
      function: { id: fn.id, semanticSha256: fn.semanticSha256 },
      requestSha256,
      teacher: {
        configurationSha256: sha256("fixture-teacher"),
        model: "fixture-artifact",
        provider: "fixture",
      },
    };
    const bytes = `${JSON.stringify(
      {
        datasetVersion: 1,
        kind: "semantscript.training-dataset",
        payload,
        payloadSha256: sha256(JSON.stringify(payload)),
      },
      null,
      2,
    )}\n`;
    const directory = join(
      cacheDir,
      "datasets",
      "v1",
      requestSha256.slice(0, 2),
    );
    await mkdir(directory, { recursive: true });
    await writeFile(join(directory, `${requestSha256}.json`), bytes);
    digests.push(sha256(bytes));
  }
  return digests;
}

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

/** A deterministic value of an IR input type, varied by `seed`. */
function sampleValue(type, seed) {
  switch (type.kind) {
    case "number":
      return (seed * 37) % 1000;
    case "string":
      return `fixture text ${String(seed)}`;
    case "boolean":
      return seed % 2 === 0;
    case "null":
      return null;
    case "literal":
      return type.value;
    case "enum":
      return type.values[seed % type.values.length].value;
    case "union":
      return sampleValue(type.variants[seed % type.variants.length], seed);
    case "object":
      return Object.fromEntries(
        type.fields
          .filter((field) => !field.optional || seed % 2 === 0)
          .map((field, index) => [
            field.name,
            sampleValue(field.type, seed + index),
          ]),
      );
    case "array":
      return [];
    case "tuple":
      return type.items.map((item, index) => sampleValue(item, seed + index));
    default:
      throw new Error(
        `the fixture cannot sample an input of kind ${type.kind}`,
      );
  }
}

export async function readBundle() {
  return JSON.parse(
    await readFile(join(appRoot, "dist", "semantscript.ir.v1.json"), "utf8"),
  );
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const args = process.argv.slice(2);
  const force = args.includes("--force");
  const quantizable = args.includes("--quantizable");
  const cacheAt = args.indexOf("--cache-dir");
  const cacheOption = cacheAt >= 0 ? args[cacheAt + 1] : undefined;
  const target = resolve(
    args.find(
      (arg, index) =>
        !arg.startsWith("--") && (cacheAt < 0 || index !== cacheAt + 1),
    ) ?? join(appRoot, ".semantscript", "artifact"),
  );
  const cacheDir = quantizable
    ? resolve(cacheOption ?? join(dirname(target), "cache"))
    : undefined;
  const entries = existsSync(target) ? await readdir(target) : [];
  const isArtifact = entries.includes("current.json");
  if (entries.length > 0 && !isArtifact) {
    console.error(
      `${target} is not empty and holds no artifact (no current.json); refusing to replace it`,
    );
    process.exit(1);
  }
  if (isArtifact && !force) {
    console.error(
      `${target} already holds an artifact; run \`npm run fixture-artifact -- --force\` to replace it with the fixture`,
    );
    process.exit(1);
  }
  const bundlePath = join(appRoot, "dist", "semantscript.ir.v1.json");
  if (!existsSync(bundlePath)) {
    console.error(
      `${bundlePath} is missing; run \`npm run build\` in examples/express-app first`,
    );
    process.exit(1);
  }
  const ir = await readBundle();
  await rm(target, { recursive: true, force: true });
  await mkdir(target, { recursive: true });
  const { manifestSha256 } = await writeFixtureArtifact(target, ir, {
    cacheDir,
  });
  console.log(
    `fixture artifact for ${ir.functions.length} functions at ${target} (release sha256-${manifestSha256.slice(0, 12)})`,
  );
  if (cacheDir !== undefined) {
    console.log(
      `quantizable encoder; fixture training datasets in ${cacheDir} (semantscript releases derive --int8 verifies on them)`,
    );
  }
}
