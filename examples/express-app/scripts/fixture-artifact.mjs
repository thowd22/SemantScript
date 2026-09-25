// Writes a fixture artifact keyed to this app's compiled bundle, so the server,
// the tests and the Docker image run without a trained model. The fixture's
// tiny ONNX graphs (runtime/test/fixtures) answer the third support value of
// every function: "review" for decideRefund and "urgent" for triage. It proves
// the wiring (artifact load, canonical input, inference worker, routes), not
// the policy.
//
//   npm run build && npm run fixture-artifact            # .semantscript/artifact
//   node scripts/fixture-artifact.mjs <dir> [--force]    # another root
//
// An existing artifact (for example a trained release) is kept unless --force
// is given.
import { existsSync } from "node:fs";
import { mkdir, readFile, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  createFixtureArtifact,
  semanticSha,
} from "../../../runtime/test/fixtures/artifact.mjs";

const appRoot = dirname(dirname(fileURLToPath(import.meta.url)));

/** Creates a fixture artifact at `root` for every function in the IR bundle. */
export async function writeFixtureArtifact(root, ir) {
  const functions = ir.functions;
  return createFixtureArtifact(root, {
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
        };
      });
    },
  });
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
  const target = resolve(
    args.find((arg) => !arg.startsWith("--")) ??
      join(appRoot, ".semantscript", "artifact"),
  );
  if (existsSync(join(target, "current.json")) && !force) {
    console.error(
      `${target} already holds an artifact; pass --force to replace it with the fixture`,
    );
    process.exit(1);
  }
  const ir = await readBundle();
  await rm(target, { recursive: true, force: true });
  await mkdir(target, { recursive: true });
  const { manifestSha256 } = await writeFixtureArtifact(target, ir);
  console.log(
    `fixture artifact for ${ir.functions.length} functions at ${target} (release sha256-${manifestSha256.slice(0, 12)})`,
  );
}
