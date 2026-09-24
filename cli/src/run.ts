import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { resolveArtifactRoot } from "./defaults.js";
import { CliUsageError, type CliIo } from "./io.js";

/**
 * `semantscript run`: load the artifact so compiled `__sema` calls resolve,
 * import the program module and optionally call one of its exports with JSON
 * input, printing the JSON result.
 */
export async function runCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values, positionals } = parseArgs({
    args: [...args],
    options: {
      artifact: { type: "string" },
      call: { type: "string" },
      input: { type: "string" },
      "input-file": { type: "string" },
    },
    allowPositionals: true,
  });
  const modulePath = positionals[0];
  if (modulePath === undefined || positionals.length !== 1) {
    throw new CliUsageError("run takes exactly one module path");
  }
  if (values.input !== undefined && values["input-file"] !== undefined) {
    throw new CliUsageError("pass either --input or --input-file, not both");
  }
  const root = resolveArtifactRoot(values, io);
  const callArguments = await resolveArguments(
    values.input,
    values["input-file"],
    io.cwd,
  );

  const { loadSemaArtifact } = await import("@semantscript/core");
  const handle = await loadSemaArtifact(root);
  try {
    const imported: unknown = await import(
      pathToFileURL(resolve(io.cwd, modulePath)).href
    );
    if (values.call === undefined) return 0;
    const namespace =
      imported !== null && typeof imported === "object"
        ? (imported as Readonly<Record<string, unknown>>)
        : {};
    const target = namespace[values.call];
    if (typeof target !== "function") {
      io.stderr(`${modulePath} has no function export named ${values.call}\n`);
      return 1;
    }
    const result: unknown = await Promise.resolve(
      (target as (...callArgs: unknown[]) => unknown)(...callArguments),
    );
    io.stdout(
      `${result === undefined ? "undefined" : JSON.stringify(result, null, 2)}\n`,
    );
    return 0;
  } finally {
    await handle.close();
  }
}

async function resolveArguments(
  inline: string | undefined,
  file: string | undefined,
  cwd: string,
): Promise<readonly unknown[]> {
  const text =
    file === undefined ? inline : await readFile(resolve(cwd, file), "utf8");
  if (text === undefined) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch (error: unknown) {
    throw new CliUsageError(
      `input must be JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return Array.isArray(parsed) ? (parsed as readonly unknown[]) : [parsed];
}
