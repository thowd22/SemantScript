import process from "node:process";
import { createInterface } from "node:readline/promises";

/** Where a command reads its environment and writes its output; tests inject their own. */
export interface CliIo {
  readonly cwd: string;
  readonly env: Readonly<Record<string, string | undefined>>;
  readonly stdout: (text: string) => void;
  readonly stderr: (text: string) => void;
  /** Ends long-running commands (`dev`); the process wires SIGINT to it. */
  readonly signal?: AbortSignal;
  /**
   * Asks the user one question and resolves to the answer; present only when stdin
   * is an interactive terminal, so tests and CI never block on a prompt.
   */
  readonly ask?: (question: string) => Promise<string>;
}

export function processIo(): CliIo {
  const controller = new AbortController();
  process.once("SIGINT", () => {
    controller.abort();
  });
  return {
    cwd: process.cwd(),
    env: process.env,
    signal: controller.signal,
    stdout: (text) => {
      process.stdout.write(text);
    },
    stderr: (text) => {
      process.stderr.write(text);
    },
    ...(process.stdin.isTTY
      ? {
          ask: async (question: string): Promise<string> => {
            const reader = createInterface({
              input: process.stdin,
              output: process.stdout,
            });
            try {
              return await reader.question(question);
            } finally {
              reader.close();
            }
          },
        }
      : {}),
  };
}

/** A command-line mistake: reported with the usage text and exit status 2. */
export class CliUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CliUsageError";
  }
}

export type OptionValues = Readonly<
  Record<string, string | boolean | undefined>
>;

export function requireString(values: OptionValues, name: string): string {
  const value = values[name];
  if (typeof value !== "string" || value.length === 0) {
    throw new CliUsageError(`--${name} is required`);
  }
  return value;
}

export function stringOption(
  values: OptionValues,
  name: string,
): string | undefined {
  const value = values[name];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

export function objectOf(
  value: unknown,
  path: string,
): Readonly<Record<string, unknown>> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
  return value as Readonly<Record<string, unknown>>;
}

export function listOf(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${path} must be an array`);
  }
  return value as readonly unknown[];
}

export function stringOf(value: unknown, path: string): string {
  if (typeof value !== "string") {
    throw new Error(`${path} must be a string`);
  }
  return value;
}

export function numberOf(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path} must be a finite number`);
  }
  return value;
}
