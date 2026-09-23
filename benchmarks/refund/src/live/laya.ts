import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  LAYA_CHECKPOINT_REVISION,
  LAYA_CODE_REVISION,
  type LayaChoiceRequest,
  type LayaChoiceResponse,
  type LayaRunner,
} from "../adapters/laya.js";
import { LiveTransportError, denseArray, plainRecord } from "./http.js";

export const LAYA_PUBLIC_PROBABILITY_TRANSFORM =
  "log(max(public_calibrated_probability_rounded_4dp, 1e-12))" as const;
export const MAXIMUM_LAYA_PROTOCOL_LINE_BYTES = 1_048_576 as const;
export const MAXIMUM_LAYA_STDERR_BYTES = 65_536 as const;
export const DEFAULT_LAYA_INITIALIZATION_TIMEOUT_MS = 180_000 as const;
// Every regular file in Hub snapshot dd079950600224fb459af2a0cb1d74e1e57ee9cf; the
// worker requires the on-disk file set to match this manifest exactly.
export const LAYA_CHECKPOINT_FILES_SHA256 = Object.freeze({
  ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
  "README.md": "096751822e1e868c4015c8cc8dee39949737d979c17f866505bd4b45f730f407",
  "encoder/config.json": "5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d",
  "model.safetensors": "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
  "rl_agent_config.json": "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8",
  "tokenizer/tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
  "tokenizer/tokenizer_config.json": "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1",
} as const);

export type LayaSpawnImplementation = typeof spawn;

export interface LiveLayaRunnerOptions {
  readonly sourcePath: string;
  readonly checkpointPath: string;
  readonly pythonExecutable?: string;
  readonly workerScriptPath?: string;
  readonly device: "cpu" | "cuda";
  readonly initializationTimeoutMs?: number;
  readonly spawnImplementation?: LayaSpawnImplementation;
  readonly environment?: Readonly<Record<string, string>>;
}

interface PendingRequest {
  readonly resolve: (value: Readonly<Record<string, unknown>>) => void;
  readonly reject: (error: LiveTransportError) => void;
}

export class LiveLayaRunner implements LayaRunner {
  readonly checkpointRevision = LAYA_CHECKPOINT_REVISION;
  readonly sourceRevision = LAYA_CODE_REVISION;
  readonly device: "cpu" | "cuda";
  readonly probabilityTransform = LAYA_PUBLIC_PROBABILITY_TRANSFORM;

  readonly #child: ChildProcessWithoutNullStreams;
  readonly #pending = new Map<number, PendingRequest>();
  #nextId = 1;
  #stdoutBuffer = Buffer.alloc(0);
  #stderrBytes = 0;
  #closed = false;
  #queue: Promise<void> = Promise.resolve();

  private constructor(
    child: ChildProcessWithoutNullStreams,
    device: "cpu" | "cuda",
  ) {
    this.#child = child;
    this.device = device;
    child.stdout.on("data", (chunk: Buffer | string) => {
      this.#consumeStdout(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    child.stderr.on("data", (chunk: Buffer | string) => {
      this.#stderrBytes += Buffer.byteLength(chunk);
      if (this.#stderrBytes > MAXIMUM_LAYA_STDERR_BYTES) {
        this.#fail(
          new LiveTransportError(
            "response-too-large",
            "Laya subprocess stderr exceeded its byte limit",
          ),
        );
      }
    });
    child.once("error", () => {
      this.#fail(
        new LiveTransportError("process-failure", "Laya subprocess could not be started"),
      );
    });
    child.once("exit", () => {
      if (!this.#closed) {
        this.#fail(
          new LiveTransportError("process-failure", "Laya subprocess exited unexpectedly"),
        );
      }
    });
  }

  static async create(options: LiveLayaRunnerOptions): Promise<LiveLayaRunner> {
    validateAbsolutePath(options.sourcePath, "sourcePath");
    validateAbsolutePath(options.checkpointPath, "checkpointPath");
    const checkpointFilesSha256 = validateHashManifest(LAYA_CHECKPOINT_FILES_SHA256);
    const initializationTimeoutMs = validateInitializationTimeout(
      options.initializationTimeoutMs ?? DEFAULT_LAYA_INITIALIZATION_TIMEOUT_MS,
    );
    const workerScriptPath =
      options.workerScriptPath ??
      fileURLToPath(new URL("../../live/laya_worker.py", import.meta.url));
    validateAbsolutePath(workerScriptPath, "workerScriptPath");
    const pythonExecutable = options.pythonExecutable ?? "python3";
    if (
      pythonExecutable.length === 0 ||
      pythonExecutable.length > 1_024 ||
      pythonExecutable.includes("\0")
    ) {
      throw new LiveTransportError("configuration", "pythonExecutable is invalid");
    }
    const spawnImplementation = options.spawnImplementation ?? spawn;
    const child = spawnImplementation(pythonExecutable, [workerScriptPath], {
      cwd: options.sourcePath,
      detached: process.platform !== "win32",
      env: buildWorkerEnvironment(options.environment),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    const runner = new LiveLayaRunner(child, options.device);
    try {
      const response = await runner.#withInitializationTimeout(
        runner.#send({
          action: "initialize",
          sourcePath: options.sourcePath,
          sourceRevision: LAYA_CODE_REVISION,
          checkpointPath: options.checkpointPath,
          checkpointRevision: LAYA_CHECKPOINT_REVISION,
          checkpointFilesSha256,
          device: options.device,
          probabilityTransform: LAYA_PUBLIC_PROBABILITY_TRANSFORM,
        }),
        initializationTimeoutMs,
      );
      if (
        response["sourceRevision"] !== LAYA_CODE_REVISION ||
        response["checkpointRevision"] !== LAYA_CHECKPOINT_REVISION ||
        response["device"] !== options.device ||
        response["probabilityTransform"] !== LAYA_PUBLIC_PROBABILITY_TRANSFORM
      ) {
        throw new LiveTransportError(
          "invalid-response",
          "Laya subprocess did not confirm the pinned runtime identity",
        );
      }
      return runner;
    } catch (error) {
      runner.#fail(
        error instanceof LiveTransportError
          ? error
          : new LiveTransportError("process-failure", "Laya subprocess initialization failed"),
      );
      throw error;
    }
  }

  async choose(
    request: LayaChoiceRequest,
    signal: AbortSignal,
  ): Promise<LayaChoiceResponse> {
    const operation = this.#queue.then(async () => this.#chooseNow(request, signal));
    this.#queue = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }

  close(): void {
    if (this.#closed) {
      return;
    }
    this.#fail(new LiveTransportError("process-failure", "Laya subprocess was closed"));
  }

  async #chooseNow(
    request: LayaChoiceRequest,
    signal: AbortSignal,
  ): Promise<LayaChoiceResponse> {
    if (signal.aborted) {
      throw new LiveTransportError("process-failure", "Laya subprocess request was aborted");
    }
    const onAbort = (): void => {
      this.#fail(
        new LiveTransportError("process-failure", "Laya subprocess request was aborted"),
      );
    };
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      const response = await this.#send({ action: "choose", request });
      const checkpointRevision = response["checkpointRevision"];
      const selectedIndex = response["selectedIndex"];
      const logitsValue = denseArray(response["logits"], "Laya subprocess logits");
      const logits = logitsValue.map((value) => {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new LiveTransportError(
            "invalid-response",
            "Laya subprocess logits must be finite numbers",
          );
        }
        return value;
      });
      if (typeof checkpointRevision !== "string") {
        throw new LiveTransportError(
          "invalid-response",
          "Laya subprocess checkpoint revision must be a string",
        );
      }
      if (!Number.isSafeInteger(selectedIndex) || (selectedIndex as number) < 0) {
        throw new LiveTransportError(
          "invalid-response",
          "Laya subprocess selected index must be a non-negative safe integer",
        );
      }
      return { checkpointRevision, selectedIndex: selectedIndex as number, logits };
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  #send(payload: Readonly<Record<string, unknown>>): Promise<Readonly<Record<string, unknown>>> {
    if (this.#closed) {
      return Promise.reject(
        new LiveTransportError("process-failure", "Laya subprocess is not running"),
      );
    }
    const id = this.#nextId;
    this.#nextId += 1;
    const encoded = Buffer.from(`${JSON.stringify({ id, ...payload })}\n`, "utf8");
    if (encoded.byteLength > MAXIMUM_LAYA_PROTOCOL_LINE_BYTES) {
      return Promise.reject(
        new LiveTransportError(
          "configuration",
          "Laya subprocess request exceeds its protocol byte limit",
        ),
      );
    }
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#child.stdin.write(encoded, (error) => {
        if (error !== null && error !== undefined) {
          this.#pending.delete(id);
          reject(
            new LiveTransportError(
              "process-failure",
              "Laya subprocess request could not be written",
            ),
          );
        }
      });
    });
  }

  #consumeStdout(chunk: Buffer): void {
    if (this.#closed) {
      return;
    }
    this.#stdoutBuffer = Buffer.concat([this.#stdoutBuffer, chunk]);
    if (this.#stdoutBuffer.byteLength > MAXIMUM_LAYA_PROTOCOL_LINE_BYTES) {
      this.#fail(
        new LiveTransportError(
          "response-too-large",
          "Laya subprocess response exceeded its protocol byte limit",
        ),
      );
      return;
    }
    let newline = this.#stdoutBuffer.indexOf(0x0a);
    while (newline >= 0) {
      const line = this.#stdoutBuffer.subarray(0, newline);
      this.#stdoutBuffer = this.#stdoutBuffer.subarray(newline + 1);
      this.#consumeLine(line);
      newline = this.#stdoutBuffer.indexOf(0x0a);
    }
  }

  #consumeLine(line: Buffer): void {
    if (this.#closed) {
      return;
    }
    let decoded: unknown;
    try {
      decoded = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(line)) as unknown;
    } catch {
      this.#fail(
        new LiveTransportError("invalid-response", "Laya subprocess emitted invalid JSON"),
      );
      return;
    }
    let response: Readonly<Record<string, unknown>>;
    try {
      response = plainRecord(decoded, "Laya subprocess");
    } catch (error) {
      this.#fail(
        error instanceof LiveTransportError
          ? error
          : new LiveTransportError("invalid-response", "Laya subprocess response is invalid"),
      );
      return;
    }
    const id = response["id"];
    if (!Number.isSafeInteger(id)) {
      this.#fail(
        new LiveTransportError("invalid-response", "Laya subprocess response id is invalid"),
      );
      return;
    }
    const pending = this.#pending.get(id as number);
    if (pending === undefined) {
      this.#fail(
        new LiveTransportError(
          "invalid-response",
          "Laya subprocess emitted an unexpected response id",
        ),
      );
      return;
    }
    this.#pending.delete(id as number);
    if (response["ok"] !== true) {
      pending.reject(
        new LiveTransportError("process-failure", "Laya subprocess rejected the request"),
      );
      return;
    }
    pending.resolve(response);
  }

  async #withInitializationTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    let timer: NodeJS.Timeout | undefined;
    const timeout = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => {
        reject(
          new LiveTransportError(
            "process-failure",
            `Laya subprocess initialization exceeded ${String(timeoutMs)}ms`,
          ),
        );
      }, timeoutMs);
    });
    try {
      return await Promise.race([promise, timeout]);
    } finally {
      if (timer !== undefined) {
        clearTimeout(timer);
      }
    }
  }

  #fail(error: LiveTransportError): void {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    killSubprocessGroup(this.#child);
    for (const pending of this.#pending.values()) {
      pending.reject(error);
    }
    this.#pending.clear();
  }
}

function killSubprocessGroup(child: ChildProcessWithoutNullStreams): void {
  if (process.platform !== "win32" && child.pid !== undefined) {
    try {
      process.kill(-child.pid, "SIGKILL");
      return;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") {
        child.kill("SIGKILL");
        return;
      }
    }
  }
  child.kill("SIGKILL");
}

export async function createLiveLayaRunner(
  options: LiveLayaRunnerOptions,
): Promise<LiveLayaRunner> {
  return LiveLayaRunner.create(options);
}

function validateHashManifest(
  value: Readonly<Record<string, string>>,
): Readonly<Record<string, string>> {
  if (
    Array.isArray(value) ||
    Object.getPrototypeOf(value) !== Object.prototype ||
    Object.getOwnPropertySymbols(value).length > 0
  ) {
    throw new LiveTransportError(
      "configuration",
      "checkpointFilesSha256 must be a plain object",
    );
  }
  const entries = Object.entries(value).sort(([left], [right]) => left.localeCompare(right));
  if (entries.length === 0 || entries.length > 256) {
    throw new LiveTransportError(
      "configuration",
      "checkpointFilesSha256 must contain between 1 and 256 files",
    );
  }
  const result: Record<string, string> = {};
  for (const [relativePath, digest] of entries) {
    if (
      relativePath.length === 0 ||
      relativePath.length > 1_024 ||
      relativePath.startsWith("/") ||
      relativePath.includes("\\") ||
      relativePath.split("/").some((part) => part === "" || part === "." || part === "..")
    ) {
      throw new LiveTransportError(
        "configuration",
        "checkpointFilesSha256 contains an invalid relative path",
      );
    }
    if (!/^[a-f0-9]{64}$/.test(digest)) {
      throw new LiveTransportError(
        "configuration",
        "checkpointFilesSha256 contains an invalid digest",
      );
    }
    result[relativePath] = digest;
  }
  for (const required of ["model.safetensors", "rl_agent_config.json"]) {
    if (!(required in result)) {
      throw new LiveTransportError(
        "configuration",
        `checkpointFilesSha256 must include ${required}`,
      );
    }
  }
  return Object.freeze(result);
}

function buildWorkerEnvironment(
  additions: Readonly<Record<string, string>> | undefined,
): NodeJS.ProcessEnv {
  const allowedNames = [
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PYTHONPATH",
    "ROCM_PATH",
    "HSA_ENABLE_DXG_DETECTION",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "TMPDIR",
  ] as const;
  const environment: NodeJS.ProcessEnv = {};
  for (const name of allowedNames) {
    const value = process.env[name];
    if (value !== undefined) {
      environment[name] = value;
    }
  }
  if (additions !== undefined) {
    for (const [name, value] of Object.entries(additions)) {
      if (/TOKEN|KEY|SECRET|PASSWORD/i.test(name) || name.includes("\0") || value.includes("\0")) {
        throw new LiveTransportError(
          "configuration",
          "Laya subprocess environment contains a forbidden entry",
        );
      }
      environment[name] = value;
    }
  }
  environment["HF_HUB_OFFLINE"] = "1";
  environment["TRANSFORMERS_OFFLINE"] = "1";
  environment["PYTHONDONTWRITEBYTECODE"] = "1";
  return environment;
}

function validateAbsolutePath(value: string, name: string): void {
  if (!value.startsWith("/") || value.length > 4_096 || value.includes("\0")) {
    throw new LiveTransportError("configuration", `${name} must be an absolute local path`);
  }
}

function validateInitializationTimeout(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > 600_000) {
    throw new LiveTransportError(
      "configuration",
      "initializationTimeoutMs must be a positive safe integer no greater than 600000",
    );
  }
  return value;
}
