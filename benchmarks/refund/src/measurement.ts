import { performance } from "node:perf_hooks";
import { Worker } from "node:worker_threads";

import type { MonotonicClock, PeakMemorySampler } from "./runner.js";

export const nodeMonotonicClock: MonotonicClock = Object.freeze({
  now: (): number => performance.now(),
});

export interface ProcessRssSamplerOptions {
  readonly intervalMs?: number;
}

const RSS_WORKER_SOURCE = String.raw`
  const { parentPort, workerData } = require("node:worker_threads");
  let peakBytes = process.memoryUsage.rss();
  const sample = () => { peakBytes = Math.max(peakBytes, process.memoryUsage.rss()); };
  const timer = setInterval(sample, workerData.intervalMs);
  parentPort.on("message", (message) => {
    if (message === "stop") {
      sample();
      clearInterval(timer);
      parentPort.postMessage({ type: "stopped", peakBytes });
    }
  });
  parentPort.postMessage({ type: "ready" });
`;

/**
 * Samples RSS for this Node client process from an independent worker thread.
 * The worker continues polling while synchronous inference blocks the main event
 * loop. RSS includes all Node worker threads, but not child/server processes,
 * accelerator allocations, shared GPU memory, or VRAM.
 */
export function createProcessRssSampler(
  options: ProcessRssSamplerOptions = {},
): PeakMemorySampler {
  const intervalMs = options.intervalMs ?? 10;
  if (!Number.isSafeInteger(intervalMs) || intervalMs < 1 || intervalMs > 60_000) {
    throw new RangeError("intervalMs must be an integer from 1 through 60000");
  }

  let worker: Worker | undefined;
  let running = false;
  let workerFailure: Error | undefined;

  return Object.freeze({
    scope: "client-only",
    async start(signal: AbortSignal): Promise<void> {
      if (running) throw new Error("process RSS sampler is already running");
      if (signal.aborted) throw new Error("process RSS sampling was aborted before start");
      running = true;
      workerFailure = undefined;
      const nextWorker = new Worker(RSS_WORKER_SOURCE, {
        eval: true,
        workerData: { intervalMs },
      });
      worker = nextWorker;
      try {
        await new Promise<void>((resolveReady, rejectReady) => {
          const onMessage = (message: unknown): void => {
            if (isWorkerMessage(message, "ready")) {
              cleanup();
              resolveReady();
            }
          };
          const onError = (error: Error): void => {
            cleanup();
            rejectReady(error);
          };
          const onExit = (code: number): void => {
            cleanup();
            rejectReady(new Error(`process RSS sampler worker exited with code ${String(code)}`));
          };
          const cleanup = (): void => {
            nextWorker.off("message", onMessage);
            nextWorker.off("error", onError);
            nextWorker.off("exit", onExit);
          };
          nextWorker.on("message", onMessage);
          nextWorker.once("error", onError);
          nextWorker.once("exit", onExit);
        });
        nextWorker.once("error", (error) => {
          workerFailure = error;
        });
        nextWorker.once("exit", (code) => {
          if (running && code !== 0) {
            workerFailure = new Error(
              `process RSS sampler worker exited with code ${String(code)}`,
            );
          }
        });
      } catch (error) {
        running = false;
        worker = undefined;
        await nextWorker.terminate().catch(() => undefined);
        throw error;
      }
    },
    async stop(): Promise<number> {
      if (!running || worker === undefined) {
        throw new Error("process RSS sampler is not running");
      }
      const activeWorker = worker;
      try {
        if (workerFailure !== undefined) throw workerFailure;
        const peakBytes = await new Promise<number>((resolvePeak, rejectPeak) => {
          const onMessage = (message: unknown): void => {
            if (!isWorkerMessage(message, "stopped")) return;
            const value = message["peakBytes"];
            cleanup();
            if (!Number.isSafeInteger(value) || (value as number) < 0) {
              rejectPeak(new TypeError("process RSS sample must be a non-negative safe integer"));
              return;
            }
            resolvePeak(value as number);
          };
          const onError = (error: Error): void => {
            cleanup();
            rejectPeak(error);
          };
          const onExit = (code: number): void => {
            cleanup();
            rejectPeak(new Error(`process RSS sampler worker exited with code ${String(code)}`));
          };
          const cleanup = (): void => {
            clearTimeout(timer);
            activeWorker.off("message", onMessage);
            activeWorker.off("error", onError);
            activeWorker.off("exit", onExit);
          };
          const timer = setTimeout(() => {
            cleanup();
            rejectPeak(new Error("process RSS sampler worker did not stop"));
          }, 5_000);
          activeWorker.on("message", onMessage);
          activeWorker.once("error", onError);
          activeWorker.once("exit", onExit);
          activeWorker.postMessage("stop");
        });
        return peakBytes;
      } finally {
        running = false;
        worker = undefined;
        await activeWorker.terminate().catch(() => undefined);
      }
    },
  });
}

function isWorkerMessage(
  value: unknown,
  type: "ready" | "stopped",
): value is Readonly<Record<string, unknown>> {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as Readonly<Record<string, unknown>>)["type"] === type
  );
}
