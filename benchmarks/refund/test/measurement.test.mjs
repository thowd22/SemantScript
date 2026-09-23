import assert from "node:assert/strict";
import test from "node:test";

import {
  createProcessRssSampler,
  nodeMonotonicClock,
} from "../dist/measurement.js";

test("Node clock is finite and monotonic across adjacent observations", () => {
  const first = nodeMonotonicClock.now();
  const second = nodeMonotonicClock.now();
  assert.equal(Number.isFinite(first), true);
  assert.ok(second >= first);
});

test("RSS sampler polls the Node process from an independent worker", async () => {
  const sampler = createProcessRssSampler({ intervalMs: 1 });
  assert.equal(sampler.scope, "client-only");
  await sampler.start(new globalThis.AbortController().signal);

  // Atomics.wait blocks this main thread. The RSS worker remains schedulable,
  // matching the production runtime's synchronous inference behavior.
  globalThis.Atomics.wait(
    new globalThis.Int32Array(new globalThis.SharedArrayBuffer(4)),
    0,
    0,
    20,
  );

  const peakBytes = await sampler.stop();
  assert.equal(Number.isSafeInteger(peakBytes), true);
  assert.ok(peakBytes > 0);
});

test("RSS sampler rejects invalid lifecycle, configuration, and pre-start abort", async () => {
  assert.throws(() => createProcessRssSampler({ intervalMs: 0 }), /intervalMs/u);
  const aborted = new globalThis.AbortController();
  aborted.abort();
  const abortSampler = createProcessRssSampler();
  await assert.rejects(abortSampler.start(aborted.signal), /aborted/u);

  const sampler = createProcessRssSampler();
  await assert.rejects(sampler.stop(), /not running/u);
  await sampler.start(new globalThis.AbortController().signal);
  await assert.rejects(
    sampler.start(new globalThis.AbortController().signal),
    /already running/u,
  );
  assert.ok((await sampler.stop()) > 0);
});
