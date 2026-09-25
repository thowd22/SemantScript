import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough, Writable } from "node:stream";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  LAYA_CHECKPOINT,
  LAYA_CHECKPOINT_REVISION,
  LAYA_CODE_REVISION,
  OLLAMA_QWEN_MANIFEST_SHA256,
  OLLAMA_QWEN_MODELS,
  REFUND_OUTPUT_SCHEMA,
} from "../dist/adapters/index.js";
import {
  ANTHROPIC_API_VERSION,
  LAYA_CHECKPOINT_FILES_SHA256,
  LAYA_PUBLIC_PROBABILITY_TRANSFORM,
  LiveTransportError,
  MAXIMUM_LAYA_STDERR_BYTES,
  createLiveAnthropicTransport,
  createLiveLayaRunner,
  createLiveOllamaTransport,
  createLiveOpenRouterAnthropicTransport,
} from "../dist/live/index.js";

const OLLAMA_REQUEST = Object.freeze({
  model: OLLAMA_QWEN_MODELS["ollama-1b"],
  system: "fixed system",
  prompt: "fixed prompt",
  stream: false,
  think: false,
  format: REFUND_OUTPUT_SCHEMA,
  options: Object.freeze({ temperature: 0, seed: 0, num_predict: 256 }),
});

const LAYA_REQUEST = Object.freeze({
  checkpoint: LAYA_CHECKPOINT,
  checkpointRevision: LAYA_CHECKPOINT_REVISION,
  codeRevision: LAYA_CODE_REVISION,
  questionType: "choice",
  question: "fixed refund input",
  options: Object.freeze(["approve", "deny", "review"]),
  applyCheckpointCalibration: true,
});
const LAYA_WORKER_PATH = fileURLToPath(
  new globalThis.URL("../live/laya_worker.py", import.meta.url),
);

function jsonResponse(value, init = {}) {
  return new globalThis.Response(JSON.stringify(value), {
    ...init,
    headers: { "content-type": "application/json", ...init.headers },
  });
}

function assertLiveError(code, forbidden) {
  return (error) => {
    assert.ok(error instanceof LiveTransportError);
    assert.equal(error.code, code);
    if (forbidden !== undefined) {
      assert.equal(String(error).includes(forbidden), false);
      assert.equal(String(error.stack).includes(forbidden), false);
      assert.equal(String(error.cause).includes(forbidden), false);
    }
    return true;
  };
}

test("Ollama verifies the full manifest digest immediately before generate", async () => {
  const calls = [];
  const transport = createLiveOllamaTransport({
    fetchImplementation: async (url, init) => {
      calls.push({ url: String(url), init });
      if (String(url).endsWith("/api/tags")) {
        return jsonResponse({
          models: [
            {
              name: OLLAMA_REQUEST.model,
              model: OLLAMA_REQUEST.model,
              digest: OLLAMA_QWEN_MANIFEST_SHA256["ollama-1b"],
            },
          ],
        });
      }
      return jsonResponse({
        model: OLLAMA_REQUEST.model,
        response:
          '{"decision":"approve","probabilities":{"approve":0.8,"deny":0.1,"review":0.1}}',
        done: true,
        done_reason: "stop",
      });
    },
  });
  const controller = new globalThis.AbortController();
  const result = await transport.generate(OLLAMA_REQUEST, controller.signal);

  assert.equal(result.model, OLLAMA_REQUEST.model);
  assert.deepEqual(
    calls.map(({ url }) => new globalThis.URL(url).pathname),
    ["/api/tags", "/api/generate"],
  );
  assert.equal(calls[0].init.method, "GET");
  assert.equal(calls[1].init.method, "POST");
  assert.deepEqual(JSON.parse(calls[1].init.body), OLLAMA_REQUEST);
  assert.strictEqual(calls[0].init.signal, controller.signal);
  assert.strictEqual(calls[1].init.signal, controller.signal);
});

test("Ollama captures server version and observed CPU/GPU model placement", async () => {
  const calls = [];
  const transport = createLiveOllamaTransport({
    fetchImplementation: async (url) => {
      const path = new globalThis.URL(url).pathname;
      calls.push(path);
      if (path === "/api/version") {
        return jsonResponse({ version: "0.34.3" });
      }
      return jsonResponse({
        models: [
          {
            name: OLLAMA_REQUEST.model,
            digest: OLLAMA_QWEN_MANIFEST_SHA256["ollama-1b"],
            size: 4096,
            size_vram: 3072,
          },
        ],
      });
    },
  });

  const backend = await transport.resolveExecutionBackend(
    OLLAMA_REQUEST.model,
    new globalThis.AbortController().signal,
  );
  assert.deepEqual(new Set(calls), new Set(["/api/version", "/api/ps"]));
  assert.deepEqual(backend, {
    kind: "ollama",
    serverVersion: "0.34.3",
    placement: "hybrid",
    modelTotalBytes: 4096,
    modelCpuBytes: 1024,
    modelGpuBytes: 3072,
  });

  const inconsistent = createLiveOllamaTransport({
    fetchImplementation: async (url) =>
      new globalThis.URL(url).pathname === "/api/version"
        ? jsonResponse({ version: "0.34.3" })
        : jsonResponse({
            models: [
              {
                name: OLLAMA_REQUEST.model,
                digest: OLLAMA_QWEN_MANIFEST_SHA256["ollama-1b"],
                size: 1024,
                size_vram: 2048,
              },
            ],
          }),
  });
  await assert.rejects(
    inconsistent.resolveExecutionBackend(
      OLLAMA_REQUEST.model,
      new globalThis.AbortController().signal,
    ),
    assertLiveError("invalid-response"),
  );
});

test("Ollama fails closed before inference on manifest drift or oversized metadata", async () => {
  let callCount = 0;
  const drifted = createLiveOllamaTransport({
    fetchImplementation: async () => {
      callCount += 1;
      return jsonResponse({
        models: [{ name: OLLAMA_REQUEST.model, digest: "0".repeat(64) }],
      });
    },
  });
  await assert.rejects(
    drifted.generate(OLLAMA_REQUEST, new globalThis.AbortController().signal),
    assertLiveError("invalid-response"),
  );
  assert.equal(callCount, 1);

  const oversized = createLiveOllamaTransport({
    fetchImplementation: async () =>
      new globalThis.Response("x", {
        headers: { "content-length": String(2 * 1024 * 1024) },
      }),
  });
  await assert.rejects(
    oversized.generate(OLLAMA_REQUEST, new globalThis.AbortController().signal),
    assertLiveError("response-too-large"),
  );
});

test("Anthropic maps the current Messages structured-output wire format exactly", async () => {
  const apiKey = "sk-ant-test-not-a-real-secret";
  let captured;
  const transport = createLiveAnthropicTransport({
    apiKey,
    fetchImplementation: async (url, init) => {
      captured = { url: String(url), init };
      return jsonResponse({
        id: "msg_test",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-5",
        stop_reason: "end_turn",
        content: [
          { type: "thinking", thinking: "not exposed to the adapter" },
          {
            type: "text",
            text: '{"decision":"review","probabilities":{"approve":0.1,"deny":0.2,"review":0.7}}',
          },
        ],
      });
    },
  });
  const request = {
    model: "claude-sonnet-5",
    maxTokens: 256,
    system: "fixed system",
    messages: [{ role: "user", content: "fixed input" }],
    outputConfig: {
      format: {
        type: "json_schema",
        name: "refund_decision",
        schema: REFUND_OUTPUT_SCHEMA,
      },
    },
  };
  const result = await transport.generate(
    request,
    new globalThis.AbortController().signal,
  );

  assert.equal(new globalThis.URL(captured.url).pathname, "/v1/messages");
  assert.equal(captured.init.headers["x-api-key"], apiKey);
  assert.equal(
    captured.init.headers["anthropic-version"],
    ANTHROPIC_API_VERSION,
  );
  const body = JSON.parse(captured.init.body);
  assert.deepEqual(Object.keys(body).sort(), [
    "max_tokens",
    "messages",
    "model",
    "output_config",
    "system",
  ]);
  assert.deepEqual(body.output_config, {
    format: { type: "json_schema", schema: REFUND_OUTPUT_SCHEMA },
  });
  assert.equal(Object.hasOwn(body.output_config.format, "name"), false);
  assert.equal(Object.hasOwn(body, "temperature"), false);
  assert.deepEqual(result, {
    model: "claude-sonnet-5",
    stopReason: "end_turn",
    output: {
      decision: "review",
      probabilities: { approve: 0.1, deny: 0.2, review: 0.7 },
    },
  });
});

test("Anthropic bounds errors and never copies API keys or provider bodies into errors", async () => {
  const apiKey = "sk-ant-do-not-leak-this-value";
  const statusFailure = createLiveAnthropicTransport({
    apiKey,
    fetchImplementation: async () =>
      jsonResponse({ error: { message: apiKey } }, { status: 401 }),
  });
  const request = {
    model: "claude-sonnet-5",
    maxTokens: 256,
    system: "system",
    messages: [{ role: "user", content: "input" }],
    outputConfig: {
      format: {
        type: "json_schema",
        name: "refund_decision",
        schema: REFUND_OUTPUT_SCHEMA,
      },
    },
  };
  await assert.rejects(
    statusFailure.generate(request, new globalThis.AbortController().signal),
    assertLiveError("http-status", apiKey),
  );

  const thrownFailure = createLiveAnthropicTransport({
    apiKey,
    fetchImplementation: async () => {
      throw new Error(apiKey);
    },
  });
  await assert.rejects(
    thrownFailure.generate(request, new globalThis.AbortController().signal),
    assertLiveError("invalid-response", apiKey),
  );
  assert.throws(
    () =>
      createLiveAnthropicTransport({
        apiKey,
        baseUrl: "https://example.invalid",
        fetchImplementation: async () => {
          throw new Error("must not run");
        },
      }),
    assertLiveError("configuration", apiKey),
  );
});

test("persistent Laya runner sends fixed pins and reuses one warmed process", async () => {
  const fake = fakeSpawn((message) => {
    if (message.action === "initialize") {
      return {
        id: message.id,
        ok: true,
        sourceRevision: LAYA_CODE_REVISION,
        checkpointRevision: LAYA_CHECKPOINT_REVISION,
        device: "cuda",
        probabilityTransform: LAYA_PUBLIC_PROBABILITY_TRANSFORM,
      };
    }
    return {
      id: message.id,
      ok: true,
      checkpointRevision: LAYA_CHECKPOINT_REVISION,
      selectedIndex: 1,
      logits: [Math.log(0.1), Math.log(0.7), Math.log(0.2)],
    };
  });
  const runner = await createLiveLayaRunner({
    sourcePath: "/pinned/laya-source",
    checkpointPath: `/cache/snapshots/${LAYA_CHECKPOINT_REVISION}`,
    device: "cuda",
    spawnImplementation: fake.spawn,
  });
  const first = await runner.choose(
    LAYA_REQUEST,
    new globalThis.AbortController().signal,
  );
  const second = await runner.choose(
    LAYA_REQUEST,
    new globalThis.AbortController().signal,
  );

  assert.equal(fake.calls.length, 1);
  assert.equal(fake.messages.length, 3);
  assert.equal(fake.messages[0].sourceRevision, LAYA_CODE_REVISION);
  assert.equal(fake.messages[0].checkpointRevision, LAYA_CHECKPOINT_REVISION);
  assert.deepEqual(
    fake.messages[0].checkpointFilesSha256,
    LAYA_CHECKPOINT_FILES_SHA256,
  );
  assert.equal(
    fake.messages[0].probabilityTransform,
    LAYA_PUBLIC_PROBABILITY_TRANSFORM,
  );
  assert.equal(fake.calls[0].options.env.HF_HUB_OFFLINE, "1");
  assert.equal(fake.calls[0].options.env.TRANSFORMERS_OFFLINE, "1");
  assert.equal(fake.calls[0].options.env.PYTHONDONTWRITEBYTECODE, "1");
  assert.equal(fake.calls[0].options.detached, process.platform !== "win32");
  assert.equal(first.selectedIndex, 1);
  assert.deepEqual(second, first);
  assert.equal(runner.probabilityTransform, LAYA_PUBLIC_PROBABILITY_TRANSFORM);
  runner.close();
  assert.equal(fake.killed, true);
});

test("Laya runner kills an aborted request and bounds subprocess stderr", async () => {
  const hanging = fakeSpawn((message) => {
    if (message.action === "initialize") {
      return {
        id: message.id,
        ok: true,
        sourceRevision: LAYA_CODE_REVISION,
        checkpointRevision: LAYA_CHECKPOINT_REVISION,
        device: "cpu",
        probabilityTransform: LAYA_PUBLIC_PROBABILITY_TRANSFORM,
      };
    }
    return undefined;
  });
  const runner = await createLiveLayaRunner({
    sourcePath: "/pinned/laya-source",
    checkpointPath: `/cache/snapshots/${LAYA_CHECKPOINT_REVISION}`,
    device: "cpu",
    spawnImplementation: hanging.spawn,
  });
  const controller = new globalThis.AbortController();
  const pending = runner.choose(LAYA_REQUEST, controller.signal);
  await new Promise((resolvePromise) =>
    globalThis.setImmediate(resolvePromise),
  );
  controller.abort();
  await assert.rejects(pending, assertLiveError("process-failure"));
  assert.equal(hanging.killed, true);

  const noisy = fakeSpawn((message) => ({
    id: message.id,
    ok: true,
    sourceRevision: LAYA_CODE_REVISION,
    checkpointRevision: LAYA_CHECKPOINT_REVISION,
    device: "cpu",
    probabilityTransform: LAYA_PUBLIC_PROBABILITY_TRANSFORM,
  }));
  const noisyRunner = await createLiveLayaRunner({
    sourcePath: "/pinned/laya-source",
    checkpointPath: `/cache/snapshots/${LAYA_CHECKPOINT_REVISION}`,
    device: "cpu",
    spawnImplementation: noisy.spawn,
  });
  noisy.child.stderr.write(Buffer.alloc(MAXIMUM_LAYA_STDERR_BYTES + 1));
  await assert.rejects(
    noisyRunner.choose(LAYA_REQUEST, new globalThis.AbortController().signal),
    assertLiveError("process-failure"),
  );
  assert.equal(noisy.killed, true);
});

test("Laya Python verifier accepts only an exact checkpoint file set and hashes", () => {
  const temporary = mkdtempSync(
    join(tmpdir(), "semantscript-laya-worker-test-"),
  );
  try {
    const snapshot = join(temporary, LAYA_CHECKPOINT_REVISION);
    mkdirSync(join(snapshot, "encoder"), { recursive: true });
    mkdirSync(join(snapshot, "tokenizer"), { recursive: true });
    const contents = {
      "encoder/config.json": "encoder",
      "model.safetensors": "weights",
      "rl_agent_config.json": "config",
      "tokenizer/tokenizer.json": "tokenizer",
    };
    const manifest = {};
    for (const [relativePath, content] of Object.entries(contents)) {
      writeFileSync(join(snapshot, relativePath), content);
      manifest[relativePath] = createHash("sha256")
        .update(content)
        .digest("hex");
    }
    const worker = LAYA_WORKER_PATH;
    const script = [
      "import importlib.util,json,sys",
      "spec=importlib.util.spec_from_file_location('worker',sys.argv[1])",
      "module=importlib.util.module_from_spec(spec)",
      "spec.loader.exec_module(module)",
      "module._verify_checkpoint(module.Path(sys.argv[2]),module.LAYA_CHECKPOINT_REVISION,json.loads(sys.argv[3]))",
    ].join(";");
    const accepted = spawnSync("python3", [
      "-B",
      "-c",
      script,
      worker,
      snapshot,
      JSON.stringify(manifest),
    ]);
    assert.equal(accepted.status, 0, accepted.stderr.toString());

    writeFileSync(join(snapshot, "model.safetensors"), "tampered");
    const rejected = spawnSync("python3", [
      "-B",
      "-c",
      script,
      worker,
      snapshot,
      JSON.stringify(manifest),
    ]);
    assert.notEqual(rejected.status, 0);
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
});

test("Laya worker pins the inspected public API and documents probability conversion", () => {
  const source = readFileSync(LAYA_WORKER_PATH, "utf8");
  assert.match(
    source,
    /laya\.load\(str\(checkpoint_path\), device=device, fast=False\)/,
  );
  assert.match(source, /agent\.predict\(question, questions\)/);
  assert.match(source, /not raw logits/);
  assert.match(source, /--untracked-files=all/);
});

function fakeSpawn(handler) {
  const calls = [];
  const messages = [];
  let killed = false;
  let child;
  const spawn = (command, args, options) => {
    calls.push({ command, args, options });
    child = new EventEmitter();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.stdin = new Writable({
      write(chunk, _encoding, callback) {
        try {
          const message = JSON.parse(Buffer.from(chunk).toString("utf8"));
          messages.push(message);
          const response = handler(message);
          if (response !== undefined) {
            globalThis.queueMicrotask(() => {
              child.stdout.write(`${JSON.stringify(response)}\n`);
            });
          }
          callback();
        } catch (error) {
          callback(error);
        }
      },
    });
    child.kill = () => {
      killed = true;
      globalThis.queueMicrotask(() => child.emit("exit", null, "SIGTERM"));
      return true;
    };
    return child;
  };
  return {
    calls,
    messages,
    spawn,
    get child() {
      return child;
    },
    get killed() {
      return killed;
    },
  };
}

test("OpenRouter route sends the pinned model with its route prefix as a bearer request and strips it from the response", async () => {
  const apiKey = "sk-or-test-not-a-real-secret";
  let captured;
  const transport = createLiveOpenRouterAnthropicTransport({
    apiKey,
    fetchImplementation: async (url, init) => {
      captured = { url: String(url), init };
      return jsonResponse({
        id: "gen-test",
        type: "message",
        role: "assistant",
        model: "anthropic/claude-sonnet-5",
        stop_reason: "end_turn",
        content: [
          {
            type: "text",
            text: '{"decision":"deny","probabilities":{"approve":0.1,"deny":0.8,"review":0.1}}',
          },
        ],
        usage: { input_tokens: 1, output_tokens: 1, cost: 0.000001 },
        provider: "Claude Platform on AWS",
      });
    },
  });
  assert.deepEqual(transport.executionBackend, {
    kind: "anthropic-api",
    apiVersion: ANTHROPIC_API_VERSION,
    endpoint: "https://openrouter.ai/api",
    placement: "provider-managed",
  });
  const result = await transport.generate(
    {
      model: "claude-sonnet-5",
      maxTokens: 256,
      system: "fixed system",
      messages: [{ role: "user", content: "fixed input" }],
      outputConfig: {
        format: {
          type: "json_schema",
          name: "refund_decision",
          schema: REFUND_OUTPUT_SCHEMA,
        },
      },
    },
    new globalThis.AbortController().signal,
  );
  const url = new globalThis.URL(captured.url);
  assert.equal(url.origin, "https://openrouter.ai");
  assert.equal(url.pathname, "/api/v1/messages");
  assert.equal(captured.init.headers.authorization, `Bearer ${apiKey}`);
  assert.equal(Object.hasOwn(captured.init.headers, "x-api-key"), false);
  assert.equal(
    captured.init.headers["anthropic-version"],
    ANTHROPIC_API_VERSION,
  );
  const body = JSON.parse(captured.init.body);
  assert.equal(body.model, "anthropic/claude-sonnet-5");
  assert.deepEqual(Object.keys(body).sort(), [
    "max_tokens",
    "messages",
    "model",
    "output_config",
    "system",
  ]);
  assert.deepEqual(body.output_config, {
    format: { type: "json_schema", schema: REFUND_OUTPUT_SCHEMA },
  });
  assert.deepEqual(result, {
    model: "claude-sonnet-5",
    stopReason: "end_turn",
    output: {
      decision: "deny",
      probabilities: { approve: 0.1, deny: 0.8, review: 0.1 },
    },
  });
});

test("OpenRouter route never copies the key into errors and needs a key", async () => {
  const apiKey = "sk-or-do-not-leak-this-value";
  const failing = createLiveOpenRouterAnthropicTransport({
    apiKey,
    fetchImplementation: async () =>
      jsonResponse({ error: { message: apiKey } }, { status: 401 }),
  });
  await assert.rejects(
    failing.generate(
      {
        model: "claude-sonnet-5",
        maxTokens: 256,
        system: "system",
        messages: [{ role: "user", content: "input" }],
        outputConfig: {
          format: {
            type: "json_schema",
            name: "refund_decision",
            schema: REFUND_OUTPUT_SCHEMA,
          },
        },
      },
      new globalThis.AbortController().signal,
    ),
    (error) =>
      error instanceof LiveTransportError && !error.message.includes(apiKey),
  );
  const previous = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    assert.throws(
      () => createLiveOpenRouterAnthropicTransport(),
      assertLiveError("configuration"),
    );
  } finally {
    if (previous !== undefined) process.env.OPENROUTER_API_KEY = previous;
  }
});
