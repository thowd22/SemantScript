import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { clearTimeout, setTimeout } from "node:timers";

import ts from "typescript";

import { planSemaCompilationSync } from "../dist/index.js";
import createPlugin, {
  PLUGIN_DIAGNOSTICS,
  SemaEditorState,
} from "../dist/ts-plugin.js";

const compilerRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = join(compilerRoot, "..");
const tsserverPath = join(
  repositoryRoot,
  "node_modules",
  "typescript",
  "lib",
  "tsserver.js",
);

const coreDeclarations = `
interface SemaExample<T> { readonly inputs: Readonly<Record<string, unknown>>; readonly output: T }
interface SemaOptions<T> { readonly examples?: readonly SemaExample<T>[]; readonly constraints?: readonly unknown[] }
interface ConfiguredSemaTag<T> { (strings: TemplateStringsArray, ...inputs: unknown[]): T }
interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: SemaOptions<T>): ConfiguredSemaTag<T>;
}
export declare const sema: SemaTag;
`;
const source = `import { sema } from "@semantscript/core";

export function triage(message: string): "urgent" | "routine" {
  return sema<"urgent" | "routine">\`PROMPT_TRIAGE \${message}\`;
}

export function approve(amount: number): boolean {
  return sema<boolean>({ examples: [{ inputs: { amount: 1 }, output: true }] })\`PROMPT_APPROVE \${amount}\`;
}
`;
const malformedSource = `import { sema } from "@semantscript/core";

export const broken = sema\`PROMPT_WITHOUT_TYPE\`;
`;

test("the language-service plugin shows artifact numbers on hover and inline guidance diagnostics", async (t) => {
  const fixture = await createFixture({ "src/app.sem.ts": source });
  t.after(fixture.dispose);
  const { service, host } = createService(fixture.root);
  const program = service.getProgram();
  const planned = planSemaCompilationSync(program, {
    projectRoot: fixture.root,
  });
  assert.ok(planned.ok);
  const [triageId, approveId] = planned.value.sites.map(
    (site) => site.functionId,
  );
  const appPath = join(fixture.root, "src", "app.sem.ts");

  // Before any training: every site is unverified.
  const plugin = createPlugin({ typescript: ts }).create(
    createInfo(service, host, fixture.root, {}),
  );
  const untrained = plugin
    .getSemanticDiagnostics(appPath)
    .filter((d) => d.source === "semantscript");
  assert.deepEqual(
    untrained.map((d) => d.code),
    [PLUGIN_DIAGNOSTICS.noArtifact, PLUGIN_DIAGNOSTICS.noArtifact],
  );
  assert.match(
    String(untrained[0].messageText),
    /no trained artifact at .*\.semantscript\/artifact; next: run semantscript train \(or semantscript dev\) to publish an artifact at .*; next: add a few examples/u,
  );
  assert.doesNotMatch(String(untrained[1].messageText), /no examples/u);
  assert.equal(untrained[0].category, ts.DiagnosticCategory.Warning);

  // A trained artifact: triage passed with weak accuracy, approve is not in it.
  await writeArtifact(fixture.root, [
    {
      id: triageId,
      accuracy: 0.91,
      ece: 0.02,
      pairConsistency: 0.97,
      attestedCases: 40,
      constraintViolations: 3,
    },
  ]);
  const trained = plugin
    .getSemanticDiagnostics(appPath)
    .filter((d) => d.source === "semantscript");
  assert.deepEqual(
    trained.map((d) => d.code),
    [
      PLUGIN_DIAGNOSTICS.accuracyBelowThreshold,
      PLUGIN_DIAGNOSTICS.notInArtifact,
    ],
  );
  assert.match(
    String(trained[0].messageText),
    /verified accuracy 0\.910 is below 0\.950; next: add a few examples/u,
  );
  assert.match(
    String(trained[1].messageText),
    /not in the latest artifact \(releases\/sha256-/u,
  );
  assert.match(
    String(trained[1].messageText),
    /; next: run semantscript train; the build cache retrains only the expressions that changed/u,
  );

  const position = source.indexOf('sema<"urgent"');
  const hover = plugin.getQuickInfoAtPosition(appPath, position + 2);
  assert.equal(hover.textSpan.start, position);
  const text = hover.documentation.map((part) => part.text).join("");
  assert.match(
    text,
    new RegExp(
      `sema function ${triageId.slice(0, 11)}… → "urgent" \\| "routine"`,
      "u",
    ),
  );
  assert.match(
    text,
    /inputs 1 · examples 0 · constraints 0 · confidence threshold none/u,
  );
  assert.match(
    text,
    /passed · accuracy 0\.910 · ECE 0\.020 · pair consistency 0\.970 · attested cases 40 · constraint violations 3/u,
  );
  assert.match(text, /⚠ verified accuracy 0\.910 is below 0\.950/u);
  const approveHover = plugin.getQuickInfoAtPosition(
    appPath,
    source.indexOf("sema<boolean>") + 1,
  );
  assert.match(
    approveHover.documentation[0].text,
    /examples 1 .*\n.*not in it \(unverified; run semantscript train\)/u,
  );
  // Outside a site the ordinary TypeScript hover is returned.
  const ordinary = plugin.getQuickInfoAtPosition(
    appPath,
    source.indexOf("triage"),
  );
  assert.equal(ordinary?.kind, ts.ScriptElementKind.functionElement);
  assert.notEqual(
    ordinary?.documentation?.[0]?.text?.includes("sema function"),
    true,
  );

  // Thresholds are configurable; a good release clears the diagnostics.
  await writeArtifact(fixture.root, [
    {
      id: triageId,
      accuracy: 0.99,
      ece: 0.15,
      pairConsistency: 0.99,
      attestedCases: 80,
      constraintViolations: 0,
    },
    {
      id: approveId,
      accuracy: 0.99,
      ece: 0.01,
      pairConsistency: 0.99,
      attestedCases: 80,
      constraintViolations: 0,
    },
  ]);
  const calibrated = plugin
    .getSemanticDiagnostics(appPath)
    .filter((d) => d.source === "semantscript");
  assert.deepEqual(
    calibrated.map((d) => d.code),
    [PLUGIN_DIAGNOSTICS.eceAboveThreshold],
  );
  const lenient = createPlugin({ typescript: ts }).create(
    createInfo(service, host, fixture.root, { eceThreshold: 0.2 }),
  );
  assert.deepEqual(
    lenient
      .getSemanticDiagnostics(appPath)
      .filter((d) => d.source === "semantscript"),
    [],
  );
});

test("compiler diagnostics for malformed sites appear inline and the hover explains them", async (t) => {
  const fixture = await createFixture({ "src/broken.sem.ts": malformedSource });
  t.after(fixture.dispose);
  const { service, host } = createService(fixture.root);
  const plugin = createPlugin({ typescript: ts }).create(
    createInfo(service, host, fixture.root, {}),
  );
  const brokenPath = join(fixture.root, "src", "broken.sem.ts");
  const diagnostics = plugin
    .getSemanticDiagnostics(brokenPath)
    .filter((d) => d.source === "semantscript");
  assert.equal(diagnostics.length, 1);
  assert.equal(diagnostics[0].code, 9100);
  assert.equal(diagnostics[0].start, malformedSource.indexOf("sema`"));
  const hover = plugin.getQuickInfoAtPosition(
    brokenPath,
    malformedSource.indexOf("sema`") + 1,
  );
  assert.match(
    hover.documentation[0].text,
    /not compiled.*\n⚠ TS9100: .*explicit output type argument/u,
  );
  // Non-sema files are untouched.
  assert.deepEqual(
    new SemaEditorState(ts, fixture.root, {}).diagnostics(
      service.getProgram(),
      join(fixture.root, "src", "other.ts"),
    ),
    [],
  );
});

test("tsserver loads the plugin from the tsconfig entry and answers quickinfo and diagnostics", async (t) => {
  const fixture = await createFixture({ "src/app.sem.ts": source });
  t.after(fixture.dispose);
  const appPath = join(fixture.root, "src", "app.sem.ts");
  // VS Code passes the workspace folders as probe locations, which is how a
  // plugin installed in the project's node_modules is found.
  const server = spawn(
    process.execPath,
    [
      tsserverPath,
      "--disableAutomaticTypingAcquisition",
      "--pluginProbeLocations",
      fixture.root,
    ],
    {
      cwd: fixture.root,
      stdio: ["pipe", "pipe", "inherit"],
    },
  );
  t.after(() => server.kill());
  const responses = collectResponses(server.stdout);
  const send = (seq, command, args) => {
    server.stdin.write(
      `${JSON.stringify({ seq, type: "request", command, arguments: args })}\n`,
    );
  };
  send(1, "open", { file: appPath });
  const line = source
    .slice(0, source.indexOf('sema<"urgent"'))
    .split("\n").length;
  const offset = source.split("\n")[line - 1].indexOf("sema") + 2;
  send(2, "quickinfo", { file: appPath, line, offset });
  const quickinfo = await responses.next(
    (message) => message.type === "response" && message.request_seq === 2,
  );
  assert.equal(quickinfo.success, true, JSON.stringify(quickinfo));
  assert.match(
    quickinfo.body.documentation,
    /sema function nf_[a-f0-9]{8}… → "urgent" \| "routine"\ninputs 1/u,
  );
  assert.match(
    quickinfo.body.documentation,
    /artifact: none at .*\.semantscript\/artifact \(unverified; run semantscript train\)/u,
  );
  send(3, "semanticDiagnosticsSync", { file: appPath });
  const diagnostics = await responses.next(
    (message) => message.type === "response" && message.request_seq === 3,
  );
  assert.equal(diagnostics.success, true, JSON.stringify(diagnostics));
  const ours = diagnostics.body.filter((d) => d.source === "semantscript");
  assert.deepEqual(
    ours.map((d) => d.code),
    [PLUGIN_DIAGNOSTICS.noArtifact, PLUGIN_DIAGNOSTICS.noArtifact],
  );
  assert.equal(ours[0].category, "warning");
  assert.equal(ours[0].start.line, line);
});

function collectResponses(stream) {
  const queue = [];
  const waiters = [];
  let buffer = Buffer.alloc(0);
  stream.on("data", (chunk) => {
    buffer = Buffer.concat([buffer, chunk]);
    for (;;) {
      const headerEnd = buffer.indexOf("\r\n\r\n");
      const header =
        headerEnd >= 0
          ? /Content-Length: (\d+)/u.exec(
              buffer.subarray(0, headerEnd).toString("utf8"),
            )
          : null;
      if (header === null) return;
      const start = headerEnd + 4;
      const length = Number(header[1]);
      if (buffer.length < start + length) return;
      const message = JSON.parse(
        buffer.subarray(start, start + length).toString("utf8"),
      );
      buffer = buffer.subarray(start + length);
      const index = waiters.findIndex(({ predicate }) => predicate(message));
      if (index >= 0) waiters.splice(index, 1)[0].resolve(message);
      else queue.push(message);
    }
  });
  return {
    next(predicate) {
      const queued = queue.findIndex(predicate);
      if (queued >= 0) return Promise.resolve(queue.splice(queued, 1)[0]);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error("tsserver response timed out")),
          30_000,
        );
        waiters.push({
          predicate,
          resolve: (message) => {
            clearTimeout(timer);
            resolve(message);
          },
        });
      });
    },
  };
}

function createInfo(service, host, root, config) {
  return {
    languageService: service,
    languageServiceHost: host,
    serverHost: ts.sys,
    config: { name: "@semantscript/compiler/ts-plugin", ...config },
    project: {
      getCurrentDirectory: () => root,
      projectService: { logger: { info() {} } },
    },
  };
}

function createService(root) {
  const configPath = join(root, "tsconfig.json");
  const parsed = ts.getParsedCommandLineOfConfigFile(
    configPath,
    {},
    {
      ...ts.sys,
      onUnRecoverableConfigFileDiagnostic: (diagnostic) => {
        throw new Error(String(diagnostic.messageText));
      },
    },
  );
  const versions = new Map();
  const host = {
    getScriptFileNames: () => parsed.fileNames,
    getScriptVersion: (fileName) => String(versions.get(fileName) ?? 0),
    getScriptSnapshot: (fileName) =>
      ts.sys.fileExists(fileName)
        ? ts.ScriptSnapshot.fromString(ts.sys.readFile(fileName))
        : undefined,
    getCurrentDirectory: () => root,
    getCompilationSettings: () => parsed.options,
    getDefaultLibFileName: (options) => ts.getDefaultLibFilePath(options),
    fileExists: ts.sys.fileExists,
    readFile: ts.sys.readFile,
    readDirectory: ts.sys.readDirectory,
    directoryExists: ts.sys.directoryExists,
    getDirectories: ts.sys.getDirectories,
  };
  return {
    service: ts.createLanguageService(host, ts.createDocumentRegistry()),
    host,
  };
}

async function writeArtifact(root, functions) {
  const release = `releases/sha256-${"a".repeat(64)}`;
  const releaseDirectory = join(root, ".semantscript", "artifact", release);
  await mkdir(releaseDirectory, { recursive: true });
  await writeFile(
    join(releaseDirectory, "manifest.json"),
    JSON.stringify({
      kind: "semantscript.application-artifact",
      functions: functions.map(({ id, ...verification }) => ({
        id,
        verification: {
          status: "passed",
          brier: 0.01,
          exampleFailures: 0,
          typeErrors: 0,
          ...verification,
        },
      })),
    }),
  );
  await writeFile(
    join(root, ".semantscript", "artifact", "current.json"),
    JSON.stringify({
      kind: "semantscript.artifact-pointer",
      pointerVersion: 1,
      release,
      manifestSha256: "a".repeat(64),
    }),
  );
  // Ensure the pointer's mtime moves even when two writes land within one tick.
  await new Promise((resolve) => setTimeout(resolve, 15));
}

async function createFixture(files) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-ts-plugin-"));
  const coreRoot = join(root, "node_modules", "@semantscript", "core");
  await mkdir(coreRoot, { recursive: true });
  await writeFile(
    join(coreRoot, "package.json"),
    JSON.stringify({ name: "@semantscript/core", types: "index.d.ts" }),
  );
  await writeFile(join(coreRoot, "index.d.ts"), coreDeclarations);
  await symlink(
    compilerRoot,
    join(root, "node_modules", "@semantscript", "compiler"),
    "dir",
  );
  await writeFile(
    join(root, "package.json"),
    JSON.stringify({ type: "module", private: true }),
  );
  await writeFile(
    join(root, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        module: "NodeNext",
        moduleResolution: "NodeNext",
        target: "ES2022",
        strict: true,
        outDir: "dist",
        skipLibCheck: true,
        plugins: [{ name: "@semantscript/compiler/ts-plugin" }],
      },
      include: ["src"],
    }),
  );
  await mkdir(join(root, "src"), { recursive: true });
  await writeFile(join(root, "src", "other.ts"), "export const other = 1;\n");
  for (const [fileName, contents] of Object.entries(files)) {
    await writeFile(join(root, fileName), contents);
  }
  return {
    root,
    async dispose() {
      await rm(root, { force: true, recursive: true });
    },
  };
}
