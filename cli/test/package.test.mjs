import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import {
  lstat,
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import test from "node:test";

import { createFixtureArtifact } from "../../runtime/test/fixtures/artifact.mjs";
import {
  MEASURED_ENCODER,
  PACKAGE_TARGETS,
  packageLevers,
  pointerBytes,
  pruneNativeBindings,
  runCli,
} from "../dist/index.js";

function capture(cwd) {
  const out = [];
  const err = [];
  return {
    io: {
      cwd,
      // No network: the stub project's only dependency is a local file: package.
      env: {
        ...process.env,
        SEMANTSCRIPT_ARTIFACT: undefined,
        npm_config_offline: "true",
      },
      stdout: (text) => out.push(text),
      stderr: (text) => err.push(text),
    },
    stdout: () => out.join(""),
    stderr: () => err.join(""),
  };
}

async function scratch(t, prefix) {
  const root = await mkdtemp(join(tmpdir(), prefix));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function run(cwd, args) {
  const output = capture(cwd);
  const code = await runCli(["package", ...args], output.io);
  return { code, stdout: output.stdout(), stderr: output.stderr() };
}

async function writeJson(file, value) {
  await mkdir(join(file, ".."), { recursive: true });
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`);
}

/**
 * A compiled application: dist/ with its IR bundle, a local file: dependency,
 * a devDependency that must not be installed, a deploy handler and an
 * artifact root with two releases (the newer one current).
 */
async function stubProject(root, { dependencies = true } = {}) {
  const project = join(root, "app");
  await writeJson(join(root, "stub-dep", "package.json"), {
    name: "stub-dep",
    version: "1.0.0",
    main: "index.js",
  });
  await writeFile(join(root, "stub-dep", "index.js"), "module.exports = 42;\n");
  await writeJson(join(project, "package.json"), {
    name: "stub-app",
    version: "1.2.3",
    type: "module",
    scripts: { start: "node dist/server.js" },
    ...(dependencies
      ? { dependencies: { "stub-dep": "file:../stub-dep" } }
      : {}),
    devDependencies: { "never-installed": "9.9.9" },
  });
  await mkdir(join(project, "dist", "nested"), { recursive: true });
  await writeFile(join(project, "dist", "server.js"), "export const ok = 1;\n");
  await writeFile(join(project, "dist", "nested", "a.js"), "export {};\n");
  await writeFile(
    join(project, "dist", "semantscript.ir.v1.json"),
    '{"prompt":"secret"}\n',
  );
  await mkdir(join(project, "deploy"), { recursive: true });
  await writeFile(
    join(project, "deploy", "handler.mjs"),
    "export const handler = () => 1;\n",
  );
  const artifact = join(project, ".semantscript", "artifact");
  const older = await createFixtureArtifact(artifact, {
    transformManifest: (manifest) => {
      manifest.application.version = "0.0.1";
      manifest.build.createdAt = "2020-01-01T00:00:00Z";
    },
  });
  const current = await createFixtureArtifact(artifact, {
    transformManifest: (manifest) => {
      manifest.application.version = "0.0.2";
      manifest.build.createdAt = "2020-01-02T00:00:00Z";
    },
  });
  return {
    project,
    artifact,
    older: older.manifestSha256,
    current: current.manifestSha256,
  };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

test("package writes dist without the IR bundle, production dependencies, the current release only, the includes and a manifest of digests", async (t) => {
  const root = await scratch(t, "semantscript-package-");
  const { project, older, current } = await stubProject(root);
  const result = await run(project, [
    "--include",
    "deploy/handler.mjs",
    "--target",
    "lambda-zip",
    "--json",
  ]);
  assert.equal(result.code, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  const out = join(project, ".semantscript", "package");
  assert.equal(report.out, out);

  assert.deepEqual((await readdir(out)).sort(), [
    ".semantscript",
    "deploy",
    "dist",
    "node_modules",
    "package.json",
    "semantscript-package.json",
  ]);
  assert.ok(existsSync(join(out, "dist", "server.js")));
  assert.ok(existsSync(join(out, "dist", "nested", "a.js")));
  assert.ok(!existsSync(join(out, "dist", "semantscript.ir.v1.json")));
  assert.ok(existsSync(join(out, "deploy", "handler.mjs")));

  // The file: dependency is a real copy, not a symlink into the repository.
  const stats = await lstat(join(out, "node_modules", "stub-dep"));
  assert.ok(stats.isDirectory() && !stats.isSymbolicLink());
  assert.ok(!existsSync(join(out, "node_modules", "never-installed")));
  assert.ok(!existsSync(join(out, "node_modules", ".package-lock.json")));
  assert.ok(!existsSync(join(out, "package-lock.json")));
  const deployed = JSON.parse(
    await readFile(join(out, "package.json"), "utf8"),
  );
  assert.equal(deployed.type, "module");
  assert.equal(deployed.devDependencies, undefined);

  // Only the current release, and a pointer to it.
  const releases = await readdir(
    join(out, ".semantscript", "artifact", "releases"),
  );
  assert.deepEqual(releases, [`sha256-${current}`]);
  assert.notEqual(older, current);
  assert.equal(
    await readFile(
      join(out, ".semantscript", "artifact", "current.json"),
      "utf8",
    ),
    pointerBytes(current),
  );

  // Every file is in the manifest with its size and digest.
  const manifest = JSON.parse(
    await readFile(join(out, "semantscript-package.json"), "utf8"),
  );
  assert.equal(manifest.kind, "semantscript.package");
  assert.equal(manifest.version, 1);
  assert.equal(manifest.release.manifestSha256, current);
  assert.equal(manifest.platform, process.platform);
  assert.equal(manifest.arch, process.arch);
  const paths = manifest.files.map((file) => file.path);
  assert.ok(paths.includes("dist/server.js"));
  assert.ok(paths.includes("deploy/handler.mjs"));
  assert.ok(
    paths.includes(
      `.semantscript/artifact/releases/sha256-${current}/manifest.json`,
    ),
  );
  assert.ok(!paths.includes("semantscript-package.json"));
  for (const file of manifest.files.filter(
    (entry) => entry.sha256 !== undefined,
  )) {
    const bytes = await readFile(join(out, file.path));
    assert.equal(bytes.length, file.bytes, file.path);
    assert.equal(sha256(bytes), file.sha256, file.path);
  }
  assert.equal(
    manifest.filesBytes,
    manifest.files.reduce((total, file) => total + file.bytes, 0),
  );

  // The report: parts, total and the target.
  const part = (name) => report.parts.find((entry) => entry.part === name);
  assert.ok(part("dist").bytes > 0);
  assert.ok(
    part("node_modules").children.some((child) => child.part === "stub-dep"),
  );
  const artifactParts = part("artifact").children.map((child) => child.part);
  assert.deepEqual(artifactParts, [
    "encoder",
    "adapter",
    "heads",
    "tokenizer",
    "manifest and pointer",
  ]);
  assert.equal(part("included").children[0].part, "deploy/handler.mjs");
  const manifestBytes = (await readFile(join(out, "semantscript-package.json")))
    .length;
  assert.equal(report.manifestBytes, manifestBytes);
  assert.equal(report.totalBytes, manifest.filesBytes + manifestBytes);
  assert.equal(report.target.name, "lambda-zip");
  assert.equal(report.target.fits, true);
  assert.deepEqual(report.levers, []);
});

test("package prints the size table and refuses to overwrite a package without --force", async (t) => {
  const root = await scratch(t, "semantscript-package-text-");
  const { project } = await stubProject(root, { dependencies: false });
  const first = await run(project, ["--out", "bundle"]);
  assert.equal(first.code, 0, first.stderr);
  assert.match(first.stdout, /^package {2}.*bundle$/mu);
  assert.match(first.stdout, /^install {2}no production dependencies$/mu);
  assert.match(first.stdout, /^part +bytes +size$/mu);
  for (const part of [
    "dist",
    "node_modules",
    "artifact",
    "  encoder",
    "  tokenizer",
    "total",
  ]) {
    assert.match(first.stdout, new RegExp(`^${part} +[0-9]+ +`, "mu"), part);
  }
  assert.ok(!existsSync(join(project, "bundle", "node_modules")));

  const again = await run(project, ["--out", "bundle"]);
  assert.equal(again.code, 1);
  assert.match(again.stderr, /PACKAGE_OUT_EXISTS: .*pass --force/u);
  const forced = await run(project, ["--out", "bundle", "--force"]);
  assert.equal(forced.code, 0, forced.stderr);

  // A directory package did not write is never replaced, even with --force.
  await mkdir(join(project, "other"));
  await writeFile(join(project, "other", "keep.txt"), "mine\n");
  const refused = await run(project, ["--out", "other", "--force"]);
  assert.equal(refused.code, 1);
  assert.match(refused.stderr, /PACKAGE_OUT_EXISTS: .*did not write/u);
  assert.equal(
    await readFile(join(project, "other", "keep.txt"), "utf8"),
    "mine\n",
  );
});

test("package over its target still writes the bundle, exits 1 with PACKAGE_OVER_TARGET and lists the levers", async (t) => {
  const root = await scratch(t, "semantscript-package-over-");
  const { project } = await stubProject(root, { dependencies: false });
  const text = await run(project, ["--max-bytes", "1000"]);
  assert.equal(text.code, 1);
  assert.match(
    text.stderr,
    /^PACKAGE_OVER_TARGET: the bundle is .* over max-bytes's/u,
  );
  assert.ok(
    existsSync(
      join(project, ".semantscript", "package", "semantscript-package.json"),
    ),
  );
  assert.match(text.stdout, /over by .*; levers/u);
  assert.match(text.stdout, /^depth routing to 4 layers +~/mu);
  assert.match(text.stdout, /^int8 dynamic quantization +~/mu);
  assert.match(text.stdout, /^a smaller encoder +/mu);
  assert.match(text.stdout, /recorded tolerance/u);
  assert.match(text.stdout, /semantscript build --domain-depth <domain>=6/u);

  const json = await run(project, ["--max-bytes", "1000", "--force", "--json"]);
  assert.equal(json.code, 1);
  const report = JSON.parse(json.stdout);
  assert.equal(report.target.fits, false);
  assert.equal(report.target.overBytes, report.totalBytes - 1000);
  assert.deepEqual(
    report.levers.map((lever) => lever.lever),
    [
      "depth",
      "depth",
      "depth",
      "int8",
      "depth+int8",
      "depth+int8",
      "smaller-encoder",
    ],
  );
  assert.ok(report.levers.every((lever) => lever.fits === false));
});

test("packageLevers projects the measured depth and int8 sizes and skips levers the release already uses", () => {
  // The trained Express release: a full-depth float32 ModernBERT-base encoder.
  const encoderBytes = MEASURED_ENCODER.fullBytes;
  const totalBytes = 685_309_329;
  const rest = totalBytes - encoderBytes;
  const limitBytes = PACKAGE_TARGETS["lambda-zip"].bytes;
  assert.equal(limitBytes, 262_144_000);
  const levers = packageLevers({
    totalBytes,
    encoderBytes,
    limitBytes,
    depthRouted: false,
    quantized: false,
  });
  const byLabel = new Map(levers.map((lever) => [lever.label, lever]));
  assert.equal(
    byLabel.get("depth routing to 4 layers").projectedBytes,
    rest + MEASURED_ENCODER.depthBytes[4],
  );
  assert.equal(byLabel.get("depth routing to 4 layers").fits, false);
  assert.equal(
    byLabel.get("int8 dynamic quantization").projectedBytes,
    rest + MEASURED_ENCODER.int8Bytes,
  );
  assert.equal(byLabel.get("int8 dynamic quantization").fits, true);
  assert.match(
    byLabel.get("int8 dynamic quantization").how,
    /recorded tolerance/u,
  );
  assert.match(
    byLabel.get("int8 dynamic quantization").how,
    /no int8 derivation exists for applications yet/u,
  );
  assert.match(
    byLabel.get("int8 dynamic quantization").how,
    /changed 1 of 80 attested cases/u,
  );
  assert.doesNotMatch(
    byLabel.get("int8 dynamic quantization").how,
    /quantize_release/u,
  );
  assert.equal(byLabel.get("depth 4 and int8").fits, true);
  const smaller = byLabel.get("a smaller encoder");
  assert.equal(smaller.encoderBudgetBytes, limitBytes - rest);
  assert.match(smaller.how, /--encoder-name/u);

  const routed = packageLevers({
    totalBytes,
    encoderBytes,
    limitBytes,
    depthRouted: true,
    quantized: false,
  });
  assert.deepEqual(
    routed.map((lever) => lever.lever),
    ["int8", "smaller-encoder"],
  );
  const quantized = packageLevers({
    totalBytes,
    encoderBytes,
    limitBytes,
    depthRouted: false,
    quantized: true,
  });
  assert.deepEqual(
    quantized.map((lever) => lever.lever),
    ["depth", "depth", "depth", "smaller-encoder"],
  );
  const hopeless = packageLevers({
    totalBytes: 2 * limitBytes,
    encoderBytes: 1,
    limitBytes,
    depthRouted: true,
    quantized: true,
  });
  assert.equal(hopeless.length, 1);
  assert.equal(hopeless[0].fits, false);
  assert.match(hopeless[0].how, /nothing fits/u);
});

test("package recognises a depth-routed release and skips the depth levers", async (t) => {
  const root = await scratch(t, "semantscript-package-routed-");
  const { project, artifact } = await stubProject(root, {
    dependencies: false,
  });
  await createFixtureArtifact(artifact, {
    extraEncoders: [{ ref: "encoder.depth4" }],
    transformManifest: (manifest) => {
      manifest.build.createdAt = "2020-01-03T00:00:00Z";
      manifest.functions[0].encoderRef = "encoder.depth4";
    },
  });
  const result = await run(project, ["--max-bytes", "1000", "--json"]);
  assert.equal(result.code, 1, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.encoder.depthRouted, true);
  assert.ok(report.levers.every((lever) => !lever.lever.startsWith("depth")));
});

async function nativeTree(root) {
  const nm = join(root, "node_modules");
  const ort = join(nm, "onnxruntime-node", "bin", "napi-v6");
  const files = {
    "linux/x64": [
      "onnxruntime_binding.node",
      "libonnxruntime.so.1",
      "libonnxruntime_providers_shared.so",
      "libonnxruntime_providers_cuda.so",
      "libonnxruntime_providers_tensorrt.so",
    ],
    "linux/arm64": ["onnxruntime_binding.node", "libonnxruntime.so.1"],
    "darwin/arm64": ["onnxruntime_binding.node", "libonnxruntime.1.dylib"],
    "win32/x64": [
      "onnxruntime_binding.node",
      "onnxruntime.dll",
      "DirectML.dll",
    ],
  };
  for (const [triple, names] of Object.entries(files)) {
    await mkdir(join(ort, triple), { recursive: true });
    for (const name of names)
      await writeFile(join(ort, triple, name), "x".repeat(10));
  }
  const tokenizerNames = [
    "index.js",
    "tokenizers.linux-x64-gnu.node",
    "tokenizers.linux-x64-musl.node",
    "tokenizers.linux-arm64-gnu.node",
    "tokenizers.darwin-universal.node",
    "tokenizers.darwin-arm64.node",
    "tokenizers.win32-x64-msvc.node",
  ];
  // A top-level copy and a nested one under another package.
  for (const directory of [
    join(nm, "tokenizers"),
    join(nm, "@scope", "user", "node_modules", "tokenizers"),
  ]) {
    await mkdir(directory, { recursive: true });
    for (const name of tokenizerNames)
      await writeFile(join(directory, name), "y".repeat(5));
  }
  return nm;
}

test("pruneNativeBindings keeps only the target's bindings and drops the GPU providers", async (t) => {
  const root = await scratch(t, "semantscript-package-prune-");
  const nm = await nativeTree(root);
  const result = await pruneNativeBindings(nm, "linux", "x64");
  const ort = join(nm, "onnxruntime-node", "bin", "napi-v6");
  assert.deepEqual(await readdir(ort), ["linux"]);
  assert.deepEqual(await readdir(join(ort, "linux")), ["x64"]);
  assert.deepEqual((await readdir(join(ort, "linux", "x64"))).sort(), [
    "libonnxruntime.so.1",
    "libonnxruntime_providers_shared.so",
    "onnxruntime_binding.node",
  ]);
  for (const directory of [
    join(nm, "tokenizers"),
    join(nm, "@scope", "user", "node_modules", "tokenizers"),
  ]) {
    assert.deepEqual((await readdir(directory)).sort(), [
      "index.js",
      "tokenizers.linux-x64-gnu.node",
      "tokenizers.linux-x64-musl.node",
    ]);
  }
  assert.ok(result.removed.includes("onnxruntime-node/bin/napi-v6/win32"));
  assert.ok(
    result.removed.includes(
      "onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime_providers_cuda.so",
    ),
  );
  // 3 other ORT platform dirs (2+2+3 files) + 2 providers, 2 x 4 tokenizers.
  assert.equal(result.bytes, 7 * 10 + 2 * 10 + 8 * 5);

  const darwin = await scratch(t, "semantscript-package-prune-darwin-");
  await pruneNativeBindings(await nativeTree(darwin), "darwin", "arm64");
  assert.deepEqual(
    (await readdir(join(darwin, "node_modules", "tokenizers"))).sort(),
    [
      "index.js",
      "tokenizers.darwin-arm64.node",
      "tokenizers.darwin-universal.node",
    ],
  );

  const missing = await scratch(t, "semantscript-package-prune-missing-");
  await assert.rejects(
    pruneNativeBindings(await nativeTree(missing), "freebsd", "x64"),
    (error) => error.code === "PACKAGE_NO_BINDING",
  );
});

test("package reports missing inputs, bad options and an unverified release with stable codes", async (t) => {
  const root = await scratch(t, "semantscript-package-errors-");
  const { project, artifact } = await stubProject(root, {
    dependencies: false,
  });

  const unknownTarget = await run(project, ["--target", "heroku"]);
  assert.equal(unknownTarget.code, 2);
  assert.match(
    unknownTarget.stderr,
    /unknown --target heroku: expected lambda-zip, lambda-image, cloud-run-functions/u,
  );
  const both = await run(project, [
    "--target",
    "lambda-zip",
    "--max-bytes",
    "5",
  ]);
  assert.equal(both.code, 2);
  const badBytes = await run(project, ["--max-bytes", "5MB"]);
  assert.equal(badBytes.code, 2);
  const positional = await run(project, ["extra"]);
  assert.equal(positional.code, 2);
  const outside = await run(project, ["--include", "../stub-dep"]);
  assert.equal(outside.code, 2);
  assert.match(outside.stderr, /inside the project/u);
  const reserved = await run(project, ["--include", "package.json"]);
  assert.equal(reserved.code, 2);
  const irBundle = await run(project, [
    "--include",
    "dist/semantscript.ir.v1.json",
  ]);
  assert.equal(irBundle.code, 2);
  assert.match(irBundle.stderr, /dist is written by package itself/u);
  const selfOut = await run(project, ["--out", "."]);
  assert.equal(selfOut.code, 2);

  const missingInclude = await run(project, ["--include", "deploy/none.mjs"]);
  assert.equal(missingInclude.code, 1);
  assert.match(missingInclude.stderr, /^PACKAGE_INCLUDE_MISSING: /u);

  const noProject = await run(root, []);
  assert.equal(noProject.code, 1);
  assert.match(noProject.stderr, /^PACKAGE_NO_PROJECT: /u);

  const noDist = await run(project, ["--dist", "build"]);
  assert.equal(noDist.code, 1);
  assert.match(noDist.stderr, /^PACKAGE_NO_DIST: /u);

  const noRelease = await run(project, ["--artifact", join(root, "empty")]);
  assert.equal(noRelease.code, 1);
  assert.match(noRelease.stderr, /^PACKAGE_NO_RELEASE: /u);

  await createFixtureArtifact(artifact, {
    transformManifest: (manifest) => {
      manifest.build.createdAt = "2020-01-04T00:00:00Z";
      manifest.functions[0].verification.status = "failed";
    },
  });
  const unverified = await run(project, []);
  assert.equal(unverified.code, 1);
  assert.match(unverified.stderr, /^RELEASE_UNVERIFIED: /u);
  assert.ok(!existsSync(join(project, ".semantscript", "package")));
  // No staging directory is left behind.
  assert.deepEqual(
    (await readdir(join(project, ".semantscript"))).filter((name) =>
      name.startsWith(".package-staging-"),
    ),
    [],
  );
});

test("package --help prints the usage", async (t) => {
  const root = await scratch(t, "semantscript-package-help-");
  const result = await run(root, ["--help"]);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /package \[--project <dir>\]/u);
});
