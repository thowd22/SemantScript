import { spawnSync } from "node:child_process";
import process from "node:process";

const mode = process.argv[2] ?? "check";
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const steps = {
  lint: [
    [npm, ["run", "lint:node"]],
    [process.execPath, ["scripts/check-python.mjs", "lint"]],
  ],
  test: [
    [npm, ["run", "test:node"]],
    [process.execPath, ["scripts/check-python.mjs", "test"]],
  ],
  // Build first: the type-aware lint resolves each workspace's imports of the
  // others through their dist/ declarations, which a fresh clone lacks.
  check: [
    [npm, ["run", "build"]],
    [npm, ["run", "lint:node"]],
    [process.execPath, ["scripts/check-python.mjs", "lint"]],
    [npm, ["run", "test:node"]],
    [process.execPath, ["scripts/check-python.mjs", "test"]],
  ],
};

if (!Object.hasOwn(steps, mode)) {
  console.error("usage: node scripts/check.mjs <check|lint|test>");
  process.exit(2);
}

for (const [command, args] of steps[mode]) {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    shell: process.platform === "win32" && command === npm,
    stdio: "inherit",
  });

  if (result.error) {
    console.error(`unable to run ${command}: ${result.error.message}`);
    process.exit(1);
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
