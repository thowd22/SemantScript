import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, join, resolve } from "node:path";
import process from "node:process";

const action = process.argv[2];
const actions = {
  lint: [
    [
      "-m",
      "ruff",
      "check",
      "trainer",
      "model",
      "benchmarks/refund/program",
      "benchmarks/refund/live",
    ],
    [
      "-m",
      "ruff",
      "format",
      "--check",
      "trainer",
      "model",
      "benchmarks/refund/program",
      "benchmarks/refund/live",
    ],
  ],
  test: [["-m", "pytest"]],
};

if (!Object.hasOwn(actions, action)) {
  console.error("usage: node scripts/check-python.mjs <lint|test>");
  process.exit(2);
}

const environmentPython = process.env.VIRTUAL_ENV
  ? join(
      process.env.VIRTUAL_ENV,
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    )
  : undefined;
const localPython =
  process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
const localPackages = ".python-packages";
const explicitPython = process.env.SEMANTSCRIPT_PYTHON;
const preparedPython = [environmentPython, localPython].find(
  (candidate) => candidate && existsSync(candidate),
);
const useLocalPackages = !explicitPython && !preparedPython && existsSync(localPackages);
const python =
  explicitPython ??
  preparedPython ??
  process.env.PYTHON ??
  (process.platform === "win32" ? "python" : "python3");
const pythonPath = [
  resolve("trainer/src"),
  resolve("model/src"),
  useLocalPackages ? resolve(localPackages) : undefined,
  process.env.PYTHONPATH,
]
  .filter(Boolean)
  .join(delimiter);

for (const args of actions[action]) {
  const result = spawnSync(python, args, {
    cwd: process.cwd(),
    env: {
      ...process.env,
      ...(pythonPath ? { PYTHONPATH: pythonPath } : {}),
      ...(action === "test" ? { PYTEST_DISABLE_PLUGIN_AUTOLOAD: "1" } : {}),
    },
    stdio: "inherit",
  });

  if (result.error) {
    console.error(`unable to run ${python}: ${result.error.message}`);
    process.exit(1);
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
