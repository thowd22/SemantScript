#!/usr/bin/env node
// The installed `semantscript` command. It lives outside dist/ so npm links
// it on the first `npm install` of a fresh clone, before `npm run build` has
// produced dist/index.js; run before a build, it says what to do instead of
// failing with a module-resolution trace.
import { existsSync } from "node:fs";
import process from "node:process";

const entry = new URL("../dist/index.js", import.meta.url);

if (!existsSync(entry)) {
  process.stderr.write(
    "semantscript: the CLI is not built yet; run `npm run build` at the repository root\n",
  );
  process.exit(1);
}

const cli = await import(entry.href);

cli.runCli(process.argv.slice(2)).then(
  (code) => {
    process.exitCode = code;
  },
  (error) => {
    process.stderr.write(
      `${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  },
);
