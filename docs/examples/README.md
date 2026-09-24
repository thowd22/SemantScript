# Language reference examples

Each file here is compiled by `compiler/test/docs-examples.test.mjs` with the
real compiler, so the snippets quoted in `docs/language-reference.md` cannot
drift from what the compiler accepts. They import `@semantscript/core` like any
application source; build them with `semantscript build --project` from a
project whose `tsconfig.json` includes this directory.
