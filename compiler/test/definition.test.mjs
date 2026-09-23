import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import ts from "typescript";

import {
  analyzeSemaSites,
  resolveDefinitionConfiguration,
} from "../dist/index.js";

const coreDeclarations = `
declare const semaConstraintKind: unique symbol;
export interface SemaConstraint<T> {
  readonly [semaConstraintKind]: T;
}
export interface SemaExample<T> {
  readonly inputs: Readonly<Record<string, unknown>>;
  readonly output: T;
}
export interface SemaOptions<T> {
  readonly examples?: readonly SemaExample<T>[];
  readonly constraints?: readonly SemaConstraint<unknown>[];
}
interface ConfiguredSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): T;
}
interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: SemaOptions<T>): ConfiguredSemaTag<T>;
}
export declare const sema: SemaTag;
export declare function always<T>(predicate: () => boolean, output: T): SemaConstraint<T>;
export declare function never<T>(predicate: () => boolean, output: T): SemaConstraint<T>;
`;

test("resolves static examples, constraint ASTs, source text, and confidence", async (t) => {
  const fixture = await createFixtureProgram({
    "definition.sem.ts": `import * as core from "@semantscript/core";

type Decision = "approve" | "deny";
enum Status { Paid = "paid", Fraud = "fraudulent" }
interface Customer { ageDays: number; tier: "standard" | "enterprise" }

const enterprise = { ageDays: -0, tier: "enterprise" } as const;
const paid = Status["Paid"];
declare const customer: Customer;
declare const status: Status;

const decision = core.sema<Decision>({
  examples: [{ inputs: { customer: enterprise, status: paid }, output: "approve" }],
  constraints: [
    core.never(() => status === Status["Fraud"], "approve"),
    core.always(() => customer.ageDays >= 90, "deny"),
  ],
})\`
  @confidence(0.95)
  Decide for \${customer} with status \${status}.
\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const analysisResult = analyzeSemaSites(fixture.program);
  assert.deepEqual(analysisResult.diagnostics, []);
  assert.equal(analysisResult.analyses.length, 1);
  const analysis = analysisResult.analyses[0];
  assert.ok(analysis);

  const result = resolveDefinitionConfiguration(
    fixture.program,
    analysis.site,
    analysis.ir.inputs,
    analysis.ir.output,
  );
  assert.equal(result.ok, true);
  if (!result.ok) return;

  assert.equal(result.value.confidenceThreshold, 0.95);
  assert.equal(
    result.value.definition.template
      .filter((part) => part.kind === "text")
      .some((part) => part.text.includes("@confidence")),
    false,
  );
  assert.deepEqual(
    result.value.definition.template.filter((part) => part.kind === "input"),
    [
      { kind: "input", name: "customer" },
      { kind: "input", name: "status" },
    ],
  );

  const example = result.value.definition.examples[0];
  assert.ok(example);
  assert.equal(example.output, "approve");
  assert.equal(example.inputs.status, "paid");
  assert.equal(Object.is(example.inputs.customer.ageDays, -0), true);

  assert.deepEqual(result.value.definition.constraints, [
    {
      kind: "never",
      source: 'status === Status["Fraud"]',
      predicate: {
        node: "binary",
        operator: "===",
        left: { node: "input", name: "status" },
        right: { node: "literal", value: "fraudulent" },
      },
      output: "approve",
    },
    {
      kind: "always",
      source: "customer.ageDays >= 90",
      predicate: {
        node: "binary",
        operator: ">=",
        left: {
          node: "property",
          object: { node: "input", name: "customer" },
          property: "ageDays",
        },
        right: { node: "literal", value: 90 },
      },
      output: "deny",
    },
  ]);
});

test("treats constant-false constraints as vacuous during contradiction checks", async (t) => {
  const fixture = await createFixtureProgram({
    "vacuous.sem.ts": `import { always, never, sema } from "@semantscript/core";
const decision = sema<"yes" | "no">({
  constraints: [
    always(() => false, "yes"),
    always(() => false, "no"),
    never(() => false, "yes"),
  ],
})\`Choose a decision.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.equal(result.value.definition.constraints.length, 3);
});

test("folds boolean identities with inputs when checking vacuous constraints", async (t) => {
  const fixture = await createFixtureProgram({
    "boolean-identities.sem.ts": `import { always, sema } from "@semantscript/core";
declare const input: boolean;
const decision = sema<"yes" | "no">({
  constraints: [
    always(() => false && input, "yes"),
    always(() => input && false, "no"),
  ],
})\`Choose from \${input}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, true);
});

test("rejects non-finite arithmetic nested below an index expression", async (t) => {
  const fixture = await createFixtureProgram({
    "index-arithmetic.sem.ts": `import { always, sema } from "@semantscript/core";
declare const input: readonly string[];
const decision = sema<"yes" | "no">({
  constraints: [always(() => input[1 / 0] === "x", "yes")],
})\`Choose from \${input}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, false);
  if (result.ok) return;
  assert.equal(result.diagnostics[0]?.code, 9122);
  assert.match(String(result.diagnostics[0]?.messageText), /finite number/);
});

test("rejects prototype and function-valued constraint property chains", async () => {
  const predicates = ["record.toString.length >= 0", 'record["valueOf"].length >= 0'];

  for (const [index, predicate] of predicates.entries()) {
    const fixture = await createFixtureProgram({
      [`prototype-${String(index)}.sem.ts`]: `import { always, sema } from "@semantscript/core";
interface RecordValue { score: number }
declare const record: RecordValue;
const decision = sema<"yes" | "no">({
  constraints: [always(() => ${predicate}, "yes")],
})\`Choose from \${record}.\`;
`,
    });

    try {
      assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);
      const result = resolveOnly(fixture.program);
      assert.equal(result.ok, false);
      if (result.ok) continue;
      assert.equal(result.diagnostics[0]?.code, 9122);
      assert.match(String(result.diagnostics[0]?.messageText), /resolve to JSON data/);
    } finally {
      await fixture.dispose();
    }
  }
});

test("rejects function and container equality in constraints", async () => {
  const predicates = [
    "record.toString === record.valueOf",
    "record === record",
    "values !== values",
  ];

  for (const [index, predicate] of predicates.entries()) {
    const fixture = await createFixtureProgram({
      [`non-scalar-equality-${String(index)}.sem.ts`]: `import { always, sema } from "@semantscript/core";
interface RecordValue { score: number }
declare const record: RecordValue;
declare const values: readonly number[];
const decision = sema<"yes" | "no">({
  constraints: [always(() => ${predicate}, "yes")],
})\`Choose from \${record} and \${values}.\`;
`,
    });

    try {
      assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);
      const result = resolveOnly(fixture.program);
      assert.equal(result.ok, false);
      if (result.ok) continue;
      assert.equal(result.diagnostics[0]?.code, 9122);
      assert.match(String(result.diagnostics[0]?.messageText), /scalar JSON values/);
    } finally {
      await fixture.dispose();
    }
  }
});

test("allows scalar equality reached through JSON-data property and index access", async (t) => {
  const fixture = await createFixtureProgram({
    "json-data-constraint.sem.ts": `import { always, sema } from "@semantscript/core";
interface RecordValue {
  nested: { score: number };
  labels: readonly string[];
  optional?: string;
}
declare const record: RecordValue;
declare const values: readonly number[];
const decision = sema<"yes" | "no">({
  constraints: [
    always(() => record.nested.score === values[0], "yes"),
    always(() => record.labels["length"] >= 1, "yes"),
    always(() => record.optional !== "blocked", "yes"),
  ],
})\`Choose from \${record} and \${values}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, true);
});

test("rejects possibly missing indexed operands without noUncheckedIndexedAccess", async () => {
  const predicates = [
    "values[0] + 1 > 2",
    'text[0] < "z"',
    "!flags[0]",
    "flags[0]",
  ];

  for (const [index, predicate] of predicates.entries()) {
    const fixture = await createFixtureProgram({
      [`possibly-missing-${String(index)}.sem.ts`]: `import { always, sema } from "@semantscript/core";
declare const values: readonly number[];
declare const text: string;
declare const flags: readonly boolean[];
const decision = sema<"yes" | "no">({
  constraints: [always(() => ${predicate}, "yes")],
})\`Choose from \${values}, \${text}, and \${flags}.\`;
`,
    });

    try {
      assert.equal(fixture.program.getCompilerOptions().noUncheckedIndexedAccess, undefined);
      assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);
      const result = resolveOnly(fixture.program);
      assert.equal(result.ok, false);
      if (result.ok) continue;
      assert.equal(result.diagnostics[0]?.code, 9122);
      assert.match(String(result.diagnostics[0]?.messageText), /index is not definitely present/);
    } finally {
      await fixture.dispose();
    }
  }
});

test("rejects a possibly missing element-access index expression", async (t) => {
  const fixture = await createFixtureProgram({
    "possibly-missing-key.sem.ts": `import { always, sema } from "@semantscript/core";
interface RecordValue { a: number; b: number }
declare const record: RecordValue;
declare const keys: readonly ("a" | "b")[];
const decision = sema<"yes" | "no">({
  constraints: [always(() => record[keys[0]] === 1, "yes")],
})\`Choose from \${record} and \${keys}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.equal(fixture.program.getCompilerOptions().noUncheckedIndexedAccess, undefined);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, false);
  if (result.ok) return;
  assert.equal(result.diagnostics[0]?.code, 9122);
  assert.match(String(result.diagnostics[0]?.messageText), /index is not definitely present/);
});

test("rejects dereferencing a possibly missing sequence element", async () => {
  const predicates = ["records[0].score === 1", "matrix[0][0] === 1"];

  for (const [index, predicate] of predicates.entries()) {
    const fixture = await createFixtureProgram({
      [`possibly-missing-base-${String(index)}.sem.ts`]: `import { always, sema } from "@semantscript/core";
interface RecordValue { score: number }
declare const records: readonly RecordValue[];
declare const matrix: readonly (readonly number[])[];
const decision = sema<"yes" | "no">({
  constraints: [always(() => ${predicate}, "yes")],
})\`Choose from \${records} and \${matrix}.\`;
`,
    });

    try {
      assert.equal(fixture.program.getCompilerOptions().noUncheckedIndexedAccess, undefined);
      assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);
      const result = resolveOnly(fixture.program);
      assert.equal(result.ok, false);
      if (result.ok) continue;
      assert.equal(result.diagnostics[0]?.code, 9122);
      assert.match(String(result.diagnostics[0]?.messageText), /index is not definitely present/);
    } finally {
      await fixture.dispose();
    }
  }
});

test("allows direct possibly missing sequence elements in scalar equality", async (t) => {
  const fixture = await createFixtureProgram({
    "possibly-missing-equality.sem.ts": `import { always, sema } from "@semantscript/core";
declare const values: readonly number[];
const decision = sema<"yes" | "no">({
  constraints: [always(() => values[0] === 1, "yes")],
})\`Choose from \${values}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.equal(fixture.program.getCompilerOptions().noUncheckedIndexedAccess, undefined);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, true);
});

test("rejects optional-property dereferences without strict null checks", async () => {
  const predicates = [
    "record.nested.score === 1",
    'record["nested"].score === 1',
    "record[key].score === 1",
  ];

  for (const [index, predicate] of predicates.entries()) {
    const fixture = await createFixtureProgram(
      {
        [`optional-property-dereference-${String(index)}.sem.ts`]: `import { always, sema } from "@semantscript/core";
interface RecordValue { nested?: { score: number } }
declare const record: RecordValue;
declare const key: "nested";
const decision = sema<"yes" | "no">({
  constraints: [always(() => ${predicate}, "yes")],
})\`Choose from \${record} using \${key}.\`;
`,
      },
      { strict: false },
    );

    try {
      assert.equal(fixture.program.getCompilerOptions().strict, false);
      assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);
      const result = resolveOnly(fixture.program);
      assert.equal(result.ok, false);
      if (result.ok) continue;
      assert.equal(result.diagnostics[0]?.code, 9122);
      assert.match(String(result.diagnostics[0]?.messageText), /optional property/);
    } finally {
      await fixture.dispose();
    }
  }
});

test("allows direct optional-property equality without strict null checks", async (t) => {
  const fixture = await createFixtureProgram(
    {
      "optional-property-equality.sem.ts": `import { always, sema } from "@semantscript/core";
interface RecordValue { label?: string }
declare const record: RecordValue;
declare const key: "label";
const decision = sema<"yes" | "no">({
  constraints: [
    always(() => record.label === "ready", "yes"),
    always(() => record["label"] !== "blocked", "yes"),
    always(() => record[key] === "ready", "yes"),
  ],
})\`Choose from \${record} using \${key}.\`;
`,
    },
    { strict: false },
  );
  t.after(fixture.dispose);
  assert.equal(fixture.program.getCompilerOptions().strict, false);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, true);
});

test("allows in-range tuple arithmetic and length operands", async (t) => {
  const fixture = await createFixtureProgram({
    "definite-index.sem.ts": `import { always, sema } from "@semantscript/core";
declare const pair: readonly [number, number];
declare const values: readonly number[];
declare const text: string;
const decision = sema<"yes" | "no">({
  constraints: [
    always(() => pair[0] + 1 > pair[1], "yes"),
    always(() => pair["0"] + 1 > pair[1], "yes"),
    always(() => values.length >= 0, "yes"),
    always(() => text["length"] >= 0, "yes"),
  ],
})\`Choose from \${pair}, \${values}, and \${text}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.equal(fixture.program.getCompilerOptions().noUncheckedIndexedAccess, undefined);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, true);
});

test("accepts positive-zero constraint output for negative-zero numeric enum support", async (t) => {
  const fixture = await createFixtureProgram({
    "signed-zero-output.sem.ts": `import { always, sema } from "@semantscript/core";
enum Decision { Zero = -0, One = 1 }
declare const flag: boolean;
const decision = sema<Decision>({
  constraints: [always(() => flag, 0)],
})\`Choose from \${flag}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const analysisResult = analyzeSemaSites(fixture.program);
  assert.deepEqual(analysisResult.diagnostics, []);
  const analysis = analysisResult.analyses[0];
  assert.ok(analysis);
  assert.equal(analysis.ir.output.kind, "scalar");
  assert.deepEqual(analysis.ir.output.head.support.slice(1), [1]);
  assert.equal(Object.is(analysis.ir.output.head.support[0], -0), true);

  const result = resolveDefinitionConfiguration(
    fixture.program,
    analysis.site,
    analysis.ir.inputs,
    analysis.ir.output,
  );
  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.equal(Object.is(result.value.definition.constraints[0]?.output, 0), true);
  assert.equal(Object.is(result.value.definition.constraints[0]?.output, -0), false);
});

test("rejects contradictory constant-true constraints with an accurate diagnostic", async (t) => {
  const fixture = await createFixtureProgram({
    "contradiction.sem.ts": `import { always, sema } from "@semantscript/core";
const decision = sema<"yes" | "no">({
  constraints: [always(() => true, "yes"), always(() => true, "no")],
})\`Choose a decision.\`;
`,
  });
  t.after(fixture.dispose);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, false);
  if (result.ok) return;
  assert.equal(result.diagnostics[0]?.code, 9125);
  assert.match(String(result.diagnostics[0]?.messageText), /require the same output/);
});

test("bounds constraint nesting before recursive parsing can exhaust the process", async (t) => {
  const predicate = `${"(".repeat(102)}true${")".repeat(102)}`;
  const fixture = await createFixtureProgram({
    "deep.sem.ts": `import { always, sema } from "@semantscript/core";
const decision = sema<"yes" | "no">({
  constraints: [always(() => ${predicate}, "yes")],
})\`Choose a decision.\`;
`,
  });
  t.after(fixture.dispose);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, false);
  if (result.ok) return;
  assert.equal(result.diagnostics[0]?.code, 9122);
  assert.match(String(result.diagnostics[0]?.messageText), /nesting exceeds the limit/);
});

test("preserves nominal input assignability while contextualizing inline literals", async (t) => {
  const fixture = await createFixtureProgram({
    "branded.sem.ts": `import { sema } from "@semantscript/core";
declare const userIdBrand: unique symbol;
type UserId = string & { readonly [userIdBrand]: true };
declare const userId: UserId;
const decision = sema<"yes" | "no">({
  examples: [{ inputs: { userId: "plain" }, output: "yes" }],
})\`Decide for \${userId}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, false);
  if (result.ok) return;
  assert.equal(result.diagnostics[0]?.code, 9121);
  assert.match(String(result.diagnostics[0]?.messageText), /not assignable/);
});

test("rejects a lookalike constraint constructor by symbol identity", async (t) => {
  const fixture = await createFixtureProgram({
    "shadow.sem.ts": `import { always as coreAlways, sema, type SemaConstraint } from "@semantscript/core";
type Decision = "approve" | "deny";
declare const input: boolean;
const always = <T>(predicate: () => boolean, output: T): SemaConstraint<T> =>
  coreAlways(predicate, output);
const decision = sema<Decision>({
  constraints: [always(() => input, "approve")],
})\`Decide from \${input}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = resolveOnly(fixture.program);
  assert.equal(result.ok, false);
  if (result.ok) return;
  assert.equal(result.diagnostics[0]?.code, 9122);
  assert.match(String(result.diagnostics[0]?.messageText), /must resolve to always or never/);
});

test("rejects non-inline options, non-static example values, and misplaced directives", async () => {
  const cases = [
    {
      source: `import { sema } from "@semantscript/core";
const options = { examples: [] };
const decision = sema<"yes" | "no">(options)\`Behavior.\`;
`,
      code: 9120,
    },
    {
      source: `import { sema } from "@semantscript/core";
declare const input: number;
const makeValue = () => 1;
const decision = sema<"yes" | "no">({
  examples: [{ inputs: { input: makeValue() }, output: "yes" }],
})\`Behavior for \${input}.\`;
`,
      code: 9121,
    },
    {
      source: `import { sema } from "@semantscript/core";
const decision = sema<"yes" | "no">\`Behavior.
@confidence(0.9)
\`;
`,
      code: 9123,
    },
    {
      source: `import { sema } from "@semantscript/core";
const decision = sema<"yes" | "no">\`
@confidence(0.9)
\`;
`,
      code: 9124,
    },
  ];

  for (const [index, fixtureCase] of cases.entries()) {
    const fixture = await createFixtureProgram({
      [`invalid-${String(index)}.sem.ts`]: fixtureCase.source,
    });

    try {
      const result = resolveOnly(fixture.program);
      assert.equal(result.ok, false);
      if (result.ok) continue;
      assert.equal(result.diagnostics[0]?.code, fixtureCase.code);
      assert.ok(result.diagnostics[0]?.file);
      assert.equal(typeof result.diagnostics[0]?.start, "number");
    } finally {
      await fixture.dispose();
    }
  }
});

function resolveOnly(program) {
  const analysisResult = analyzeSemaSites(program);
  assert.deepEqual(analysisResult.diagnostics, []);
  assert.equal(analysisResult.analyses.length, 1);
  const analysis = analysisResult.analyses[0];
  assert.ok(analysis);
  return resolveDefinitionConfiguration(
    program,
    analysis.site,
    analysis.ir.inputs,
    analysis.ir.output,
  );
}

async function createFixtureProgram(files, compilerOptions = {}) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-definition-"));
  const coreRoot = join(root, "node_modules", "@semantscript", "core");
  await mkdir(coreRoot, { recursive: true });
  await writeFile(
    join(coreRoot, "package.json"),
    JSON.stringify({ name: "@semantscript/core", type: "module", types: "index.d.ts" }),
  );
  await writeFile(join(coreRoot, "index.d.ts"), coreDeclarations);
  await writeFile(join(root, "package.json"), JSON.stringify({ type: "module" }));

  for (const [fileName, contents] of Object.entries(files)) {
    await writeFile(join(root, fileName), contents);
  }

  const program = ts.createProgram({
    rootNames: Object.keys(files).map((fileName) => join(root, fileName)),
    options: {
      module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      skipLibCheck: true,
      strict: true,
      target: ts.ScriptTarget.ES2023,
      ...compilerOptions,
    },
  });

  const fixture = {
    program,
    async dispose() {
      fixture.program = undefined;
      await rm(root, { force: true, recursive: true });
    },
  };
  return fixture;
}
