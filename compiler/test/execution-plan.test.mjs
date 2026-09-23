import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import test from "node:test";

import ts from "typescript";

import { buildSemaExecutionPlan, ExecutionPlanError } from "../dist/execution-plan.js";

const declarations = `
declare function sema<T>(strings: TemplateStringsArray, ...values: unknown[]): T;
declare const external: string;
`;

test("builds deterministic independent, chain, and diamond stages", async (t) => {
  const fixture = await createFixture({
    "graph.sem.ts": `
const root = sema<boolean>\`ROOT \${external}\`;
const other = sema<boolean>\`OTHER \${external}\`;
const mutable = root;
const left = sema<boolean>\`LEFT \${mutable}\`;
const right = sema<boolean>\`RIGHT \${root}\`;
const joined = String(left) + String(right);
const final = sema<boolean>\`FINAL \${joined}\`;
`,
  });
  t.after(fixture.dispose);

  const descriptors = collectDescriptors(fixture).reverse();
  const plan = buildSemaExecutionPlan(fixture.checker, descriptors);

  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-root", "nf-other"] },
    { index: 1, functionIds: ["nf-left", "nf-right"] },
    { index: 2, functionIds: ["nf-final"] },
  ]);
  assert.deepEqual(plan.dependencies, [
    dependency("root", "left", "mutable"),
    dependency("root", "right", "root"),
    dependency("left", "final", "joined"),
    dependency("right", "final", "joined"),
  ]);
});

test("follows aliases, destructuring defaults, derived initializers, and duplicate edge labels", async (t) => {
  const fixture = await createFixture({
    "aliases.sem.ts": `
const producer = sema<{ value: string }>\`PRODUCER\`;
const alias = producer;
const { value: picked = "fallback" } = alias;
const derived = Object.freeze({ picked, alias });
const consumer = sema<boolean>\`CONSUMER \${derived} \${picked}\`;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture));
  assert.deepEqual(plan.dependencies, [
    dependency("producer", "consumer", "derived"),
    dependency("producer", "consumer", "picked"),
  ]);
  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-producer"] },
    { index: 1, functionIds: ["nf-consumer"] },
  ]);
});

test("follows initialized let aliases conservatively", async (t) => {
  const fixture = await createFixture({
    "mutable.sem.ts": `
const producer = sema<boolean>\`PRODUCER\`;
let mutable = producer;
mutable = false;
const consumer = sema<boolean>\`CONSUMER \${mutable}\`;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture));
  assert.deepEqual(plan.dependencies, [dependency("producer", "consumer", "mutable")]);
  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-producer"] },
    { index: 1, functionIds: ["nf-consumer"] },
  ]);
});

test("follows direct reassignment and property-write provenance conservatively", async (t) => {
  const fixture = await createFixture({
    "writes.sem.ts": `
const producer = sema<boolean>\`PRODUCER\`;
let reassigned = false;
reassigned = producer;
const holder = { value: false };
holder.value = producer;
const direct = sema<boolean>\`DIRECT \${reassigned}\`;
const property = sema<boolean>\`PROPERTY \${holder}\`;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture));
  assert.deepEqual(plan.dependencies, [
    dependency("producer", "direct", "reassigned"),
    dependency("producer", "property", "holder"),
  ]);
  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-producer"] },
    { index: 1, functionIds: ["nf-direct", "nf-property"] },
  ]);
});

test("indexes property writes across source files", async (t) => {
  const fixture = await createFixture({
    "a.sem.ts": `
export const producer = sema<boolean>\`PRODUCER\`;
export const holder = { value: false };
`,
    "b.sem.ts": `
import { holder, producer } from "./a.sem.js";
holder.value = producer;
const consumer = sema<boolean>\`CONSUMER \${holder}\`;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture), {
    sourceFiles: fixture.program.getSourceFiles(),
  });
  assert.deepEqual(plan.dependencies, [dependency("producer", "consumer", "holder")]);
  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-producer"] },
    { index: 1, functionIds: ["nf-consumer"] },
  ]);
});

test("does not memoize provenance pruned by another symbol write", async (t) => {
  const fixture = await createFixture({
    "write-cycle.sem.ts": `
const producer = sema<boolean>\`PRODUCER\`;
let x = false;
let y = false;
x = y;
x = producer;
y = x;
const first = sema<boolean>\`FIRST \${x}\`;
const second = sema<boolean>\`SECOND \${y}\`;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture));
  assert.deepEqual(plan.dependencies, [
    dependency("producer", "first", "x"),
    dependency("producer", "second", "y"),
  ]);
});

test("ignores same-container writes that occur after the consumer", async (t) => {
  const fixture = await createFixture({
    "forward-write.sem.ts": `
let value = false;
const consumer = sema<boolean>\`CONSUMER \${value}\`;
value = consumer;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture));
  assert.deepEqual(plan.dependencies, []);
  assert.deepEqual(plan.stages, [{ index: 0, functionIds: ["nf-consumer"] }]);
});

test("uses checker symbol identity for shadowed bindings", async (t) => {
  const fixture = await createFixture({
    "shadow.sem.ts": `
const result = sema<boolean>\`PRODUCER\`;
{
  const result = false;
  const consumer = sema<boolean>\`CONSUMER \${result}\`;
}
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture));
  assert.deepEqual(plan.dependencies, []);
  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-producer", "nf-consumer"] },
  ]);
});

test("resolves renamed imports across files", async (t) => {
  const fixture = await createFixture({
    "a.sem.ts": "export const producer = sema<boolean>`PRODUCER`;\n",
    "b.sem.ts": `
import { producer as renamed } from "./a.sem.js";
export const consumer = sema<boolean>\`CONSUMER \${renamed}\`;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture).reverse());
  assert.deepEqual(plan.dependencies, [dependency("producer", "consumer", "renamed")]);
  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-producer"] },
    { index: 1, functionIds: ["nf-consumer"] },
  ]);
});

test("orders sites by UTF-8 path bytes regardless of descriptor order", async (t) => {
  const fixture = await createFixture({
    "\uE000.sem.ts": "const bmp = sema<boolean>`BMP`;\n",
    "\u{10000}.sem.ts": "const astral = sema<boolean>`ASTRAL`;\n",
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture).reverse());
  assert.deepEqual(plan.stages, [
    { index: 0, functionIds: ["nf-bmp", "nf-astral"] },
  ]);
});

test("orders dependencies by consumer, consumer input, then producer", async (t) => {
  const fixture = await createFixture({
    "dependency-order.sem.ts": `
const first = sema<boolean>\`FIRST\`;
const second = sema<boolean>\`SECOND\`;
const consumer = sema<boolean>\`CONSUMER \${second} \${first}\`;
`,
  });
  t.after(fixture.dispose);

  const plan = buildSemaExecutionPlan(fixture.checker, collectDescriptors(fixture).reverse());
  assert.deepEqual(plan.dependencies, [
    dependency("second", "consumer", "second"),
    dependency("first", "consumer", "first"),
  ]);
});

test("reports deterministic dependency and self cycles", async (t) => {
  const cycleFixture = await createFixture({
    "cycle.sem.ts": `
const a = sema<boolean>\`A \${b}\`;
const b = sema<boolean>\`B \${a}\`;
`,
  });
  t.after(cycleFixture.dispose);

  assert.throws(
    () => buildSemaExecutionPlan(cycleFixture.checker, collectDescriptors(cycleFixture)),
    (error) =>
      error instanceof ExecutionPlanError &&
      error.code === "SEMA_EXECUTION_PLAN_CYCLE" &&
      error.message === "sema dependency cycle: nf-a -> nf-b -> nf-a",
  );

  const selfFixture = await createFixture({
    "self.sem.ts": "const self = sema<boolean>`SELF ${self}`;\n",
  });
  t.after(selfFixture.dispose);
  assert.throws(
    () => buildSemaExecutionPlan(selfFixture.checker, collectDescriptors(selfFixture)),
    (error) =>
      error instanceof ExecutionPlanError &&
      error.code === "SEMA_EXECUTION_PLAN_CYCLE" &&
      error.message === "sema dependency cycle: nf-self -> nf-self",
  );
});

test("reports stable provenance cycles and bounded traversal failures", async (t) => {
  const aliasFixture = await createFixture({
    "alias-cycle.sem.ts": `
const a = b;
const b = a;
const consumer = sema<boolean>\`CONSUMER \${a}\`;
`,
  });
  t.after(aliasFixture.dispose);
  assert.throws(
    () => buildSemaExecutionPlan(aliasFixture.checker, collectDescriptors(aliasFixture)),
    (error) =>
      error instanceof ExecutionPlanError &&
      error.code === "SEMA_EXECUTION_PLAN_CYCLE" &&
      error.message === "initializer provenance cycle: a -> b -> a",
  );

  const budgetFixture = await createFixture({
    "budget.sem.ts": `
const producer = sema<boolean>\`PRODUCER\`;
const consumer = sema<boolean>\`CONSUMER \${producer}\`;
`,
  });
  t.after(budgetFixture.dispose);
  assert.throws(
    () =>
      buildSemaExecutionPlan(budgetFixture.checker, collectDescriptors(budgetFixture), {
        maximumWork: 1,
      }),
    (error) =>
      error instanceof ExecutionPlanError &&
      error.code === "SEMA_EXECUTION_PLAN_BUDGET" &&
      error.message === "execution-plan provenance exceeds maximum work 1",
  );
});

function dependency(producer, consumer, consumerInput) {
  return {
    producerFunctionId: `nf-${producer}`,
    consumerFunctionId: `nf-${consumer}`,
    consumerInput,
  };
}

async function createFixture(files) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-execution-plan-"));
  const declarationPath = join(root, "globals.d.ts");
  await writeFile(declarationPath, declarations);
  const rootNames = [declarationPath];
  for (const [path, source] of Object.entries(files)) {
    const absolutePath = join(root, path);
    await mkdir(dirname(absolutePath), { recursive: true });
    await writeFile(absolutePath, source);
    rootNames.push(absolutePath);
  }

  const program = ts.createProgram({
    rootNames,
    options: {
      module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext,
      noEmit: true,
      strict: true,
      target: ts.ScriptTarget.ES2022,
    },
  });
  return {
    root,
    program,
    checker: program.getTypeChecker(),
    dispose: () => rm(root, { recursive: true, force: true }),
  };
}

function collectDescriptors(fixture) {
  const descriptors = [];
  for (const sourceFile of fixture.program.getSourceFiles()) {
    if (!sourceFile.fileName.startsWith(fixture.root) || !sourceFile.fileName.endsWith(".sem.ts")) {
      continue;
    }
    const visit = (node) => {
      if (
        ts.isTaggedTemplateExpression(node) &&
        ts.isIdentifier(node.tag) &&
        node.tag.text === "sema"
      ) {
        const markerText = ts.isNoSubstitutionTemplateLiteral(node.template)
          ? node.template.text
          : node.template.head.text;
        const marker = /[A-Z]+/u.exec(markerText)?.[0];
        assert.ok(marker, `missing marker in ${node.getText(sourceFile)}`);
        const inputs = ts.isTemplateExpression(node.template)
          ? node.template.templateSpans.map(({ expression }) => ({
              name: expression.getText(sourceFile),
              expression,
            }))
          : [];
        descriptors.push({
          functionId: `nf-${marker.toLowerCase()}`,
          normalizedSourcePath: relative(fixture.root, sourceFile.fileName).replaceAll("\\", "/"),
          start: node.getStart(sourceFile),
          expression: node,
          inputs,
        });
      }
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(sourceFile, visit);
  }
  return descriptors;
}
