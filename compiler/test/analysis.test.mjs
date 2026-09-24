import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import ts from "typescript";

import { analyzeSemaSites, createSourceNeuralFunctionIr } from "../dist/index.js";

const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const schema = JSON.parse(
  await readFile(join(repositoryRoot, "schemas", "neural-function.v1.schema.json"), "utf8"),
);
const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
addFormats(ajv);
const validateIr = ajv.compile(schema);

const coreDeclarations = `
declare const ordinalKind: unique symbol;
declare const boundedIntKind: unique symbol;
declare const boundedNumberKind: unique symbol;
export type Ordinal<V extends readonly [string, string, ...string[]]> =
  V[number] & { readonly [ordinalKind]?: V };
export type BoundedInt<Min extends number, Max extends number> =
  number & { readonly [boundedIntKind]?: readonly [Min, Max] };
export type BoundedNumber<Min extends number, Max extends number, Step extends number> =
  number & { readonly [boundedNumberKind]?: readonly [Min, Max, Step] };
interface ConfiguredSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): T;
}
interface DiagnosticSemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): { value: T };
  <T>(options: unknown): ConfiguredSemaTag<{ value: T }>;
}
interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: unknown): ConfiguredSemaTag<T>;
  readonly withConfidence: DiagnosticSemaTag;
}
export declare const sema: SemaTag;
`;

test("resolves nominal, ordinal, bounded, and flat-interface output heads", async (t) => {
  const fixture = await createFixtureProgram({
    "outputs.sem.ts": `import {
  sema,
  type BoundedInt,
  type BoundedNumber,
  type Ordinal,
} from "@semantscript/core";

enum TextChoice { Zed = "z", First = "a" }
enum NumberChoice { Shifted = 1 << 3, Next = Shifted + 1 }
type UnicodeChoice = "😀" | "\uE000";
type Ordered<V extends readonly [string, string, ...string[]]> = Ordinal<V>;
type Ranked<V extends readonly [string, string, ...string[]]> = Ordered<V>;
type Severity = Ranked<["low", "medium", "high"]>;
type Range<Min extends number, Max extends number> = BoundedInt<Min, Max>;
type Stars = Range<-2, 2>;
type Probability = BoundedNumber<-0.5, 0.5, 0.25>;
type LocalOrdinal<V extends readonly string[]> = V[number];

interface Triage {
  urgency: Severity;
  route: "self-service" | "agent";
  readonly abusive: boolean;
  score: BoundedInt<0, 2>;
}
interface Box<T> { value: T }
type Boxed = Box<BoundedNumber<0, 1, 0.5>>;
interface Base<T> { inherited: T }
interface Derived extends Base<BoundedInt<1, 3>> { ok: boolean }
type BaseAlias<T> = Base<T>;
interface ViaAlias extends BaseAlias<BoundedInt<0, 2>> { ready: boolean }
interface Extra { extra: boolean }
type Combined<T> = Base<T> & Extra;
interface ViaCombined extends Combined<BoundedInt<-1, 1>> { combined: boolean }
type LiteralBase<T> = { literal: T };
interface ViaLiteral extends LiteralBase<Ordinal<["first", "last"]>> { done: boolean }
type SharedLiteral<T> = { shared: T };
type WrapperA<U> = SharedLiteral<U>;
type WrapperB<V> = SharedLiteral<V>;
interface ViaRepeatedLiteral extends WrapperA<BoundedInt<0, 2>>, WrapperB<BoundedInt<0, 2>> { repeat: boolean }
interface ViaReorderedUnion extends WrapperA<"a" | "b">, WrapperB<"b" | "a"> { union: boolean }
interface ViaBooleanUnion extends WrapperA<boolean>, WrapperB<true | false> { booleanUnion: boolean }

const booleanResult = sema<boolean>\`boolean\`;
const unionResult = sema<UnicodeChoice>\`union\`;
const stringEnumResult = sema<TextChoice>\`string enum\`;
const numberEnumResult = sema<NumberChoice>\`number enum\`;
const ordinalResult = sema<Severity>\`ordinal\`;
const boundedIntResult = sema<Stars>\`bounded int\`;
const boundedNumberResult = sema<Probability>\`bounded number\`;
const objectResult = sema<Triage>\`object\`;
const lookalikeResult = sema<LocalOrdinal<["z", "a"]>>\`lookalike\`;
const boxedResult = sema<Boxed>\`generic object\`;
const inheritedResult = sema<Derived>\`inherited generic object\`;
const aliasInheritedResult = sema<ViaAlias>\`alias-inherited generic object\`;
const combinedInheritedResult = sema<ViaCombined>\`intersection-alias inheritance\`;
const literalInheritedResult = sema<ViaLiteral>\`type-literal alias inheritance\`;
const repeatedLiteralResult = sema<ViaRepeatedLiteral>\`equivalent wrapper inheritance\`;
const reorderedUnionResult = sema<ViaReorderedUnion>\`equivalent reordered union inheritance\`;
const booleanUnionResult = sema<ViaBooleanUnion>\`equivalent boolean union inheritance\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = analyzeSemaSites(fixture.program);
  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.analyses.length, 17);

  const outputs = result.analyses.map(({ ir }) => ir.output);
  assert.deepEqual(outputs[0], {
    kind: "scalar",
    tsType: "boolean",
    head: { kind: "nominal", sourceKind: "boolean", support: [false, true] },
  });
  assert.deepEqual(outputs[1].head, {
    kind: "nominal",
    sourceKind: "string-union",
    support: ["\uE000", "😀"],
  });
  assert.deepEqual(outputs[2].head, {
    kind: "nominal",
    sourceKind: "string-enum",
    support: ["z", "a"],
  });
  assert.deepEqual(outputs[3].head, {
    kind: "nominal",
    sourceKind: "number-enum",
    support: [8, 9],
  });
  assert.deepEqual(outputs[4].head, {
    kind: "ordinal",
    sourceKind: "ordinal-string",
    support: ["low", "medium", "high"],
    expectedValue: "zero-based-rank",
  });
  assert.deepEqual(outputs[5].head, {
    kind: "ordinal",
    sourceKind: "bounded-int",
    minimum: "-2",
    maximum: "2",
    step: "1",
    supportDecimal: ["-2", "-1", "0", "1", "2"],
    expectedValue: "numeric",
  });
  assert.deepEqual(outputs[6].head, {
    kind: "ordinal",
    sourceKind: "bounded-number",
    minimum: "-0.5",
    maximum: "0.5",
    step: "0.25",
    supportDecimal: ["-0.5", "-0.25", "0", "0.25", "0.5"],
    expectedValue: "numeric",
  });
  assert.deepEqual(
    outputs[7].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["abusive", "boolean"],
      ["route", "string-union"],
      ["score", "bounded-int"],
      ["urgency", "ordinal-string"],
    ],
  );
  assert.deepEqual(outputs[8].head, {
    kind: "nominal",
    sourceKind: "string-union",
    support: ["a", "z"],
  });
  assert.deepEqual(outputs[9].fields, [
    {
      name: "value",
      head: {
        kind: "ordinal",
        sourceKind: "bounded-number",
        minimum: "0",
        maximum: "1",
        step: "0.5",
        supportDecimal: ["0", "0.5", "1"],
        expectedValue: "numeric",
      },
    },
  ]);
  assert.deepEqual(
    outputs[10].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["inherited", "bounded-int"],
      ["ok", "boolean"],
    ],
  );
  assert.deepEqual(
    outputs[11].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["inherited", "bounded-int"],
      ["ready", "boolean"],
    ],
  );
  assert.deepEqual(
    outputs[12].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["combined", "boolean"],
      ["extra", "boolean"],
      ["inherited", "bounded-int"],
    ],
  );
  assert.deepEqual(
    outputs[13].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["done", "boolean"],
      ["literal", "ordinal-string"],
    ],
  );
  assert.deepEqual(
    outputs[14].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["repeat", "boolean"],
      ["shared", "bounded-int"],
    ],
  );
  assert.deepEqual(
    outputs[15].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["shared", "string-union"],
      ["union", "boolean"],
    ],
  );
  assert.deepEqual(
    outputs[16].fields.map(({ name, head }) => [name, head.sourceKind]),
    [
      ["booleanUnion", "boolean"],
      ["shared", "boolean"],
    ],
  );
});

test("resolves every interpolation into deterministic recursive input IR", async (t) => {
  const fixture = await createFixtureProgram({
    "inputs.sem.ts": `import {
  sema,
  type BoundedInt,
  type Ordinal,
} from "@semantscript/core";

enum Tone { Warm = "warm", Cool = "cool" }
interface Payload {
  z?: string;
  a: readonly [number, boolean];
  tags: readonly string[];
  choice: "x" | null;
}

declare const text: string;
declare const flag: boolean;
declare const count: number;
declare const none: null;
declare const exact: "fixed";
declare const tone: Tone;
declare const payload: Payload;
declare const rank: Ordinal<["low", "high"]>;
declare const score: BoundedInt<0, 5>;
declare const member: Tone.Warm;
declare const mixed: readonly string[] | { x: string };
const inferred = { b: 1, a: "value" };
declare const intersection: { left: string } & { right: number };

const decision = sema<"yes" | "no">\`Text: \${text}; flag: \${flag}; count: \${count}; none: \${none}; exact: \${exact}; tone: \${tone}; payload: \${payload}; rank: \${rank}; score: \${score}; member: \${member}; mixed: \${mixed}; inferred: \${inferred}; intersection: \${intersection}.\`;
`,
  });
  t.after(fixture.dispose);
  assert.deepEqual(ts.getPreEmitDiagnostics(fixture.program), []);

  const result = analyzeSemaSites(fixture.program);
  assert.deepEqual(result.diagnostics, []);
  const analysis = result.analyses[0];
  assert.ok(analysis);
  assert.deepEqual(analysis.ir.inputs, [
    { name: "text", index: 0, tsType: "string", type: { kind: "string" } },
    { name: "flag", index: 1, tsType: "boolean", type: { kind: "boolean" } },
    { name: "count", index: 2, tsType: "number", type: { kind: "number" } },
    { name: "none", index: 3, tsType: "null", type: { kind: "null" } },
    {
      name: "exact",
      index: 4,
      tsType: '"fixed"',
      type: { kind: "literal", value: "fixed" },
    },
    {
      name: "tone",
      index: 5,
      tsType: "Tone",
      type: { kind: "enum", name: "Tone", base: "string", values: ["warm", "cool"] },
    },
    {
      name: "payload",
      index: 6,
      tsType: "Payload",
      type: {
        kind: "object",
        name: "Payload",
        fields: [
          {
            name: "a",
            optional: false,
            type: {
              kind: "tuple",
              items: [{ kind: "number" }, { kind: "boolean" }],
            },
          },
          {
            name: "choice",
            optional: false,
            type: {
              kind: "union",
              variants: [
                { kind: "literal", value: "x" },
                { kind: "null" },
              ],
            },
          },
          {
            name: "tags",
            optional: false,
            type: { kind: "array", items: { kind: "string" } },
          },
          { name: "z", optional: true, type: { kind: "string" } },
        ],
      },
    },
    {
      name: "rank",
      index: 7,
      tsType: 'Ordinal<["low", "high"]>',
      type: {
        kind: "union",
        variants: [
          { kind: "literal", value: "high" },
          { kind: "literal", value: "low" },
        ],
      },
    },
    {
      name: "score",
      index: 8,
      tsType: "BoundedInt<0, 5>",
      type: { kind: "number" },
    },
    {
      name: "member",
      index: 9,
      tsType: "Tone.Warm",
      type: { kind: "enum", name: "Tone", base: "string", values: ["warm"] },
    },
    {
      name: "mixed",
      index: 10,
      tsType: "readonly string[] | { x: string; }",
      type: {
        kind: "union",
        variants: [
          {
            kind: "object",
            name: "anonymous",
            fields: [{ name: "x", optional: false, type: { kind: "string" } }],
          },
          { kind: "array", items: { kind: "string" } },
        ],
      },
    },
    {
      name: "inferred",
      index: 11,
      tsType: "{ a: string; b: number; }",
      type: {
        kind: "object",
        name: "anonymous",
        fields: [
          { name: "a", optional: false, type: { kind: "string" } },
          { name: "b", optional: false, type: { kind: "number" } },
        ],
      },
    },
    {
      name: "intersection",
      index: 12,
      tsType: "{ left: string; } & { right: number; }",
      type: {
        kind: "object",
        name: "anonymous",
        fields: [
          { name: "left", optional: false, type: { kind: "string" } },
          { name: "right", optional: false, type: { kind: "number" } },
        ],
      },
    },
  ]);
  assert.deepEqual(
    analysis.ir.template.filter(({ kind }) => kind === "input").map(({ name }) => name),
    [
      "text",
      "flag",
      "count",
      "none",
      "exact",
      "tone",
      "payload",
      "rank",
      "score",
      "member",
      "mixed",
      "inferred",
      "intersection",
    ],
  );
  assert.equal(analysis.ir.template.length, 27);

});

test("reports located errors for unsupported outputs and invalid interpolations", async (t) => {
  const fixture = await createFixtureProgram({
    "invalid.sem.ts": `import { sema, type BoundedNumber } from "@semantscript/core";

interface Nested { child: { flag: boolean } }
enum Choice { A = "a", B = "b" }
type OnlyChoice = Choice.A;
declare const item: { value: string };
declare const duplicate: string;

const freeText = sema<string>\`free text\`;
const nested = sema<Nested>\`nested\`;
const array = sema<boolean[]>\`array\`;
const badGrid = sema<BoundedNumber<0, 1, 0.3>>\`grid\`;
const expression = sema<"yes" | "no">\`value \${item.value}\`;
const repeated = sema<"yes" | "no">\`first \${duplicate}, second \${duplicate}\`;
const enumMember = sema<OnlyChoice>\`member\`;
`,
  });
  t.after(fixture.dispose);

  const result = analyzeSemaSites(fixture.program);
  assert.equal(result.analyses.length, 0);
  assert.deepEqual(
    result.diagnostics.map(({ code }) => code),
    [9101, 9105, 9101, 9104, 9110, 9111, 9101],
  );

  for (const diagnostic of result.diagnostics) {
    assert.match(String(diagnostic.messageText), /invalid\.sem\.ts:\d+:\d+:/);
    assert.match(String(diagnostic.messageText), /site: sema</);
    assert.equal(diagnostic.category, ts.DiagnosticCategory.Error);
  }
});

test("production source-stage IR records validate against the committed schema", async (t) => {
  const fixture = await createFixtureProgram({
    "schema.sem.ts": `import { sema, type BoundedInt } from "@semantscript/core";
interface Result { "": boolean; accepted: boolean; score: BoundedInt<0, 2> }
declare const message: string;
declare const payload: { "": string };
const result = sema.withConfidence<Result>\`Judge: \${message}; payload: \${payload}\`;
`,
  });
  t.after(fixture.dispose);

  const result = analyzeSemaSites(fixture.program);
  assert.deepEqual(result.diagnostics, []);
  const analysis = result.analyses[0];
  assert.ok(analysis);
  const ir = createSourceNeuralFunctionIr(analysis, {
    id: `nf_${"0".repeat(64)}`,
    semanticSha256: "1".repeat(64),
    sourcePath: "schema.sem.ts",
    sourceSha256: "2".repeat(64),
    encoderRef: "encoder.main",
    adapterRef: "adapter.application",
    headRefs: ["head.site.empty", "head.site.accepted", "head.site.score"],
  });
  const roundTripped = JSON.parse(JSON.stringify(ir));

  assert.equal(
    validateIr(roundTripped),
    true,
    ajv.errorsText(validateIr.errors, { separator: "\n" }),
  );
  assert.equal(roundTripped.runtime.resultMode, "diagnostic");
  assert.equal(roundTripped.inputs[1].type.fields[0].name, "");
  assert.equal(roundTripped.output.fields[0].name, "");
  // A flat interface output compiles to exactly one head per field, in field order.
  assert.deepEqual(
    roundTripped.output.fields.map(({ name }) => name),
    ["", "accepted", "score"],
  );
  assert.deepEqual(roundTripped.model.heads, [
    { outputPath: "/", ref: "head.site.empty" },
    { outputPath: "/accepted", ref: "head.site.accepted" },
    { outputPath: "/score", ref: "head.site.score" },
  ]);
});

async function createFixtureProgram(files) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-analysis-"));
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
    },
  });

  return {
    program,
    async dispose() {
      await rm(root, { force: true, recursive: true });
    },
  };
}
