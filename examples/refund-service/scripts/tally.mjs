// Tally the service's source lines by category so the walkthrough can say how
// much logic moved into sema expressions and what stayed deterministic. Walks
// the TypeScript AST of src/*.ts and gives every non-blank, non-comment line
// the category of the innermost node that has one (rules in `categorize`);
// lines no node claims take the file's default. Prints a Markdown table.
//
//   node scripts/tally.mjs [--lines]
import { readdir, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const source = join(dirname(fileURLToPath(import.meta.url)), "..", "src");
const files = (await readdir(source))
  .filter((name) => name.endsWith(".ts"))
  .sort();

export const categories = {
  "sema: policy text": "the natural-language policy inside a sema template",
  "sema: constraints": "always/never predicates that bound the policy",
  "sema: gold examples":
    "attested input/output pairs the verifier must reproduce",
  "sema: declaration":
    "the sema call, its output type and the function around it",
  "decision glue":
    "calling a sema function with plain values and reading its result",
  types: "interfaces, type aliases and row shapes shared by both sides",
  persistence: "SQL, schema, seed rows and row-to-value mapping",
  transactions: "transactional, gate and rollback",
  validation: "require guards on request input and lookups",
  routing: "controllers, decorators, responses, Express wiring and startup",
  "imports and exports": "module plumbing",
};
const semaFunctions = new Set();

function calleeText(node) {
  return ts.isCallExpression(node) ? node.expression.getText() : "";
}

function categorize(node, fileName) {
  if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node))
    return "imports and exports";
  if (ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node))
    return "types";
  if (
    ts.isTaggedTemplateExpression(node) &&
    /\bsema\b/.test(node.tag.getText())
  )
    return "sema: declaration";
  if (
    ts.isTemplateLiteral(node) &&
    ts.isTaggedTemplateExpression(node.parent) &&
    /\bsema\b/.test(node.parent.tag.getText())
  )
    return "sema: policy text";
  if (ts.isPropertyAssignment(node) && node.name.getText() === "constraints")
    return "sema: constraints";
  if (ts.isPropertyAssignment(node) && node.name.getText() === "examples")
    return "sema: gold examples";
  if (fileName.endsWith(".sem.ts")) return undefined;
  const callee = calleeText(node);
  if (callee === "require") return "validation";
  if (
    /^(tx|db)\.(query|exec)$/.test(callee) ||
    /^(new PGlite)/.test(node.getText())
  )
    return "persistence";
  if (ts.isFunctionDeclaration(node) && node.name?.getText() === "migrate")
    return "persistence";
  if (/^(transactional|gate|rollback)$/.test(callee)) return "transactions";
  if (semaFunctions.has(callee)) return "decision glue";
  if (
    ts.isVariableStatement(node) &&
    [...semaFunctions].some((name) =>
      new RegExp(`\\b${name}\\(`).test(node.getText()),
    )
  )
    return "decision glue";
  if (
    ts.isVariableStatement(node) &&
    /\brows?\b/.test(node.getText()) &&
    !/\bsema\b/.test(node.getText())
  )
    return "persistence";
  if (
    ts.isReturnStatement(node) &&
    node.expression &&
    ts.isObjectLiteralExpression(node.expression)
  )
    return "routing";
  if (ts.isDecorator(node) || ts.isClassDeclaration(node)) return "routing";
  return undefined;
}

const totals = Object.fromEntries(
  Object.keys(categories).map((key) => [key, 0]),
);
const perFile = [];
const sources = [];
for (const name of files) {
  const text = await readFile(join(source, name), "utf8");
  const file = ts.createSourceFile(name, text, ts.ScriptTarget.ES2022, true);
  sources.push([name, file]);
  if (name.endsWith(".sem.ts")) {
    for (const statement of file.statements) {
      if (ts.isFunctionDeclaration(statement) && statement.name)
        semaFunctions.add(statement.name.getText());
    }
  }
}

for (const [name, file] of sources) {
  const lines = file.text.split("\n");
  const fallback = name.endsWith(".sem.ts")
    ? "sema: declaration"
    : name === "db.ts"
      ? "persistence"
      : "routing";
  const lineCategory = lines.map(() => undefined);
  const claim = (node, category) => {
    const start = file.getLineAndCharacterOfPosition(node.getStart()).line;
    const end = file.getLineAndCharacterOfPosition(node.getEnd()).line;
    for (let line = start; line <= end; line += 1)
      lineCategory[line] = category;
  };
  const visit = (node) => {
    const category = categorize(node, name);
    if (category !== undefined) claim(node, category);
    ts.forEachChild(node, visit);
  };
  visit(file);
  const counts = Object.fromEntries(
    Object.keys(categories).map((key) => [key, 0]),
  );
  lines.forEach((raw, index) => {
    const line = raw.trim();
    if (
      line === "" ||
      line.startsWith("//") ||
      line.startsWith("/*") ||
      line.startsWith("*")
    )
      return;
    const category = lineCategory[index] ?? fallback;
    counts[category] += 1;
    totals[category] += 1;
    if (process.argv.includes("--lines"))
      process.stdout.write(`${name}:${index + 1}\t${category}\t${line}\n`);
  });
  perFile.push({ name, counts });
}

const total = Object.values(totals).reduce((sum, value) => sum + value, 0);
const semaTotal = [
  "sema: policy text",
  "sema: constraints",
  "sema: gold examples",
  "sema: declaration",
].reduce((sum, key) => sum + totals[key], 0);
const rows = [
  "| Category | Lines | Share | By file |",
  "| --- | ---: | ---: | --- |",
];
for (const [key, description] of Object.entries(categories)) {
  const owners = perFile
    .filter((entry) => entry.counts[key] > 0)
    .map((entry) => `${entry.name} ${entry.counts[key]}`)
    .join(", ");
  rows.push(
    `| ${key}: ${description} | ${totals[key]} | ${((100 * totals[key]) / total).toFixed(0)}% | ${owners} |`,
  );
}
rows.push(
  `| total | ${total} | 100% | sema expressions ${semaTotal} (${((100 * semaTotal) / total).toFixed(0)}%) |`,
);
process.stdout.write(`${rows.join("\n")}\n`);
