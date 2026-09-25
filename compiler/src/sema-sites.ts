import ts from "typescript";

import {
  collectCoreExportSymbols,
  resolveAliases,
  symbolMatches,
} from "./core-symbols.js";

export interface SemaSourceLocation {
  readonly fileName: string;
  readonly start: number;
  readonly end: number;
  readonly line: number;
  readonly column: number;
}

export interface SemaSite {
  readonly sourceFile: ts.SourceFile;
  readonly node: ts.TaggedTemplateExpression;
  readonly outputTypeNode: ts.TypeNode;
  readonly options: ts.Expression | undefined;
  readonly resultMode: "value" | "diagnostic";
  readonly configured: boolean;
  readonly location: SemaSourceLocation;
}

interface ParsedTag {
  readonly callableExpression: ts.Expression;
  readonly outputTypeNode: ts.TypeNode;
  readonly options: ts.Expression | undefined;
  readonly configured: boolean;
}

/**
 * A tagged template whose tag is the core `sema` (or `sema.withConfidence`) but
 * which is not a canonical site: it would otherwise stay an untransformed
 * runtime tag, so the compiler reports it instead of ignoring it.
 */
export interface MalformedSemaSite {
  readonly sourceFile: ts.SourceFile;
  readonly node: ts.TaggedTemplateExpression;
  readonly location: SemaSourceLocation;
  readonly reason: string;
}

/** Whether a TypeScript source file is a `.sem.ts` file the compiler analyzes. */
export function isSemantScriptSourceFile(sourceFile: ts.SourceFile): boolean {
  return (
    !sourceFile.isDeclarationFile && sourceFile.fileName.endsWith(".sem.ts")
  );
}

/** Find every canonical `sema<T>` tagged template of a program (or one file) in source order, matched by import identity. */
export function findSemaSites(
  program: ts.Program,
  sourceFile?: ts.SourceFile,
): readonly SemaSite[] {
  const checker = program.getTypeChecker();
  const coreSemaSymbols = collectCoreExportSymbols(program, checker, [
    "sema",
  ]).get("sema");

  if (!coreSemaSymbols || coreSemaSymbols.size === 0) {
    return [];
  }

  const sourceFiles = sourceFile
    ? [sourceFile]
    : program
        .getSourceFiles()
        .filter(isSemantScriptSourceFile)
        .sort(compareSourceFiles);
  const sites: SemaSite[] = [];

  for (const currentSourceFile of sourceFiles) {
    if (!isSemantScriptSourceFile(currentSourceFile)) {
      continue;
    }

    const visit = (node: ts.Node): void => {
      if (ts.isTaggedTemplateExpression(node)) {
        const parsed = parseSemaTag(node);
        const resultMode = parsed
          ? classifySemaExpression(
              parsed.callableExpression,
              checker,
              coreSemaSymbols,
            )
          : undefined;

        if (parsed && resultMode) {
          sites.push({
            sourceFile: currentSourceFile,
            node,
            outputTypeNode: parsed.outputTypeNode,
            options: parsed.options,
            resultMode,
            configured: parsed.configured,
            location: sourceLocation(currentSourceFile, node),
          });
        }
      }

      ts.forEachChild(node, visit);
    };

    visit(currentSourceFile);
  }

  return sites;
}

/** Every core `sema` tagged template that `findSemaSites` cannot accept, with why. */
export function findMalformedSemaSites(
  program: ts.Program,
  sourceFile?: ts.SourceFile,
): readonly MalformedSemaSite[] {
  const checker = program.getTypeChecker();
  const coreSemaSymbols = collectCoreExportSymbols(program, checker, [
    "sema",
  ]).get("sema");

  if (!coreSemaSymbols || coreSemaSymbols.size === 0) {
    return [];
  }

  const sourceFiles = sourceFile
    ? [sourceFile]
    : program
        .getSourceFiles()
        .filter(isSemantScriptSourceFile)
        .sort(compareSourceFiles);
  const malformed: MalformedSemaSite[] = [];

  for (const currentSourceFile of sourceFiles) {
    if (!isSemantScriptSourceFile(currentSourceFile)) {
      continue;
    }

    const visit = (node: ts.Node): void => {
      if (
        ts.isTaggedTemplateExpression(node) &&
        parseSemaTag(node) === undefined
      ) {
        const reason = describeMalformedTag(node, checker, coreSemaSymbols);

        if (reason !== undefined) {
          malformed.push({
            sourceFile: currentSourceFile,
            node,
            location: sourceLocation(currentSourceFile, node),
            reason,
          });
        }
      }

      ts.forEachChild(node, visit);
    };

    visit(currentSourceFile);
  }

  return malformed;
}

function describeMalformedTag(
  node: ts.TaggedTemplateExpression,
  checker: ts.TypeChecker,
  coreSemaSymbols: ReadonlySet<ts.Symbol>,
): string | undefined {
  const configured = ts.isCallExpression(node.tag);
  const tagExpression = configured ? node.tag.expression : node.tag;

  if (
    !ts.isIdentifier(tagExpression) &&
    !ts.isPropertyAccessExpression(tagExpression)
  ) {
    return undefined;
  }

  const resultMode = classifySemaExpression(
    tagExpression,
    checker,
    coreSemaSymbols,
  );

  if (resultMode === undefined) {
    return undefined;
  }

  const form = resultMode === "diagnostic" ? "sema.withConfidence" : "sema";

  if (ts.isOptionalChain(node.tag)) {
    return `${form} cannot be used through optional chaining`;
  }

  const typeArguments = configured
    ? node.tag.typeArguments
    : node.typeArguments;
  const typeArgumentCount = typeArguments?.length ?? 0;

  if (typeArgumentCount !== 1) {
    return typeArgumentCount === 0
      ? `${form} requires an explicit output type argument, for example ${form}<"yes" | "no">\`...\``
      : `${form} takes exactly one output type argument (found ${String(typeArgumentCount)})`;
  }

  if (configured && node.tag.arguments.length !== 1) {
    return `${form}({ examples, constraints }) takes exactly one options argument (found ${String(node.tag.arguments.length)})`;
  }

  return `${form} is not used as a canonical tagged template`;
}

function parseSemaTag(
  node: ts.TaggedTemplateExpression,
): ParsedTag | undefined {
  if (ts.isOptionalChain(node.tag)) {
    return undefined;
  }

  const configured = ts.isCallExpression(node.tag);
  const tagExpression = configured ? node.tag.expression : node.tag;
  const typeArguments = configured
    ? node.tag.typeArguments
    : node.typeArguments;

  if (typeArguments?.length !== 1) {
    return undefined;
  }

  const outputTypeNode = typeArguments[0];

  if (!outputTypeNode) {
    return undefined;
  }

  const options =
    configured && node.tag.arguments.length === 1
      ? node.tag.arguments[0]
      : undefined;

  if (configured && !options) {
    return undefined;
  }

  if (
    !ts.isIdentifier(tagExpression) &&
    !ts.isPropertyAccessExpression(tagExpression)
  ) {
    return undefined;
  }

  return {
    callableExpression: tagExpression,
    outputTypeNode,
    options,
    configured,
  };
}

function classifySemaExpression(
  expression: ts.Expression,
  checker: ts.TypeChecker,
  coreSemaSymbols: ReadonlySet<ts.Symbol>,
): "value" | "diagnostic" | undefined {
  if (expressionReferencesCoreSema(expression, checker, coreSemaSymbols)) {
    return "value";
  }

  if (
    ts.isPropertyAccessExpression(expression) &&
    expression.name.text === "withConfidence" &&
    expressionReferencesCoreSema(
      expression.expression,
      checker,
      coreSemaSymbols,
    )
  ) {
    return "diagnostic";
  }

  return undefined;
}

function expressionReferencesCoreSema(
  expression: ts.Expression,
  checker: ts.TypeChecker,
  coreSemaSymbols: ReadonlySet<ts.Symbol>,
): boolean {
  const symbol = symbolForExpression(expression, checker);

  if (!symbol) {
    return false;
  }

  if (!symbolMatches(symbol, coreSemaSymbols, checker)) {
    return false;
  }

  return (
    !ts.isPropertyAccessExpression(expression) ||
    expressionResolvesToModule(expression.expression, checker)
  );
}

function expressionResolvesToModule(
  expression: ts.Expression,
  checker: ts.TypeChecker,
): boolean {
  const symbol = symbolForExpression(expression, checker);

  if (!symbol) {
    return false;
  }

  return (resolveAliases(symbol, checker).flags & ts.SymbolFlags.Module) !== 0;
}

function symbolForExpression(
  expression: ts.Expression,
  checker: ts.TypeChecker,
): ts.Symbol | undefined {
  if (ts.isIdentifier(expression)) {
    return checker.getSymbolAtLocation(expression);
  }

  if (ts.isPropertyAccessExpression(expression)) {
    return checker.getSymbolAtLocation(expression.name);
  }

  return undefined;
}

function sourceLocation(
  sourceFile: ts.SourceFile,
  node: ts.TaggedTemplateExpression,
): SemaSourceLocation {
  const start = node.getStart(sourceFile);
  const lineAndCharacter = sourceFile.getLineAndCharacterOfPosition(start);

  return {
    fileName: sourceFile.fileName,
    start,
    end: node.getEnd(),
    line: lineAndCharacter.line + 1,
    column: lineAndCharacter.character + 1,
  };
}

function compareSourceFiles(left: ts.SourceFile, right: ts.SourceFile): number {
  if (left.fileName < right.fileName) {
    return -1;
  }

  if (left.fileName > right.fileName) {
    return 1;
  }

  return 0;
}
