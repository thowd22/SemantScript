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

export function isSemantScriptSourceFile(sourceFile: ts.SourceFile): boolean {
  return !sourceFile.isDeclarationFile && sourceFile.fileName.endsWith(".sem.ts");
}

export function findSemaSites(
  program: ts.Program,
  sourceFile?: ts.SourceFile,
): readonly SemaSite[] {
  const checker = program.getTypeChecker();
  const coreSemaSymbols = collectCoreExportSymbols(program, checker, ["sema"]).get("sema");

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
          ? classifySemaExpression(parsed.callableExpression, checker, coreSemaSymbols)
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

function parseSemaTag(node: ts.TaggedTemplateExpression): ParsedTag | undefined {
  if (ts.isOptionalChain(node.tag)) {
    return undefined;
  }

  const configured = ts.isCallExpression(node.tag);
  const tagExpression = configured ? node.tag.expression : node.tag;
  const typeArguments = configured ? node.tag.typeArguments : node.typeArguments;

  if (typeArguments?.length !== 1) {
    return undefined;
  }

  const outputTypeNode = typeArguments[0];

  if (!outputTypeNode) {
    return undefined;
  }

  const options = configured && node.tag.arguments.length === 1 ? node.tag.arguments[0] : undefined;

  if (configured && !options) {
    return undefined;
  }

  if (!ts.isIdentifier(tagExpression) && !ts.isPropertyAccessExpression(tagExpression)) {
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
    expressionReferencesCoreSema(expression.expression, checker, coreSemaSymbols)
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
