import ts from "typescript";

export const CORE_MODULE = "@semantscript/core";

export function collectCoreExportSymbols(
  program: ts.Program,
  checker: ts.TypeChecker,
  exportNames: readonly string[],
): ReadonlyMap<string, ReadonlySet<ts.Symbol>> {
  const requestedNames = new Set(exportNames);
  const symbolsByName = new Map<string, Set<ts.Symbol>>();

  for (const exportName of requestedNames) {
    symbolsByName.set(exportName, new Set());
  }

  for (const sourceFile of program.getSourceFiles()) {
    for (const statement of sourceFile.statements) {
      if (!hasCoreModuleSpecifier(statement)) {
        continue;
      }

      const moduleSymbol = checker.getSymbolAtLocation(
        statement.moduleSpecifier,
      );

      if (!moduleSymbol) {
        continue;
      }

      for (const exportedSymbol of checker.getExportsOfModule(moduleSymbol)) {
        const exportName = exportedSymbol.getName();
        const symbols = symbolsByName.get(exportName);

        if (symbols) {
          symbols.add(exportedSymbol);
          symbols.add(resolveAliases(exportedSymbol, checker));
        }
      }
    }
  }

  return symbolsByName;
}

export function symbolMatches(
  symbol: ts.Symbol | undefined,
  expected: ReadonlySet<ts.Symbol> | undefined,
  checker: ts.TypeChecker,
): boolean {
  return (
    !!symbol &&
    !!expected &&
    (expected.has(symbol) || expected.has(resolveAliases(symbol, checker)))
  );
}

export function resolveAliases(
  symbol: ts.Symbol,
  checker: ts.TypeChecker,
): ts.Symbol {
  const visited = new Set<ts.Symbol>();
  let current = symbol;

  while (
    (current.flags & ts.SymbolFlags.Alias) !== 0 &&
    !visited.has(current)
  ) {
    visited.add(current);
    const resolved = checker.getAliasedSymbol(current);

    if (resolved === current) {
      break;
    }

    current = resolved;
  }

  return current;
}

function hasCoreModuleSpecifier(statement: ts.Statement): statement is (
  ts.ImportDeclaration | ts.ExportDeclaration
) & {
  readonly moduleSpecifier: ts.StringLiteral;
} {
  if (
    !ts.isImportDeclaration(statement) &&
    !ts.isExportDeclaration(statement)
  ) {
    return false;
  }

  return (
    !!statement.moduleSpecifier &&
    ts.isStringLiteral(statement.moduleSpecifier) &&
    statement.moduleSpecifier.text === CORE_MODULE
  );
}
