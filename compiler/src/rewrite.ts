import ts from "typescript";

const RUNTIME_MODULE = "@semantscript/core";
const RUNTIME_EXPORT = "__sema";
const FUNCTION_ID = /^nf_[a-f0-9]{64}$/;
const INPUT_NAME = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

export interface PlannedSemaRewrite {
  readonly sourceFile: ts.SourceFile;
  readonly start: number;
  readonly end: number;
  readonly functionId: string;
  readonly inputNames: readonly string[];
}

/**
 * Creates the SemantScript transformer that must run first in the `before`
 * transformer list passed to TypeScript emit.
 */
export function createFirstBeforeSemaRewriteTransformer(
  plannedSites: readonly PlannedSemaRewrite[],
): ts.TransformerFactory<ts.SourceFile> {
  const sitesByFile = indexPlannedSites(plannedSites);

  return (context) => {
    const { factory } = context;

    return (sourceFile) => {
      const sites = sitesByFile.get(sourceFile.fileName);

      if (!sites || sites.size === 0) {
        return sourceFile;
      }

      const runtimeIdentifier = factory.createUniqueName(
        RUNTIME_EXPORT,
        ts.GeneratedIdentifierFlags.Optimistic |
          ts.GeneratedIdentifierFlags.FileLevel |
          ts.GeneratedIdentifierFlags.ReservedInNestedScopes,
      );
      let replacementCount = 0;

      const visitor: ts.Visitor = (node) => {
        if (ts.isTaggedTemplateExpression(node)) {
          const site = sites.get(
            siteKey(node.getStart(sourceFile), node.getEnd()),
          );

          if (site) {
            replacementCount += 1;
            return createRuntimeCall(
              factory,
              runtimeIdentifier,
              node,
              site,
              sourceFile,
            );
          }
        }

        return ts.visitEachChild(node, visitor, context);
      };

      const transformed = ts.visitEachChild(sourceFile, visitor, context);

      if (replacementCount !== sites.size) {
        throw new RangeError(
          `planned ${String(sites.size)} sema rewrites for ${sourceFile.fileName}, but matched ${String(replacementCount)}`,
        );
      }

      const runtimeImport = createRuntimeImport(factory, runtimeIdentifier);
      const statements = [...transformed.statements];
      statements.splice(importInsertionIndex(statements), 0, runtimeImport);
      return factory.updateSourceFile(transformed, statements);
    };
  };
}

function indexPlannedSites(
  plannedSites: readonly PlannedSemaRewrite[],
): ReadonlyMap<string, ReadonlyMap<string, PlannedSemaRewrite>> {
  const sitesByFile = new Map<string, Map<string, PlannedSemaRewrite>>();

  for (const site of plannedSites) {
    validatePlannedSite(site);
    const fileName = site.sourceFile.fileName;
    let sites = sitesByFile.get(fileName);

    if (!sites) {
      sites = new Map();
      sitesByFile.set(fileName, sites);
    }

    const key = siteKey(site.start, site.end);

    if (sites.has(key)) {
      throw new RangeError(
        `duplicate planned sema rewrite at ${fileName}:${key}`,
      );
    }

    sites.set(key, site);
  }

  return sitesByFile;
}

function validatePlannedSite(site: PlannedSemaRewrite): void {
  if (
    !Number.isSafeInteger(site.start) ||
    !Number.isSafeInteger(site.end) ||
    site.start < 0 ||
    site.end <= site.start
  ) {
    throw new RangeError("planned sema rewrite must have a valid source range");
  }

  if (!FUNCTION_ID.test(site.functionId)) {
    throw new TypeError(
      "planned sema rewrite function id must match nf_<64 lowercase hexadecimal characters>",
    );
  }

  const seenInputNames = new Set<string>();

  for (const inputName of site.inputNames) {
    if (!INPUT_NAME.test(inputName)) {
      throw new TypeError(
        `invalid planned sema input name ${JSON.stringify(inputName)}`,
      );
    }

    if (seenInputNames.has(inputName)) {
      throw new TypeError(
        `duplicate planned sema input name ${JSON.stringify(inputName)}`,
      );
    }

    seenInputNames.add(inputName);
  }
}

function createRuntimeCall(
  factory: ts.NodeFactory,
  runtimeIdentifier: ts.Identifier,
  node: ts.TaggedTemplateExpression,
  site: PlannedSemaRewrite,
  sourceFile: ts.SourceFile,
): ts.CallExpression {
  const inputExpressions = ts.isTemplateExpression(node.template)
    ? node.template.templateSpans.map(({ expression }) => expression)
    : [];

  if (inputExpressions.length !== site.inputNames.length) {
    throw new RangeError(
      `planned sema rewrite at ${sourceFile.fileName}:${String(site.start)} has ${String(site.inputNames.length)} input names for ${String(inputExpressions.length)} interpolations`,
    );
  }

  const properties = site.inputNames.map((inputName, index) => {
    const property = factory.createShorthandPropertyAssignment(inputName);
    const original = inputExpressions[index];

    if (original) {
      ts.setOriginalNode(property, original);
      ts.setTextRange(property, original);
      ts.setSourceMapRange(property, {
        pos: original.getStart(sourceFile),
        end: original.getEnd(),
        source: sourceFile,
      });
    }

    return property;
  });
  const call = factory.createCallExpression(
    factory.createPropertyAccessExpression(runtimeIdentifier, "call"),
    undefined,
    [
      factory.createStringLiteral(site.functionId),
      factory.createObjectLiteralExpression(properties),
    ],
  );

  ts.setOriginalNode(call, node);
  ts.setTextRange(call, node);
  ts.setSourceMapRange(call, {
    pos: site.start,
    end: site.end,
    source: sourceFile,
  });
  return call;
}

function createRuntimeImport(
  factory: ts.NodeFactory,
  runtimeIdentifier: ts.Identifier,
): ts.ImportDeclaration {
  const declaration = factory.createImportDeclaration(
    undefined,
    factory.createImportClause(
      undefined,
      undefined,
      factory.createNamedImports([
        factory.createImportSpecifier(
          false,
          factory.createIdentifier(RUNTIME_EXPORT),
          runtimeIdentifier,
        ),
      ]),
    ),
    factory.createStringLiteral(RUNTIME_MODULE),
  );

  ts.setEmitFlags(declaration, ts.EmitFlags.NoSourceMap);
  return declaration;
}

function importInsertionIndex(statements: readonly ts.Statement[]): number {
  let index = 0;

  while (index < statements.length && isDirective(statements[index])) {
    index += 1;
  }

  while (index < statements.length) {
    const statement = statements[index];

    if (
      !statement ||
      (!ts.isImportDeclaration(statement) &&
        !ts.isImportEqualsDeclaration(statement))
    ) {
      break;
    }

    index += 1;
  }

  return index;
}

function isDirective(statement: ts.Statement | undefined): boolean {
  return (
    !!statement &&
    ts.isExpressionStatement(statement) &&
    ts.isStringLiteral(statement.expression)
  );
}

function siteKey(start: number, end: number): string {
  return `${String(start)}:${String(end)}`;
}
