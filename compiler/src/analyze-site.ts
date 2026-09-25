import ts from "typescript";

import {
  collectCoreExportSymbols,
  resolveAliases,
  symbolMatches,
} from "./core-symbols.js";
import {
  buildDecimalGrid,
  parseDecimalLexeme,
  parseDecimalTypeNode,
  type DecimalRational,
} from "./decimal.js";
import type {
  HeadSpec,
  InputEntry,
  InputType,
  ObjectInputField,
  OutputSpec,
  TemplatePart,
} from "./ir-types.js";
import {
  bytesToHex,
  compareBytes,
  semanticJsonBytes,
} from "./semantic-json.js";
import { findSemaSites, type SemaSite } from "./sema-sites.js";

const DIAGNOSTIC_CODE = {
  unsupportedOutput: 9101,
  invalidEnum: 9102,
  invalidOrdinal: 9103,
  invalidBounds: 9104,
  invalidFlatOutput: 9105,
  invalidInterpolation: 9110,
  duplicateInterpolation: 9111,
  unsupportedInput: 9112,
} as const;

const TYPE_FORMAT_FLAGS =
  ts.TypeFormatFlags.NoTruncation |
  ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope;
const INPUT_NAME = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

export interface ResolvedSemaSiteIr {
  readonly template: readonly TemplatePart[];
  readonly inputs: readonly InputEntry[];
  readonly output: OutputSpec;
  readonly resultMode: "value" | "diagnostic";
}

export interface SemaSiteAnalysis {
  readonly site: SemaSite;
  readonly ir: ResolvedSemaSiteIr;
}

export interface SemaAnalysisDiagnostic extends ts.DiagnosticWithLocation {
  readonly site: SemaSite;
}

export type AnalyzeSemaSiteResult =
  | { readonly ok: true; readonly value: SemaSiteAnalysis }
  | {
      readonly ok: false;
      readonly diagnostics: readonly SemaAnalysisDiagnostic[];
    };

export interface AnalyzeSemaSitesResult {
  readonly analyses: readonly SemaSiteAnalysis[];
  readonly diagnostics: readonly SemaAnalysisDiagnostic[];
}

interface MarkerApplication {
  readonly kind: "Ordinal" | "BoundedInt" | "BoundedNumber";
  readonly arguments: readonly ts.TypeNode[];
  readonly environment: ReadonlyMap<ts.Symbol, ts.TypeNode>;
}

interface EnumDetails {
  readonly name: string;
  readonly base: "string" | "number";
  readonly values: readonly string[] | readonly number[];
}

interface AnalysisContext {
  readonly checker: ts.TypeChecker;
  readonly markers: MarkerResolver;
  readonly site: SemaSite;
}

class AnalysisFailure extends Error {
  public constructor(
    public readonly code: number,
    public readonly node: ts.Node,
    message: string,
  ) {
    super(message);
  }
}

class MarkerResolver {
  readonly #checker: ts.TypeChecker;
  readonly #symbols: ReadonlyMap<string, ReadonlySet<ts.Symbol>>;

  public constructor(program: ts.Program, checker: ts.TypeChecker) {
    this.#checker = checker;
    this.#symbols = collectCoreExportSymbols(program, checker, [
      "Ordinal",
      "BoundedInt",
      "BoundedNumber",
    ]);
  }

  public find(
    node: ts.TypeNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode> = new Map(),
  ): MarkerApplication | undefined {
    return this.#find(node, environment, new Set());
  }

  public resolveTransparentAlias(
    node: ts.TypeNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  ): ts.TypeNode {
    return this.resolveTransparentAliasWithEnvironment(node, environment).node;
  }

  public resolveTransparentAliasWithEnvironment(
    node: ts.TypeNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode> = new Map(),
  ): {
    readonly node: ts.TypeNode;
    readonly environment: ReadonlyMap<ts.Symbol, ts.TypeNode>;
  } {
    return this.#resolveTransparentAlias(node, environment, new Set());
  }

  #find(
    node: ts.TypeNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
    visited: ReadonlySet<ts.Symbol>,
  ): MarkerApplication | undefined {
    const current = this.#resolveEnvironmentReference(
      unwrapParenthesizedType(node),
      environment,
    );

    if (!ts.isTypeReferenceNode(current)) {
      return undefined;
    }

    const symbol = this.#checker.getSymbolAtLocation(current.typeName);

    for (const kind of ["Ordinal", "BoundedInt", "BoundedNumber"] as const) {
      if (symbolMatches(symbol, this.#symbols.get(kind), this.#checker)) {
        return {
          kind,
          arguments: current.typeArguments ?? [],
          environment,
        };
      }
    }

    if (!symbol) {
      return undefined;
    }

    const resolved = resolveAliases(symbol, this.#checker);

    if (visited.has(resolved)) {
      return undefined;
    }

    const declaration = singleTypeAliasDeclaration(resolved);

    if (!declaration) {
      return undefined;
    }

    const nextEnvironment = bindTypeParameters(
      declaration,
      current.typeArguments ?? [],
      environment,
      this.#checker,
    );
    const nextVisited = new Set(visited);
    nextVisited.add(resolved);
    return this.#find(declaration.type, nextEnvironment, nextVisited);
  }

  #resolveTransparentAlias(
    node: ts.TypeNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
    visited: ReadonlySet<ts.Symbol>,
  ): {
    readonly node: ts.TypeNode;
    readonly environment: ReadonlyMap<ts.Symbol, ts.TypeNode>;
  } {
    const current = this.#resolveEnvironmentReference(
      unwrapParenthesizedType(node),
      environment,
    );

    if (!ts.isTypeReferenceNode(current)) {
      return { node: current, environment };
    }

    const symbol = this.#checker.getSymbolAtLocation(current.typeName);

    if (!symbol) {
      return { node: current, environment };
    }

    const resolved = resolveAliases(symbol, this.#checker);

    if (visited.has(resolved)) {
      return { node: current, environment };
    }

    const declaration = singleTypeAliasDeclaration(resolved);

    if (!declaration) {
      return { node: current, environment };
    }

    const nextEnvironment = bindTypeParameters(
      declaration,
      current.typeArguments ?? [],
      environment,
      this.#checker,
    );
    const nextVisited = new Set(visited);
    nextVisited.add(resolved);
    return this.#resolveTransparentAlias(
      declaration.type,
      nextEnvironment,
      nextVisited,
    );
  }

  #resolveEnvironmentReference(
    node: ts.TypeNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  ): ts.TypeNode {
    let current = node;
    const visited = new Set<ts.Symbol>();

    while (ts.isTypeReferenceNode(current) && !current.typeArguments?.length) {
      const symbol = this.#checker.getSymbolAtLocation(current.typeName);

      if (!symbol || visited.has(symbol)) {
        break;
      }

      const replacement = environment.get(symbol);

      if (!replacement) {
        break;
      }

      visited.add(symbol);
      current = unwrapParenthesizedType(replacement);
    }

    return current;
  }
}

/** Resolve one sema site's output head, inputs, definition and identity into an analysis, or diagnostics. */
export function analyzeSemaSite(
  program: ts.Program,
  site: SemaSite,
): AnalyzeSemaSiteResult {
  const checker = program.getTypeChecker();
  const context: AnalysisContext = {
    checker,
    markers: new MarkerResolver(program, checker),
    site,
  };

  try {
    const { inputs, template } = resolveInputsAndTemplate(context);
    const output = resolveOutput(context);

    return {
      ok: true,
      value: {
        site,
        ir: {
          template,
          inputs,
          output,
          resultMode: site.resultMode,
        },
      },
    };
  } catch (error) {
    if (!(error instanceof AnalysisFailure)) {
      throw error;
    }

    return { ok: false, diagnostics: [toDiagnostic(error, site)] };
  }
}

/** Analyze every discovered sema site of a program in source order. */
export function analyzeSemaSites(
  program: ts.Program,
  sourceFile?: ts.SourceFile,
): AnalyzeSemaSitesResult {
  const analyses: SemaSiteAnalysis[] = [];
  const diagnostics: SemaAnalysisDiagnostic[] = [];

  for (const site of findSemaSites(program, sourceFile)) {
    const result = analyzeSemaSite(program, site);

    if (result.ok) {
      analyses.push(result.value);
    } else {
      diagnostics.push(...result.diagnostics);
    }
  }

  return { analyses, diagnostics };
}

function resolveOutput(context: AnalysisContext): OutputSpec {
  const { checker, site } = context;
  const type = checker.getTypeFromTypeNode(site.outputTypeNode);
  const tsType = typeToString(type, site.outputTypeNode, checker);
  const scalarHead = resolveScalarHead(
    site.outputTypeNode,
    type,
    "output",
    context,
  );

  if (scalarHead) {
    return { kind: "scalar", tsType, head: scalarHead };
  }

  return resolveFlatOutput(type, tsType, context);
}

function resolveScalarHead(
  typeNode: ts.TypeNode,
  type: ts.Type,
  path: string,
  context: AnalysisContext,
  markerEnvironment: ReadonlyMap<ts.Symbol, ts.TypeNode> = new Map(),
): HeadSpec | undefined {
  const marker = context.markers.find(typeNode, markerEnvironment);

  if (marker) {
    return resolveMarkerHead(marker, path, context);
  }

  const enumDeclaration = enumDeclarationForType(type);

  if (enumDeclaration) {
    const details = resolveEnumDetails(enumDeclaration, 2, typeNode, context);

    if (details.base === "string") {
      return {
        kind: "nominal",
        sourceKind: "string-enum",
        support: details.values as readonly string[],
      };
    }

    return {
      kind: "nominal",
      sourceKind: "number-enum",
      support: details.values as readonly number[],
    };
  }

  if ((type.flags & ts.TypeFlags.Boolean) !== 0) {
    return { kind: "nominal", sourceKind: "boolean", support: [false, true] };
  }

  if (type.isUnion()) {
    const values: string[] = [];

    for (const member of type.types) {
      if (
        (member.flags & ts.TypeFlags.StringLiteral) === 0 ||
        (member.flags & ts.TypeFlags.EnumLiteral) !== 0
      ) {
        return undefined;
      }

      const value = (member as ts.StringLiteralType).value;
      assertUnicodeScalarString(
        value,
        typeNode,
        DIAGNOSTIC_CODE.unsupportedOutput,
      );
      values.push(value);
    }

    const support = [...new Set(values)].sort(compareUnicodeScalars);

    if (support.length >= 2) {
      return { kind: "nominal", sourceKind: "string-union", support };
    }
  }

  if (isDefinitelyScalarType(type)) {
    fail(
      DIAGNOSTIC_CODE.unsupportedOutput,
      typeNode,
      `${path} type ${quoteType(type, typeNode, context.checker)} is not a supported finite scalar output; use boolean, a union of at least two string literals, a homogeneous enum, Ordinal, BoundedInt, or BoundedNumber`,
    );
  }

  return undefined;
}

function resolveMarkerHead(
  marker: MarkerApplication,
  path: string,
  context: AnalysisContext,
): HeadSpec {
  if (marker.kind === "Ordinal") {
    if (marker.arguments.length !== 1) {
      fail(
        DIAGNOSTIC_CODE.invalidOrdinal,
        context.site.outputTypeNode,
        `${path} Ordinal marker must have exactly one tuple argument`,
      );
    }

    const argument = marker.arguments[0];

    if (!argument) {
      throw new Error("missing checked Ordinal argument");
    }

    const support = extractOrdinalSupport(
      argument,
      marker.environment,
      context,
    );
    return {
      kind: "ordinal",
      sourceKind: "ordinal-string",
      support,
      expectedValue: "zero-based-rank",
    };
  }

  const expectedArguments = marker.kind === "BoundedInt" ? 2 : 3;

  if (marker.arguments.length !== expectedArguments) {
    fail(
      DIAGNOSTIC_CODE.invalidBounds,
      context.site.outputTypeNode,
      `${path} ${marker.kind} marker must have exactly ${String(expectedArguments)} numeric literal arguments`,
    );
  }

  const decimalArguments = marker.arguments.map((argument) =>
    resolveDecimalArgument(argument, marker.environment, context),
  );
  const minimum = decimalArguments[0];
  const maximum = decimalArguments[1];
  const step =
    marker.kind === "BoundedInt"
      ? parseDecimalLexeme("1")
      : decimalArguments[2];

  if (!minimum || !maximum || !step) {
    throw new Error("missing checked bounded numeric argument");
  }

  const grid = buildDecimalGrid(
    minimum,
    maximum,
    step,
    marker.kind === "BoundedInt",
  );

  if (typeof grid === "string") {
    fail(
      DIAGNOSTIC_CODE.invalidBounds,
      context.site.outputTypeNode,
      `${path} ${marker.kind} is invalid: ${grid}`,
    );
  }

  return {
    kind: "ordinal",
    sourceKind: marker.kind === "BoundedInt" ? "bounded-int" : "bounded-number",
    ...grid,
    expectedValue: "numeric",
  };
}

function resolveFlatOutput(
  type: ts.Type,
  tsType: string,
  context: AnalysisContext,
): OutputSpec {
  const { checker, site } = context;
  const symbol = type.getSymbol();
  const interfaceDeclarations =
    symbol?.declarations?.filter(ts.isInterfaceDeclaration) ?? [];

  if (
    interfaceDeclarations.length === 0 ||
    type.isUnionOrIntersection() ||
    checker.isArrayType(type) ||
    checker.isTupleType(type) ||
    checker.getSignaturesOfType(type, ts.SignatureKind.Call).length > 0 ||
    checker.getSignaturesOfType(type, ts.SignatureKind.Construct).length > 0 ||
    checker.getIndexInfosOfType(type).length > 0
  ) {
    fail(
      DIAGNOSTIC_CODE.unsupportedOutput,
      site.outputTypeNode,
      `output type ${quoteType(type, site.outputTypeNode, checker)} is unsupported; object outputs must be a flat interface of required scalar fields`,
    );
  }

  const properties = checker
    .getPropertiesOfType(type)
    .sort((left, right) =>
      compareUnicodeScalars(left.getName(), right.getName()),
    );
  const outputSyntax = context.markers.resolveTransparentAliasWithEnvironment(
    site.outputTypeNode,
  );
  const interfaceEnvironments = collectInterfaceEnvironments(
    outputSyntax,
    context,
  );

  if (properties.length === 0) {
    fail(
      DIAGNOSTIC_CODE.invalidFlatOutput,
      site.outputTypeNode,
      `output interface ${tsType} must declare at least one field`,
    );
  }

  const fields = properties.map((property) => {
    const declaration = property.valueDeclaration ?? property.declarations?.[0];
    const name = property.getName();

    if (
      !declaration ||
      !ts.isPropertySignature(declaration) ||
      !declaration.type ||
      (property.flags & ts.SymbolFlags.Optional) !== 0 ||
      name.startsWith("__@")
    ) {
      fail(
        DIAGNOSTIC_CODE.invalidFlatOutput,
        declaration ?? site.outputTypeNode,
        `output field ${tsType}.${name} must be a required property with a supported scalar type`,
      );
    }

    const fieldType = checker.getTypeOfSymbolAtLocation(
      property,
      site.outputTypeNode,
    );
    const fieldTypeNode = declaration.type;
    const declaringContainer =
      ts.isInterfaceDeclaration(declaration.parent) ||
      ts.isTypeLiteralNode(declaration.parent)
        ? declaration.parent
        : undefined;
    const fieldMarkerEnvironments = declaringContainer
      ? (interfaceEnvironments.get(declaringContainer) ?? [
          outputSyntax.environment,
        ])
      : [outputSyntax.environment];
    const candidateHeads = fieldMarkerEnvironments.map((environment) =>
      resolveScalarHead(
        fieldTypeNode,
        fieldType,
        `${tsType}.${name}`,
        context,
        environment,
      ),
    );
    const head = candidateHeads[0];

    if (
      head &&
      candidateHeads.some(
        (candidate) =>
          !candidate ||
          bytesToHex(semanticJsonBytes(candidate)) !==
            bytesToHex(semanticJsonBytes(head)),
      )
    ) {
      fail(
        DIAGNOSTIC_CODE.invalidFlatOutput,
        fieldTypeNode,
        `output field ${tsType}.${name} is inherited with conflicting scalar types`,
      );
    }

    if (!head) {
      fail(
        DIAGNOSTIC_CODE.invalidFlatOutput,
        fieldTypeNode,
        `output field ${tsType}.${name} has unsupported nested or non-scalar type ${quoteType(fieldType, declaration.type, checker)}`,
      );
    }

    assertUnicodeScalarString(
      name,
      declaration.name,
      DIAGNOSTIC_CODE.invalidFlatOutput,
    );
    return { name, head };
  });

  return { kind: "object", tsType, fields };
}

function resolveInputsAndTemplate(context: AnalysisContext): {
  readonly inputs: readonly InputEntry[];
  readonly template: readonly TemplatePart[];
} {
  const { checker, site } = context;
  const { template } = site.node;

  if (ts.isNoSubstitutionTemplateLiteral(template)) {
    return { inputs: [], template: [{ kind: "text", text: template.text }] };
  }

  const inputs: InputEntry[] = [];
  const parts: TemplatePart[] = [{ kind: "text", text: template.head.text }];
  const seenNames = new Set<string>();

  for (const [index, span] of template.templateSpans.entries()) {
    const expression = span.expression;

    if (!ts.isIdentifier(expression)) {
      fail(
        DIAGNOSTIC_CODE.invalidInterpolation,
        expression,
        "template interpolations must be bare identifiers; assign this expression to a named const first",
      );
    }

    const name = expression.text;

    if (!INPUT_NAME.test(name) || !isUnicodeScalarString(name)) {
      fail(
        DIAGNOSTIC_CODE.invalidInterpolation,
        expression,
        `interpolation name ${JSON.stringify(name)} is not representable by the v1 IR identifier format`,
      );
    }

    if (seenNames.has(name)) {
      fail(
        DIAGNOSTIC_CODE.duplicateInterpolation,
        expression,
        `interpolation ${name} appears more than once; each input name must be unique within a sema template`,
      );
    }

    seenNames.add(name);
    const type = checker.getTypeAtLocation(expression);
    const inputType = resolveInputType(type, name, context, new Set(), false);
    inputs.push({
      name,
      index,
      tsType: typeToString(type, expression, checker),
      type: inputType,
    });
    parts.push({ kind: "input", name });
    parts.push({ kind: "text", text: span.literal.text });
  }

  return { inputs, template: parts };
}

function resolveInputType(
  type: ts.Type,
  path: string,
  context: AnalysisContext,
  activeTypes: ReadonlySet<ts.Type>,
  allowUndefined: boolean,
): InputType {
  const { checker, site } = context;
  const enumDeclaration = enumDeclarationForType(type);

  if (enumDeclaration) {
    const details = resolveEnumDetails(enumDeclaration, 1, site.node, context);
    return {
      kind: "enum",
      name: details.name,
      base: details.base,
      values: details.values,
    };
  }

  const enumMember = enumMemberDeclarationForType(type);

  if (enumMember) {
    const details = resolveEnumDetails(
      enumMember.parent,
      1,
      site.node,
      context,
    );
    const value = checker.getConstantValue(enumMember);

    if (typeof value !== details.base) {
      fail(
        DIAGNOSTIC_CODE.invalidEnum,
        enumMember,
        `enum member ${enumMember.name.getText()} does not have a finite constant value`,
      );
    }

    return {
      kind: "enum",
      name: details.name,
      base: details.base,
      values: details.base === "string" ? [value as string] : [value as number],
    };
  }

  if (type === checker.getTrueType()) {
    return { kind: "literal", value: true };
  }

  if (type === checker.getFalseType()) {
    return { kind: "literal", value: false };
  }

  if ((type.flags & ts.TypeFlags.StringLiteral) !== 0) {
    const value = (type as ts.StringLiteralType).value;
    assertUnicodeScalarString(
      value,
      site.node,
      DIAGNOSTIC_CODE.unsupportedInput,
    );
    return { kind: "literal", value };
  }

  if ((type.flags & ts.TypeFlags.NumberLiteral) !== 0) {
    const value = (type as ts.NumberLiteralType).value;

    if (!Number.isFinite(value)) {
      fail(
        DIAGNOSTIC_CODE.unsupportedInput,
        site.node,
        `${path} has a non-finite numeric type`,
      );
    }

    return { kind: "literal", value };
  }

  if ((type.flags & ts.TypeFlags.Null) !== 0) {
    return { kind: "null" };
  }

  if ((type.flags & ts.TypeFlags.Boolean) !== 0) {
    return { kind: "boolean" };
  }

  if (
    (type.flags & ts.TypeFlags.String) !== 0 ||
    (type.flags & ts.TypeFlags.TemplateLiteral) !== 0
  ) {
    return { kind: "string" };
  }

  if ((type.flags & ts.TypeFlags.Number) !== 0) {
    return { kind: "number" };
  }

  if (type.isIntersection()) {
    const primitiveMembers = type.types.filter(
      (member) =>
        (member.flags &
          (ts.TypeFlags.StringLike |
            ts.TypeFlags.NumberLike |
            ts.TypeFlags.BooleanLike)) !==
        0,
    );

    if (primitiveMembers.length === 1) {
      const primitiveMember = primitiveMembers[0];

      if (!primitiveMember) {
        throw new Error("missing checked primitive intersection member");
      }

      return resolveInputType(
        primitiveMember,
        path,
        context,
        activeTypes,
        allowUndefined,
      );
    }

    if (
      type.types.every((member) => (member.flags & ts.TypeFlags.Object) !== 0)
    ) {
      return resolveInputObject(type, path, context, activeTypes);
    }
  }

  if (type.isUnion()) {
    const members = allowUndefined
      ? type.types.filter(
          (member) => (member.flags & ts.TypeFlags.Undefined) === 0,
        )
      : type.types;

    if (
      members.length !== type.types.length &&
      (!allowUndefined || members.length === 0)
    ) {
      fail(
        DIAGNOSTIC_CODE.unsupportedInput,
        site.node,
        `${path} contains undefined, which is supported only as the omitted branch of an optional object field`,
      );
    }

    if (
      !allowUndefined &&
      members.some((member) => (member.flags & ts.TypeFlags.Undefined) !== 0)
    ) {
      fail(
        DIAGNOSTIC_CODE.unsupportedInput,
        site.node,
        `${path} contains unsupported undefined`,
      );
    }

    if (members.length === 1) {
      const onlyMember = members[0];

      if (!onlyMember) {
        throw new Error("missing checked input union member");
      }

      return resolveInputType(onlyMember, path, context, activeTypes, false);
    }

    const byEncoding = new Map<
      string,
      { readonly bytes: Uint8Array; readonly value: InputType }
    >();

    for (const member of members) {
      const resolved = resolveInputType(
        member,
        path,
        context,
        activeTypes,
        false,
      );
      const bytes = semanticJsonBytes(resolved);
      byEncoding.set(bytesToHex(bytes), { bytes, value: resolved });
    }

    const variants = [...byEncoding.values()]
      .sort((left, right) => compareBytes(left.bytes, right.bytes))
      .map(({ value }) => value);

    if (variants.length === 1) {
      const onlyVariant = variants[0];

      if (!onlyVariant) {
        throw new Error("missing checked input variant");
      }

      return onlyVariant;
    }

    return { kind: "union", variants };
  }

  if (checker.isTupleType(type)) {
    return withActiveType(
      type,
      path,
      context,
      activeTypes,
      (nextActiveTypes) => {
        const tuple = type as ts.TupleTypeReference;

        if (
          tuple.target.elementFlags.some(
            (flag) =>
              (flag &
                (ts.ElementFlags.Optional |
                  ts.ElementFlags.Rest |
                  ts.ElementFlags.Variadic)) !==
              0,
          )
        ) {
          fail(
            DIAGNOSTIC_CODE.unsupportedInput,
            site.node,
            `${path} uses an optional or rest tuple element, which v1 input schemas do not support`,
          );
        }

        return {
          kind: "tuple",
          items: checker
            .getTypeArguments(tuple)
            .map((item, index) =>
              resolveInputType(
                item,
                `${path}[${String(index)}]`,
                context,
                nextActiveTypes,
                false,
              ),
            ),
        };
      },
    );
  }

  if (checker.isArrayType(type)) {
    return withActiveType(
      type,
      path,
      context,
      activeTypes,
      (nextActiveTypes) => {
        const itemType = checker.getIndexTypeOfType(type, ts.IndexKind.Number);

        if (!itemType) {
          fail(
            DIAGNOSTIC_CODE.unsupportedInput,
            site.node,
            `${path} array element type is unresolved`,
          );
        }

        return {
          kind: "array",
          items: resolveInputType(
            itemType,
            `${path}[]`,
            context,
            nextActiveTypes,
            false,
          ),
        };
      },
    );
  }

  if ((type.flags & ts.TypeFlags.Object) !== 0) {
    return resolveInputObject(type, path, context, activeTypes);
  }

  const constraint = checker.getBaseConstraintOfType(type);

  if (constraint && constraint !== type) {
    return resolveInputType(
      constraint,
      path,
      context,
      activeTypes,
      allowUndefined,
    );
  }

  fail(
    DIAGNOSTIC_CODE.unsupportedInput,
    site.node,
    `${path} has unsupported input type ${quoteType(type, site.node, checker)}; v1 inputs must be serializable primitives, enums, arrays, fixed tuples, objects, or unions of those types`,
  );
}

function resolveInputObject(
  type: ts.Type,
  path: string,
  context: AnalysisContext,
  activeTypes: ReadonlySet<ts.Type>,
): InputType {
  const { checker, site } = context;
  const symbol = type.getSymbol();
  const objectConstituents = type.isIntersection() ? type.types : [type];

  if (
    objectConstituents.some(
      (member) =>
        member.isClass() ||
        member
          .getSymbol()
          ?.declarations?.some(
            (declaration) =>
              ts.isClassDeclaration(declaration) ||
              ts.isClassExpression(declaration),
          ),
    ) ||
    checker.getSignaturesOfType(type, ts.SignatureKind.Call).length > 0 ||
    checker.getSignaturesOfType(type, ts.SignatureKind.Construct).length > 0 ||
    checker.getIndexInfosOfType(type).length > 0
  ) {
    fail(
      DIAGNOSTIC_CODE.unsupportedInput,
      site.node,
      `${path} must be a plain object without class identity, signatures, or index signatures`,
    );
  }

  return withActiveType(type, path, context, activeTypes, (nextActiveTypes) => {
    const fields: ObjectInputField[] = checker
      .getPropertiesOfType(type)
      .sort((left, right) =>
        compareUnicodeScalars(left.getName(), right.getName()),
      )
      .map((property) => {
        const declaration =
          property.valueDeclaration ?? property.declarations?.[0];
        const name = property.getName();

        if (
          !declaration ||
          (!ts.isPropertySignature(declaration) &&
            !ts.isPropertyDeclaration(declaration) &&
            !ts.isPropertyAssignment(declaration) &&
            !ts.isShorthandPropertyAssignment(declaration)) ||
          name.startsWith("__@")
        ) {
          fail(
            DIAGNOSTIC_CODE.unsupportedInput,
            declaration ?? site.node,
            `${path}.${name} must be an ordinary data property`,
          );
        }

        assertUnicodeScalarString(
          name,
          declaration.name,
          DIAGNOSTIC_CODE.unsupportedInput,
        );
        const optional = (property.flags & ts.SymbolFlags.Optional) !== 0;
        const propertyType = checker.getTypeOfSymbolAtLocation(
          property,
          declaration,
        );
        return {
          name,
          optional,
          type: resolveInputType(
            propertyType,
            `${path}.${name}`,
            context,
            nextActiveTypes,
            optional,
          ),
        };
      });
    const symbolName = symbol?.getName();
    const name =
      symbolName && !symbolName.startsWith("__") ? symbolName : "anonymous";
    return { kind: "object", name, fields };
  });
}

function resolveEnumDetails(
  declaration: ts.EnumDeclaration,
  minimumMembers: number,
  anchor: ts.Node,
  context: AnalysisContext,
): EnumDetails {
  const enumSymbol = context.checker.getSymbolAtLocation(declaration.name);
  const enumDeclarations = enumSymbol?.declarations?.filter(
    ts.isEnumDeclaration,
  ) ?? [declaration];

  if (enumDeclarations.length !== 1) {
    fail(
      DIAGNOSTIC_CODE.invalidEnum,
      anchor,
      `enum ${declaration.name.text} is merged across declarations and has no canonical v1 member order`,
    );
  }

  const values: Array<string | number> = [];
  let base: "string" | "number" | undefined;

  for (const member of declaration.members) {
    const value = context.checker.getConstantValue(member);

    if (
      (typeof value !== "string" && typeof value !== "number") ||
      !isFiniteEnumValue(value)
    ) {
      fail(
        DIAGNOSTIC_CODE.invalidEnum,
        member,
        `enum ${declaration.name.text}.${member.name.getText()} is not a finite constant string or number`,
      );
    }

    if (typeof value === "string") {
      assertUnicodeScalarString(value, member, DIAGNOSTIC_CODE.invalidEnum);
    }

    const memberBase: "string" | "number" =
      typeof value === "string" ? "string" : "number";

    if (base && base !== memberBase) {
      fail(
        DIAGNOSTIC_CODE.invalidEnum,
        member,
        `enum ${declaration.name.text} is heterogeneous; all members must share one primitive base`,
      );
    }

    base = memberBase;
    values.push(value);
  }

  if (!base || values.length < minimumMembers) {
    fail(
      DIAGNOSTIC_CODE.invalidEnum,
      anchor,
      `enum ${declaration.name.text} must contain at least ${String(minimumMembers)} constant member${minimumMembers === 1 ? "" : "s"}`,
    );
  }

  const uniqueKeys = new Set(
    values.map((value) => `${typeof value}:${String(value)}`),
  );

  if (uniqueKeys.size !== values.length) {
    fail(
      DIAGNOSTIC_CODE.invalidEnum,
      anchor,
      `enum ${declaration.name.text} contains duplicate values`,
    );
  }

  return {
    name: declaration.name.text,
    base,
    values:
      base === "string"
        ? (values as readonly string[])
        : (values as readonly number[]),
  };
}

function extractOrdinalSupport(
  node: ts.TypeNode,
  environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  context: AnalysisContext,
): readonly string[] {
  let current = context.markers.resolveTransparentAlias(node, environment);

  if (
    ts.isTypeOperatorNode(current) &&
    current.operator === ts.SyntaxKind.ReadonlyKeyword
  ) {
    current = context.markers.resolveTransparentAlias(
      current.type,
      environment,
    );
  }

  if (!ts.isTupleTypeNode(current)) {
    fail(
      DIAGNOSTIC_CODE.invalidOrdinal,
      node,
      "Ordinal requires a fixed tuple of at least two distinct string literal types",
    );
  }

  const support = current.elements.map((element) => {
    if (ts.isNamedTupleMember(element)) {
      if (element.questionToken || element.dotDotDotToken) {
        fail(
          DIAGNOSTIC_CODE.invalidOrdinal,
          element,
          "Ordinal tuples cannot contain optional or rest elements",
        );
      }

      element = element.type;
    }

    const resolved = context.markers.resolveTransparentAlias(
      element,
      environment,
    );

    if (
      !ts.isLiteralTypeNode(resolved) ||
      !ts.isStringLiteral(resolved.literal)
    ) {
      fail(
        DIAGNOSTIC_CODE.invalidOrdinal,
        element,
        "every Ordinal tuple member must be a string literal type",
      );
    }

    assertUnicodeScalarString(
      resolved.literal.text,
      resolved,
      DIAGNOSTIC_CODE.invalidOrdinal,
    );
    return resolved.literal.text;
  });

  if (support.length < 2 || new Set(support).size !== support.length) {
    fail(
      DIAGNOSTIC_CODE.invalidOrdinal,
      current,
      "Ordinal requires at least two distinct string literal values",
    );
  }

  return support;
}

function resolveDecimalArgument(
  node: ts.TypeNode,
  environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  context: AnalysisContext,
): DecimalRational {
  const resolved = context.markers.resolveTransparentAlias(node, environment);
  const value = parseDecimalTypeNode(resolved);

  if (!value) {
    fail(
      DIAGNOSTIC_CODE.invalidBounds,
      node,
      "bounded numeric marker arguments must be decimal numeric literal types",
    );
  }

  return value;
}

function enumDeclarationForType(type: ts.Type): ts.EnumDeclaration | undefined {
  const declarations = type.getSymbol()?.declarations;

  if (!declarations) {
    return undefined;
  }

  for (const declaration of declarations) {
    if (ts.isEnumDeclaration(declaration)) {
      return declaration;
    }
  }

  return undefined;
}

function enumMemberDeclarationForType(
  type: ts.Type,
): ts.EnumMember | undefined {
  return type.getSymbol()?.declarations?.find(ts.isEnumMember);
}

function bindTypeParameters(
  declaration: ts.TypeAliasDeclaration | ts.InterfaceDeclaration,
  arguments_: readonly ts.TypeNode[],
  environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  checker: ts.TypeChecker,
): ReadonlyMap<ts.Symbol, ts.TypeNode> {
  const next = new Map(environment);

  for (const [index, parameter] of (
    declaration.typeParameters ?? []
  ).entries()) {
    const argument = arguments_[index] ?? parameter.default;
    const symbol = checker.getSymbolAtLocation(parameter.name);

    if (argument && symbol) {
      next.set(symbol, argument);
    }
  }

  return next;
}

function collectInterfaceEnvironments(
  root: {
    readonly node: ts.TypeNode;
    readonly environment: ReadonlyMap<ts.Symbol, ts.TypeNode>;
  },
  context: AnalysisContext,
): ReadonlyMap<
  ts.InterfaceDeclaration | ts.TypeLiteralNode,
  readonly ReadonlyMap<ts.Symbol, ts.TypeNode>[]
> {
  const environments = new Map<
    ts.InterfaceDeclaration | ts.TypeLiteralNode,
    ReadonlyMap<ts.Symbol, ts.TypeNode>[]
  >();
  const activeAliases = new Set<ts.Symbol>();
  const activeInterfaces = new Set<ts.InterfaceDeclaration>();

  function recordEnvironment(
    declaration: ts.InterfaceDeclaration | ts.TypeLiteralNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  ): void {
    const existing = environments.get(declaration);

    if (existing) {
      existing.push(environment);
    } else {
      environments.set(declaration, [environment]);
    }
  }

  function visitTypeNode(
    node: ts.TypeNode,
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  ): void {
    const target = context.markers.resolveTransparentAliasWithEnvironment(
      node,
      environment,
    );

    if (ts.isTypeReferenceNode(target.node)) {
      const symbol = context.checker.getSymbolAtLocation(target.node.typeName);

      if (symbol) {
        visitSymbol(
          symbol,
          target.node.typeArguments ?? [],
          target.environment,
        );
      }

      return;
    }

    if (ts.isIntersectionTypeNode(target.node)) {
      for (const constituent of target.node.types) {
        visitTypeNode(constituent, target.environment);
      }

      return;
    }

    if (ts.isTypeLiteralNode(target.node)) {
      recordEnvironment(target.node, target.environment);
    }
  }

  function visitSymbol(
    symbol: ts.Symbol,
    arguments_: readonly ts.TypeNode[],
    environment: ReadonlyMap<ts.Symbol, ts.TypeNode>,
  ): void {
    const resolvedSymbol = resolveAliases(symbol, context.checker);
    const aliasDeclarations =
      resolvedSymbol.declarations?.filter(ts.isTypeAliasDeclaration) ?? [];

    if (aliasDeclarations.length === 1) {
      if (activeAliases.has(resolvedSymbol)) {
        fail(
          DIAGNOSTIC_CODE.invalidFlatOutput,
          context.site.outputTypeNode,
          `output interface inheritance contains a recursive type alias ${resolvedSymbol.getName()}`,
        );
      }

      const aliasDeclaration = aliasDeclarations[0];

      if (!aliasDeclaration) {
        throw new Error("missing checked type alias declaration");
      }

      const aliasEnvironment = bindTypeParameters(
        aliasDeclaration,
        arguments_,
        environment,
        context.checker,
      );
      activeAliases.add(resolvedSymbol);
      const target = context.markers.resolveTransparentAliasWithEnvironment(
        aliasDeclaration.type,
        aliasEnvironment,
      );
      visitTypeNode(target.node, target.environment);

      activeAliases.delete(resolvedSymbol);
      return;
    }

    const declarations =
      resolvedSymbol.declarations?.filter(ts.isInterfaceDeclaration) ?? [];

    for (const declaration of declarations) {
      if (activeInterfaces.has(declaration)) {
        fail(
          DIAGNOSTIC_CODE.invalidFlatOutput,
          context.site.outputTypeNode,
          `output interface inheritance is recursive at ${declaration.name.text}`,
        );
      }

      const boundEnvironment = bindTypeParameters(
        declaration,
        arguments_,
        environment,
        context.checker,
      );
      recordEnvironment(declaration, boundEnvironment);
      activeInterfaces.add(declaration);

      for (const clause of declaration.heritageClauses ?? []) {
        if (clause.token !== ts.SyntaxKind.ExtendsKeyword) {
          continue;
        }

        for (const heritageType of clause.types) {
          const heritageSymbol = context.checker.getSymbolAtLocation(
            heritageType.expression,
          );

          if (heritageSymbol) {
            visitSymbol(
              heritageSymbol,
              heritageType.typeArguments ?? [],
              boundEnvironment,
            );
          }
        }
      }

      activeInterfaces.delete(declaration);
    }
  }

  visitTypeNode(root.node, root.environment);
  return environments;
}

function singleTypeAliasDeclaration(
  symbol: ts.Symbol,
): ts.TypeAliasDeclaration | undefined {
  const declarations =
    symbol.declarations?.filter(ts.isTypeAliasDeclaration) ?? [];
  return declarations.length === 1 ? declarations[0] : undefined;
}

function withActiveType<T>(
  type: ts.Type,
  path: string,
  context: AnalysisContext,
  activeTypes: ReadonlySet<ts.Type>,
  resolve: (nextActiveTypes: ReadonlySet<ts.Type>) => T,
): T {
  if (activeTypes.has(type)) {
    fail(
      DIAGNOSTIC_CODE.unsupportedInput,
      context.site.node,
      `${path} is recursive; v1 input schemas must be finite`,
    );
  }

  const next = new Set(activeTypes);
  next.add(type);
  return resolve(next);
}

function typeToString(
  type: ts.Type,
  node: ts.Node,
  checker: ts.TypeChecker,
): string {
  return checker.typeToString(type, node, TYPE_FORMAT_FLAGS);
}

function quoteType(
  type: ts.Type,
  node: ts.Node,
  checker: ts.TypeChecker,
): string {
  return JSON.stringify(typeToString(type, node, checker));
}

function isDefinitelyScalarType(type: ts.Type): boolean {
  const flags = type.flags;
  return (
    (flags &
      (ts.TypeFlags.Any |
        ts.TypeFlags.Unknown |
        ts.TypeFlags.Never |
        ts.TypeFlags.Void |
        ts.TypeFlags.Undefined |
        ts.TypeFlags.Null |
        ts.TypeFlags.StringLike |
        ts.TypeFlags.NumberLike |
        ts.TypeFlags.BigIntLike |
        ts.TypeFlags.ESSymbolLike)) !==
      0 || type.isUnion()
  );
}

function isFiniteEnumValue(value: string | number): boolean {
  return typeof value === "string" || Number.isFinite(value);
}

function compareUnicodeScalars(left: string, right: string): number {
  const leftCodePoints = Array.from(
    left,
    (character) => character.codePointAt(0) ?? 0,
  );
  const rightCodePoints = Array.from(
    right,
    (character) => character.codePointAt(0) ?? 0,
  );
  const length = Math.min(leftCodePoints.length, rightCodePoints.length);

  for (let index = 0; index < length; index += 1) {
    const difference =
      (leftCodePoints[index] ?? 0) - (rightCodePoints[index] ?? 0);

    if (difference !== 0) {
      return difference;
    }
  }

  return leftCodePoints.length - rightCodePoints.length;
}

function assertUnicodeScalarString(
  value: string,
  node: ts.Node,
  code: number,
): void {
  if (!isUnicodeScalarString(value)) {
    fail(code, node, "v1 IR strings cannot contain unpaired UTF-16 surrogates");
  }
}

function isUnicodeScalarString(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);

    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);

      if (next < 0xdc00 || next > 0xdfff) {
        return false;
      }

      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }

  return true;
}

function toDiagnostic(
  failure: AnalysisFailure,
  site: SemaSite,
): SemaAnalysisDiagnostic {
  const anchor =
    failure.node.getSourceFile() === site.sourceFile
      ? failure.node
      : site.outputTypeNode;
  const start = anchor.getStart(site.sourceFile);
  const siteText = site.node.getText(site.sourceFile);
  const prefix = `${site.location.fileName}:${String(site.location.line)}:${String(site.location.column)}`;

  return {
    category: ts.DiagnosticCategory.Error,
    code: failure.code,
    file: site.sourceFile,
    start,
    length: anchor.getEnd() - start,
    messageText: `${prefix}: ${failure.message} (site: ${siteText})`,
    site,
  };
}

function fail(code: number, node: ts.Node, message: string): never {
  throw new AnalysisFailure(code, node, message);
}

function unwrapParenthesizedType(node: ts.TypeNode): ts.TypeNode {
  let current = node;

  while (ts.isParenthesizedTypeNode(current)) {
    current = current.type;
  }

  return current;
}
