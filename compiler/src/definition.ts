import ts from "typescript";

import { collectCoreExportSymbols, symbolMatches } from "./core-symbols.js";
import type { HeadSpec, InputEntry, InputType, OutputSpec, TemplatePart } from "./ir-types.js";
import { semanticJsonString, type JsonValue } from "./semantic-json.js";
import type { SemaSite } from "./sema-sites.js";

const DIAGNOSTIC_CODE = {
  invalidOptions: 9120,
  invalidExample: 9121,
  invalidConstraint: 9122,
  invalidConfidence: 9123,
  emptyBehavior: 9124,
  contradictoryConstraint: 9125,
} as const;

const CONFIDENCE_LINE =
  /^\s*@confidence\(\s*(0(?:\.[0-9]+)?|1(?:\.0+)?)\s*\)\s*$/;
const UNKNOWN_CONSTANT = Symbol("unknown-constraint-constant");
const MAX_DEFINITION_DEPTH = 100;
const MAX_DEFINITION_NODES = 100_000;
const MAX_DIAGNOSTIC_SITE_CHARACTERS = 200;

export type ConstraintExpression =
  | { readonly node: "literal"; readonly value: string | number | boolean | null }
  | { readonly node: "input"; readonly name: string }
  | {
      readonly node: "property";
      readonly object: ConstraintExpression;
      readonly property: string;
    }
  | {
      readonly node: "index";
      readonly object: ConstraintExpression;
      readonly index: ConstraintExpression;
    }
  | {
      readonly node: "unary";
      readonly operator: "!" | "+" | "-";
      readonly operand: ConstraintExpression;
    }
  | {
      readonly node: "binary";
      readonly operator:
        | "==="
        | "!=="
        | "<"
        | "<="
        | ">"
        | ">="
        | "&&"
        | "||"
        | "+"
        | "-"
        | "*"
        | "/"
        | "%"
        | "**";
      readonly left: ConstraintExpression;
      readonly right: ConstraintExpression;
    };

type ConstraintBinaryOperator = Extract<
  ConstraintExpression,
  { readonly node: "binary" }
>["operator"];

export interface ResolvedExample {
  readonly inputs: { readonly [name: string]: JsonValue };
  readonly output: JsonValue;
}

export interface ResolvedConstraint {
  readonly kind: "always" | "never";
  readonly source: string;
  readonly predicate: ConstraintExpression;
  readonly output: JsonValue;
}

export interface ResolvedDefinition {
  readonly template: readonly TemplatePart[];
  readonly examples: readonly ResolvedExample[];
  readonly constraints: readonly ResolvedConstraint[];
}

export interface ResolvedDefinitionConfiguration {
  readonly definition: ResolvedDefinition;
  readonly confidenceThreshold: number | null;
}

export interface DefinitionDiagnostic extends ts.DiagnosticWithLocation {
  readonly site: SemaSite;
}

export type ResolveDefinitionConfigurationResult =
  | { readonly ok: true; readonly value: ResolvedDefinitionConfiguration }
  | { readonly ok: false; readonly diagnostics: readonly DefinitionDiagnostic[] };

interface DefinitionContext {
  readonly checker: ts.TypeChecker;
  readonly program: ts.Program;
  readonly site: SemaSite;
  readonly inputs: readonly InputEntry[];
  readonly output: OutputSpec;
  readonly inputTypes: ReadonlyMap<string, ts.Type>;
  readonly outputType: ts.Type;
  readonly constraintSymbols: ReadonlyMap<string, ReadonlySet<ts.Symbol>>;
  readonly constraintScalarTypes: Map<ts.Type, boolean>;
  readonly constraintJsonDataTypes: Map<ts.Type, boolean>;
  readonly budget: { remaining: number };
}

interface ParsedConstraint {
  readonly node: ts.CallExpression;
  readonly value: ResolvedConstraint;
  readonly overlapKey: string | undefined;
}

class DefinitionFailure extends Error {
  public constructor(
    public readonly code: number,
    public readonly node: ts.Node,
    message: string,
  ) {
    super(message);
  }
}

class DefinitionLimitFailure extends DefinitionFailure {}

export function resolveDefinitionConfiguration(
  program: ts.Program,
  site: SemaSite,
  inputs: readonly InputEntry[],
  output: OutputSpec,
): ResolveDefinitionConfigurationResult {
  const checker = program.getTypeChecker();
  const context: DefinitionContext = {
    checker,
    program,
    site,
    inputs,
    output,
    inputTypes: inputTypesForSite(site, checker),
    outputType: checker.getTypeFromTypeNode(site.outputTypeNode),
    constraintSymbols: collectCoreExportSymbols(program, checker, ["always", "never"]),
    constraintScalarTypes: new Map(),
    constraintJsonDataTypes: new Map(),
    budget: { remaining: MAX_DEFINITION_NODES },
  };

  try {
    const parsedTemplate = parseTemplate(site, inputs);
    const { confidenceThreshold, template } = parseConfidence(parsedTemplate, site);
    const { examples, constraints } = parseOptions(context);

    return {
      ok: true,
      value: {
        definition: { template, examples, constraints },
        confidenceThreshold,
      },
    };
  } catch (error) {
    if (!(error instanceof DefinitionFailure)) {
      throw error;
    }

    return { ok: false, diagnostics: [toDiagnostic(error, site)] };
  }
}

function parseOptions(context: DefinitionContext): {
  readonly examples: readonly ResolvedExample[];
  readonly constraints: readonly ResolvedConstraint[];
} {
  const options = context.site.options;

  if (!options) {
    return { examples: [], constraints: [] };
  }

  if (!ts.isObjectLiteralExpression(options)) {
    fail(
      DIAGNOSTIC_CODE.invalidOptions,
      options,
      "sema options must be an inline object literal containing only examples and constraints",
    );
  }

  const properties = objectProperties(options, "sema options", DIAGNOSTIC_CODE.invalidOptions);

  for (const name of properties.keys()) {
    if (name !== "examples" && name !== "constraints") {
      fail(
        DIAGNOSTIC_CODE.invalidOptions,
        properties.get(name)?.node ?? options,
        `unknown sema option ${JSON.stringify(name)}; only examples and constraints are supported`,
      );
    }
  }

  const examplesProperty = properties.get("examples");
  const constraintsProperty = properties.get("constraints");
  return {
    examples: examplesProperty ? parseExamples(examplesProperty.expression, context) : [],
    constraints: constraintsProperty
      ? parseConstraints(constraintsProperty.expression, context)
      : [],
  };
}

function parseExamples(
  expression: ts.Expression,
  context: DefinitionContext,
): readonly ResolvedExample[] {
  const array = unwrapStaticExpression(expression);

  if (!ts.isArrayLiteralExpression(array)) {
    fail(
      DIAGNOSTIC_CODE.invalidExample,
      expression,
      "examples must be an inline array literal",
    );
  }

  const examples = array.elements.map((element) => parseExample(element, context));
  const outputsByInputs = new Map<string, string>();

  for (const [index, example] of examples.entries()) {
    const inputsKey = semanticJsonString(example.inputs);
    const outputKey = semanticJsonString(example.output);
    const previousOutput = outputsByInputs.get(inputsKey);

    if (previousOutput !== undefined && previousOutput !== outputKey) {
      fail(
        DIAGNOSTIC_CODE.invalidExample,
        array.elements[index] ?? array,
        "duplicate example inputs cannot have different outputs",
      );
    }

    outputsByInputs.set(inputsKey, outputKey);
  }

  return examples;
}

function parseExample(expression: ts.Expression, context: DefinitionContext): ResolvedExample {
  const example = unwrapStaticExpression(expression);

  if (!ts.isObjectLiteralExpression(example)) {
    fail(
      DIAGNOSTIC_CODE.invalidExample,
      expression,
      "each example must be an inline object literal with inputs and output",
    );
  }

  const properties = objectProperties(example, "example", DIAGNOSTIC_CODE.invalidExample);

  if (properties.size !== 2 || !properties.has("inputs") || !properties.has("output")) {
    fail(
      DIAGNOSTIC_CODE.invalidExample,
      example,
      "each example must contain exactly the properties inputs and output",
    );
  }

  const inputsProperty = properties.get("inputs");
  const outputProperty = properties.get("output");

  if (!inputsProperty || !outputProperty) {
    throw new Error("missing checked example properties");
  }

  const inputObject = unwrapStaticExpression(inputsProperty.expression);

  if (!ts.isObjectLiteralExpression(inputObject)) {
    fail(
      DIAGNOSTIC_CODE.invalidExample,
      inputsProperty.expression,
      "example inputs must be an inline object literal",
    );
  }

  const inputProperties = objectProperties(
    inputObject,
    "example inputs",
    DIAGNOSTIC_CODE.invalidExample,
  );
  const expectedNames = new Set(context.inputs.map(({ name }) => name));

  for (const name of inputProperties.keys()) {
    if (!expectedNames.has(name)) {
      fail(
        DIAGNOSTIC_CODE.invalidExample,
        inputProperties.get(name)?.node ?? inputObject,
        `example contains unknown input ${JSON.stringify(name)}`,
      );
    }
  }

  const inputValues: Record<string, JsonValue> = {};

  for (const input of context.inputs) {
    const property = inputProperties.get(input.name);

    if (!property) {
      fail(
        DIAGNOSTIC_CODE.invalidExample,
        inputObject,
        `example is missing input ${JSON.stringify(input.name)}`,
      );
    }

    const targetType = context.inputTypes.get(input.name);

    if (!targetType) {
      throw new Error(`missing resolved TypeScript input type for ${input.name}`);
    }

    // `SemaExample.inputs` is intentionally `Record<string, unknown>`, so nested
    // object literals are widened by TypeScript before they reach this compiler.
    // Recursively contextualize literal members against the target while still
    // enforcing nominal properties such as brands, then validate the exact value
    // against the resolved IR schema.
    assertStaticExampleInputAssignable(property.expression, targetType, context);
    const value = evaluateStaticValue(
      property.expression,
      context,
      new Set(),
      DIAGNOSTIC_CODE.invalidExample,
    );
    validateInputValue(value, input.type, property.expression, context.budget);
    defineJsonProperty(inputValues, input.name, value);
  }

  assertAssignable(
    outputProperty.expression,
    context.outputType,
    context,
    DIAGNOSTIC_CODE.invalidExample,
  );
  const outputValue = evaluateStaticValue(
    outputProperty.expression,
    context,
    new Set(),
    DIAGNOSTIC_CODE.invalidExample,
  );
  validateOutputValue(
    outputValue,
    context.output,
    outputProperty.expression,
    DIAGNOSTIC_CODE.invalidExample,
  );
  return { inputs: inputValues, output: outputValue };
}

function parseConstraints(
  expression: ts.Expression,
  context: DefinitionContext,
): readonly ResolvedConstraint[] {
  if (context.output.kind !== "scalar") {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      expression,
      "constraints on a complete flat-interface output are not supported in v1",
    );
  }

  const array = unwrapStaticExpression(expression);

  if (!ts.isArrayLiteralExpression(array)) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      expression,
      "constraints must be an inline array literal",
    );
  }

  const constraints = array.elements.map((element) => parseConstraint(element, context));
  validateConstraintContradictions(constraints);
  return constraints.map(({ value }) => value);
}

function parseConstraint(
  expression: ts.Expression,
  context: DefinitionContext,
): ParsedConstraint {
  const candidate = unwrapStaticExpression(expression);

  if (!ts.isCallExpression(candidate) || candidate.questionDotToken) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      expression,
      "each constraint must be a direct call to always(...) or never(...)",
    );
  }

  if (candidate.arguments.length !== 2 || candidate.typeArguments?.length) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      candidate,
      "always and never constraints require exactly a predicate and an output value",
    );
  }

  const symbol = symbolForCallable(candidate.expression, context.checker);
  const kind = symbolMatches(symbol, context.constraintSymbols.get("always"), context.checker)
    ? "always"
    : symbolMatches(symbol, context.constraintSymbols.get("never"), context.checker)
      ? "never"
      : undefined;

  if (!kind) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      candidate.expression,
      "constraint constructors must resolve to always or never from @semantscript/core",
    );
  }

  const predicateArgument = candidate.arguments[0];
  const outputArgument = candidate.arguments[1];

  if (!predicateArgument || !outputArgument) {
    throw new Error("missing checked constraint arguments");
  }

  if (
    !ts.isArrowFunction(predicateArgument) ||
    predicateArgument.modifiers?.some(
      (modifier) => modifier.kind === ts.SyntaxKind.AsyncKeyword,
    ) ||
    predicateArgument.parameters.length !== 0 ||
    ts.isBlock(predicateArgument.body)
  ) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      predicateArgument,
      "a constraint predicate must be a synchronous zero-argument arrow with an expression body",
    );
  }

  const booleanType = context.checker.getBooleanType();
  assertDefinitelyPresentAssignable(
    predicateArgument.body,
    booleanType,
    context,
    DIAGNOSTIC_CODE.invalidConstraint,
  );
  const predicate = parseConstraintExpression(predicateArgument.body, context);
  assertAssignable(
    outputArgument,
    context.outputType,
    context,
    DIAGNOSTIC_CODE.invalidConstraint,
  );
  const output = evaluateStaticValue(
    outputArgument,
    context,
    new Set(),
    DIAGNOSTIC_CODE.invalidConstraint,
  );
  validateOutputValue(
    output,
    context.output,
    outputArgument,
    DIAGNOSTIC_CODE.invalidConstraint,
  );
  const source = predicateArgument.body.getText(context.site.sourceFile);

  if (!source || !isUnicodeScalarString(source)) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      predicateArgument.body,
      "constraint source must be nonempty Unicode scalar text",
    );
  }

  const constant = evaluateConstantConstraint(predicate, predicateArgument.body);
  const overlapKey =
    constant === false
      ? undefined
      : constant === true
        ? "constant:true"
        : semanticJsonString(predicate);
  return {
    node: candidate,
    value: { kind, source, predicate, output },
    overlapKey,
  };
}

function parseConstraintExpression(
  expression: ts.Expression,
  context: DefinitionContext,
  depth = 0,
): ConstraintExpression {
  consumeDefinitionNode(context, expression, DIAGNOSTIC_CODE.invalidConstraint, depth);

  if (ts.isParenthesizedExpression(expression)) {
    return parseConstraintExpression(expression.expression, context, depth + 1);
  }

  const literal = constraintLiteral(expression, context);

  if (literal !== undefined) {
    return { node: "literal", value: literal };
  }

  if (ts.isIdentifier(expression)) {
    if (!context.inputTypes.has(expression.text)) {
      fail(
        DIAGNOSTIC_CODE.invalidConstraint,
        expression,
        `constraint identifier ${JSON.stringify(expression.text)} is not an interpolated input`,
      );
    }

    return { node: "input", name: expression.text };
  }

  if (ts.isPropertyAccessExpression(expression) && !expression.questionDotToken) {
    if (ts.isPrivateIdentifier(expression.name)) {
      fail(
        DIAGNOSTIC_CODE.invalidConstraint,
        expression.name,
        "private property access is not supported in constraints",
      );
    }

    assertConstraintJsonDataAccess(expression, context);
    assertConstraintValueIsDefinitelyPresent(expression.expression, context);
    return {
      node: "property",
      object: parseConstraintExpression(expression.expression, context, depth + 1),
      property: expression.name.text,
    };
  }

  if (
    ts.isElementAccessExpression(expression) &&
    !expression.questionDotToken
  ) {
    assertConstraintJsonDataAccess(expression, context);
    assertConstraintValueIsDefinitelyPresent(expression.expression, context);
    assertConstraintValueIsDefinitelyPresent(expression.argumentExpression, context);
    return {
      node: "index",
      object: parseConstraintExpression(expression.expression, context, depth + 1),
      index: parseConstraintExpression(expression.argumentExpression, context, depth + 1),
    };
  }

  if (ts.isPrefixUnaryExpression(expression)) {
    const operator = unaryOperator(expression.operator);

    if (!operator) {
      fail(
        DIAGNOSTIC_CODE.invalidConstraint,
        expression,
        `unsupported constraint unary operator ${ts.tokenToString(expression.operator) ?? "unknown"}`,
      );
    }

    const expectedType =
      operator === "!" ? context.checker.getBooleanType() : context.checker.getNumberType();
    assertDefinitelyPresentAssignable(
      expression.operand,
      expectedType,
      context,
      DIAGNOSTIC_CODE.invalidConstraint,
    );
    const result: ConstraintExpression = {
      node: "unary",
      operator,
      operand: parseConstraintExpression(expression.operand, context, depth + 1),
    };
    return result;
  }

  if (ts.isBinaryExpression(expression)) {
    const operator = binaryOperator(expression.operatorToken.kind);

    if (!operator) {
      fail(
        DIAGNOSTIC_CODE.invalidConstraint,
        expression.operatorToken,
        `unsupported constraint binary operator ${expression.operatorToken.getText()}`,
      );
    }

    validateBinaryOperandTypes(expression, operator, context);
    const result: ConstraintExpression = {
      node: "binary",
      operator,
      left: parseConstraintExpression(expression.left, context, depth + 1),
      right: parseConstraintExpression(expression.right, context, depth + 1),
    };
    return result;
  }

  fail(
    DIAGNOSTIC_CODE.invalidConstraint,
    expression,
    "constraint predicates support only literals, input/property/index access, comparisons, boolean operators, and finite-number arithmetic",
  );
}

function validateBinaryOperandTypes(
  expression: ts.BinaryExpression,
  operator: ConstraintBinaryOperator,
  context: DefinitionContext,
): void {
  if (operator === "===" || operator === "!==") {
    const leftType = context.checker.getTypeAtLocation(expression.left);
    const rightType = context.checker.getTypeAtLocation(expression.right);

    if (
      !isConstraintScalarType(leftType, context) ||
      !isConstraintScalarType(rightType, context)
    ) {
      fail(
        DIAGNOSTIC_CODE.invalidConstraint,
        expression,
        "strict constraint equality requires two scalar JSON values",
      );
    }
    return;
  }

  if (["+", "-", "*", "/", "%", "**"].includes(operator)) {
    const numberType = context.checker.getNumberType();
    assertDefinitelyPresentAssignable(
      expression.left,
      numberType,
      context,
      DIAGNOSTIC_CODE.invalidConstraint,
    );
    assertDefinitelyPresentAssignable(
      expression.right,
      numberType,
      context,
      DIAGNOSTIC_CODE.invalidConstraint,
    );
    return;
  }

  if (operator === "&&" || operator === "||") {
    const booleanType = context.checker.getBooleanType();
    assertDefinitelyPresentAssignable(
      expression.left,
      booleanType,
      context,
      DIAGNOSTIC_CODE.invalidConstraint,
    );
    assertDefinitelyPresentAssignable(
      expression.right,
      booleanType,
      context,
      DIAGNOSTIC_CODE.invalidConstraint,
    );
    return;
  }

  if (["<", "<=", ">", ">="].includes(operator)) {
    const leftType = context.checker.getTypeAtLocation(expression.left);
    const rightType = context.checker.getTypeAtLocation(expression.right);
    const numberType = context.checker.getNumberType();
    const stringType = context.checker.getStringType();
    const numeric =
      context.checker.isTypeAssignableTo(leftType, numberType) &&
      context.checker.isTypeAssignableTo(rightType, numberType);
    const textual =
      context.checker.isTypeAssignableTo(leftType, stringType) &&
      context.checker.isTypeAssignableTo(rightType, stringType);

    if (!numeric && !textual) {
      fail(
        DIAGNOSTIC_CODE.invalidConstraint,
        expression,
        "ordered constraint comparisons require two numbers or two strings",
      );
    }
    assertConstraintValueIsDefinitelyPresent(expression.left, context);
    assertConstraintValueIsDefinitelyPresent(expression.right, context);
  }
}

function assertDefinitelyPresentAssignable(
  expression: ts.Expression,
  target: ts.Type,
  context: DefinitionContext,
  code: number,
): void {
  assertAssignable(expression, target, context, code);
  assertConstraintValueIsDefinitelyPresent(expression, context);
}

function assertConstraintValueIsDefinitelyPresent(
  expression: ts.Expression,
  context: DefinitionContext,
): void {
  if (!constraintAccessMayBeMissing(expression, context)) {
    return;
  }

  fail(
    DIAGNOSTIC_CODE.invalidConstraint,
    expression,
    "constraint operand may be missing because an optional property or array, string, or tuple index is not definitely present",
  );
}

function constraintAccessMayBeMissing(
  expression: ts.Expression,
  context: DefinitionContext,
): boolean {
  if (ts.isParenthesizedExpression(expression)) {
    return constraintAccessMayBeMissing(expression.expression, context);
  }

  if (ts.isPropertyAccessExpression(expression) && !expression.questionDotToken) {
    if (constraintAccessMayBeMissing(expression.expression, context)) {
      return true;
    }

    const property = context.checker.getSymbolAtLocation(expression.name);
    return property !== undefined && (property.flags & ts.SymbolFlags.Optional) !== 0;
  }

  if (!ts.isElementAccessExpression(expression) || expression.questionDotToken) {
    return false;
  }

  if (constraintAccessMayBeMissing(expression.expression, context)) {
    return true;
  }
  if (constraintAccessMayBeMissing(expression.argumentExpression, context)) {
    return true;
  }

  const property = context.checker.getSymbolAtLocation(expression.argumentExpression);
  if (property !== undefined && (property.flags & ts.SymbolFlags.Optional) !== 0) {
    return true;
  }

  const objectType = context.checker.getTypeAtLocation(expression.expression);
  const indexType = context.checker.getTypeAtLocation(expression.argumentExpression);
  return indexedAccessMayBeMissing(objectType, indexType, context, new Set());
}

function indexedAccessMayBeMissing(
  objectType: ts.Type,
  indexType: ts.Type,
  context: DefinitionContext,
  activeTypes: ReadonlySet<ts.Type>,
): boolean {
  if (activeTypes.has(objectType)) {
    return true;
  }

  const nextActiveTypes = new Set(activeTypes);
  nextActiveTypes.add(objectType);
  if (objectType.isUnion()) {
    return objectType.types.some((member) =>
      indexedAccessMayBeMissing(member, indexType, context, nextActiveTypes),
    );
  }

  if (context.checker.isTupleType(objectType)) {
    const keys = constraintIndexKeys(indexType, context.checker, new Set());
    const length = context.checker.getTypeArguments(objectType as ts.TypeReference).length;
    return keys === undefined || keys.some((key) => key !== "length" && key >= length);
  }

  if (
    context.checker.isArrayType(objectType) ||
    context.checker.isTypeAssignableTo(objectType, context.checker.getStringType())
  ) {
    const keys = constraintIndexKeys(indexType, context.checker, new Set());
    return keys === undefined || keys.some((key) => key !== "length");
  }

  const propertyKeys = constraintPropertyKeys(indexType, context.checker, new Set());
  if (propertyKeys !== undefined) {
    return propertyKeys.some((key) => {
      const property = context.checker.getPropertyOfType(objectType, key);
      return property === undefined || (property.flags & ts.SymbolFlags.Optional) !== 0;
    });
  }

  const constraint = context.checker.getBaseConstraintOfType(objectType);
  return constraint && constraint !== objectType
    ? indexedAccessMayBeMissing(constraint, indexType, context, nextActiveTypes)
    : false;
}

function constraintPropertyKeys(
  type: ts.Type,
  checker: ts.TypeChecker,
  activeTypes: ReadonlySet<ts.Type>,
): readonly string[] | undefined {
  if (activeTypes.has(type)) {
    return undefined;
  }
  const nextActiveTypes = new Set(activeTypes);
  nextActiveTypes.add(type);

  if (type.isUnion()) {
    const keys: string[] = [];

    for (const member of type.types) {
      const memberKeys = constraintPropertyKeys(member, checker, nextActiveTypes);

      if (memberKeys === undefined) {
        return undefined;
      }
      keys.push(...memberKeys);
    }
    return keys;
  }

  if ((type.flags & ts.TypeFlags.StringLiteral) !== 0) {
    return [(type as ts.StringLiteralType).value];
  }

  if ((type.flags & ts.TypeFlags.NumberLiteral) !== 0) {
    return [String((type as ts.NumberLiteralType).value)];
  }

  const constraint = checker.getBaseConstraintOfType(type);
  return constraint && constraint !== type
    ? constraintPropertyKeys(constraint, checker, nextActiveTypes)
    : undefined;
}

function constraintIndexKeys(
  type: ts.Type,
  checker: ts.TypeChecker,
  activeTypes: ReadonlySet<ts.Type>,
): readonly (number | "length")[] | undefined {
  if (activeTypes.has(type)) {
    return undefined;
  }
  const nextActiveTypes = new Set(activeTypes);
  nextActiveTypes.add(type);

  if (type.isUnion()) {
    const keys: Array<number | "length"> = [];

    for (const member of type.types) {
      const memberKeys = constraintIndexKeys(member, checker, nextActiveTypes);

      if (memberKeys === undefined) {
        return undefined;
      }
      keys.push(...memberKeys);
    }
    return keys;
  }

  if ((type.flags & ts.TypeFlags.NumberLiteral) !== 0) {
    const value = (type as ts.NumberLiteralType).value;
    return Number.isInteger(value) && value >= 0 && value < 2 ** 32 - 1
      ? [value]
      : undefined;
  }

  if ((type.flags & ts.TypeFlags.StringLiteral) !== 0) {
    const value = (type as ts.StringLiteralType).value;

    if (value === "length") {
      return [value];
    }
    const position = arrayIndexFromString(value);
    return position === undefined ? undefined : [position];
  }

  const constraint = checker.getBaseConstraintOfType(type);
  return constraint && constraint !== type
    ? constraintIndexKeys(constraint, checker, nextActiveTypes)
    : undefined;
}

function arrayIndexFromString(value: string): number | undefined {
  if (value === "0") {
    return 0;
  }
  if (!/^[1-9][0-9]*$/.test(value)) {
    return undefined;
  }
  const result = Number(value);
  return Number.isSafeInteger(result) && result < 2 ** 32 - 1 ? result : undefined;
}

function assertConstraintJsonDataAccess(
  expression: ts.PropertyAccessExpression | ts.ElementAccessExpression,
  context: DefinitionContext,
): void {
  const resultType = context.checker.getTypeAtLocation(expression);

  if (!isConstraintJsonDataType(resultType, context)) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      expression,
      "constraint property and element access must resolve to JSON data",
    );
  }
}

function isConstraintScalarType(
  type: ts.Type,
  context: DefinitionContext,
  activeTypes: ReadonlySet<ts.Type> = new Set(),
): boolean {
  const cached = context.constraintScalarTypes.get(type);

  if (cached !== undefined) {
    return cached;
  }
  if (activeTypes.has(type)) {
    return false;
  }

  let result: boolean;
  if (
    (type.flags &
      (ts.TypeFlags.Any |
        ts.TypeFlags.Unknown |
        ts.TypeFlags.Never |
        ts.TypeFlags.Void |
        ts.TypeFlags.BigIntLike |
        ts.TypeFlags.ESSymbolLike)) !==
    0
  ) {
    result = false;
  } else if (type.isUnion()) {
    const nextActiveTypes = new Set(activeTypes);
    nextActiveTypes.add(type);
    result = type.types.every((member) =>
      isConstraintScalarType(member, context, nextActiveTypes),
    );
  } else if (
    (type.flags & (ts.TypeFlags.Null | ts.TypeFlags.Undefined)) !== 0 ||
    context.checker.isTypeAssignableTo(type, context.checker.getStringType()) ||
    context.checker.isTypeAssignableTo(type, context.checker.getNumberType()) ||
    context.checker.isTypeAssignableTo(type, context.checker.getBooleanType())
  ) {
    result = true;
  } else if (type.isIntersection()) {
    const nextActiveTypes = new Set(activeTypes);
    nextActiveTypes.add(type);
    result = type.types.some((member) =>
      isConstraintScalarType(member, context, nextActiveTypes),
    );
  } else {
    const constraint = context.checker.getBaseConstraintOfType(type);

    if (!constraint || constraint === type) {
      result = false;
    } else {
      const nextActiveTypes = new Set(activeTypes);
      nextActiveTypes.add(type);
      result = isConstraintScalarType(constraint, context, nextActiveTypes);
    }
  }
  context.constraintScalarTypes.set(type, result);
  return result;
}

function isConstraintJsonDataType(
  type: ts.Type,
  context: DefinitionContext,
  activeTypes: ReadonlySet<ts.Type> = new Set(),
): boolean {
  const cached = context.constraintJsonDataTypes.get(type);

  if (cached !== undefined) {
    return cached;
  }
  if (activeTypes.has(type)) {
    return false;
  }

  let result: boolean;
  if (isConstraintScalarType(type, context)) {
    result = true;
  } else if (type.isUnion()) {
    const nextActiveTypes = new Set(activeTypes);
    nextActiveTypes.add(type);
    result = type.types.every((member) =>
      isConstraintJsonDataType(member, context, nextActiveTypes),
    );
  } else if (context.checker.isTupleType(type)) {
    const nextActiveTypes = new Set(activeTypes);
    nextActiveTypes.add(type);
    result = context.checker
      .getTypeArguments(type as ts.TypeReference)
      .every((member) => isConstraintJsonDataType(member, context, nextActiveTypes));
  } else if (context.checker.isArrayType(type)) {
    const itemType = context.checker.getIndexTypeOfType(type, ts.IndexKind.Number);

    if (!itemType) {
      result = false;
    } else {
      const nextActiveTypes = new Set(activeTypes);
      nextActiveTypes.add(type);
      result = isConstraintJsonDataType(itemType, context, nextActiveTypes);
    }
  } else {
    const constraint = context.checker.getBaseConstraintOfType(type);

    if (constraint && constraint !== type) {
      const nextActiveTypes = new Set(activeTypes);
      nextActiveTypes.add(type);
      result = isConstraintJsonDataType(constraint, context, nextActiveTypes);
    } else if (
      (type.flags & ts.TypeFlags.Object) === 0 ||
      context.checker.getSignaturesOfType(type, ts.SignatureKind.Call).length > 0 ||
      context.checker.getSignaturesOfType(type, ts.SignatureKind.Construct).length > 0 ||
      context.checker.getIndexInfosOfType(type).length > 0
    ) {
      result = false;
    } else {
      const nextActiveTypes = new Set(activeTypes);
      nextActiveTypes.add(type);
      result = context.checker.getPropertiesOfType(type).every((property) => {
        const declaration = property.valueDeclaration ?? property.declarations?.[0];

        if (
          !declaration ||
          (!ts.isPropertySignature(declaration) &&
            !ts.isPropertyDeclaration(declaration) &&
            !ts.isPropertyAssignment(declaration) &&
            !ts.isShorthandPropertyAssignment(declaration))
        ) {
          return false;
        }

        return isConstraintJsonDataType(
          context.checker.getTypeOfSymbolAtLocation(property, declaration),
          context,
          nextActiveTypes,
        );
      });
    }
  }
  context.constraintJsonDataTypes.set(type, result);
  return result;
}

function validateConstraintContradictions(constraints: readonly ParsedConstraint[]): void {
  const alwaysByPredicate = new Map<string, string>();
  const neverByPredicate = new Map<string, Set<string>>();

  for (const constraint of constraints) {
    if (constraint.overlapKey === undefined) {
      continue;
    }

    const outputKey = semanticJsonString(constraint.value.output);

    if (constraint.value.kind === "always") {
      const previous = alwaysByPredicate.get(constraint.overlapKey);

      if (previous !== undefined && previous !== outputKey) {
        fail(
          DIAGNOSTIC_CODE.contradictoryConstraint,
          constraint.node,
          "overlapping always constraints require the same output",
        );
      }

      if (neverByPredicate.get(constraint.overlapKey)?.has(outputKey)) {
        fail(
          DIAGNOSTIC_CODE.contradictoryConstraint,
          constraint.node,
          "an always constraint contradicts a matching never constraint",
        );
      }

      alwaysByPredicate.set(constraint.overlapKey, outputKey);
      continue;
    }

    if (alwaysByPredicate.get(constraint.overlapKey) === outputKey) {
      fail(
        DIAGNOSTIC_CODE.contradictoryConstraint,
        constraint.node,
        "a never constraint contradicts a matching always constraint",
      );
    }

    const outputs = neverByPredicate.get(constraint.overlapKey) ?? new Set<string>();
    outputs.add(outputKey);
    neverByPredicate.set(constraint.overlapKey, outputs);
  }
}

function evaluateStaticValue(
  expression: ts.Expression,
  context: DefinitionContext,
  activeSymbols: ReadonlySet<ts.Symbol>,
  code: number,
  depth = 0,
): JsonValue {
  consumeDefinitionNode(context, expression, code, depth);
  const unwrapped = unwrapStaticExpressionWithBudget(expression, context, code, depth);
  const current = unwrapped.expression;
  const childDepth = unwrapped.depth + 1;

  if (ts.isStringLiteral(current) || ts.isNoSubstitutionTemplateLiteral(current)) {
    assertUnicodeScalarString(current.text, current, code);
    return current.text;
  }

  if (current.kind === ts.SyntaxKind.TrueKeyword) {
    return true;
  }

  if (current.kind === ts.SyntaxKind.FalseKeyword) {
    return false;
  }

  if (current.kind === ts.SyntaxKind.NullKeyword) {
    return null;
  }

  if (ts.isNumericLiteral(current)) {
    return finiteNumericLiteral(current, code);
  }

  if (
    ts.isPrefixUnaryExpression(current) &&
    (current.operator === ts.SyntaxKind.PlusToken ||
      current.operator === ts.SyntaxKind.MinusToken)
  ) {
    const operand = evaluateStaticValue(
      current.operand,
      context,
      activeSymbols,
      code,
      childDepth,
    );

    if (typeof operand !== "number") {
      fail(
        code,
        current,
        "compile-time unary plus and minus require a numeric literal",
      );
    }

    const value = current.operator === ts.SyntaxKind.MinusToken ? -operand : operand;

    if (!Number.isFinite(value)) {
      fail(code, current, "definition numbers must be finite");
    }

    return value;
  }

  const enumValue = constantEnumValue(current, context, code);

  if (enumValue !== undefined) {
    return enumValue;
  }

  if (ts.isArrayLiteralExpression(current)) {
    return current.elements.map((element) => {
      if (ts.isSpreadElement(element) || ts.isOmittedExpression(element)) {
        fail(
          code,
          element,
          "compile-time arrays cannot contain spreads or omitted elements",
        );
      }

      return evaluateStaticValue(element, context, activeSymbols, code, childDepth);
    });
  }

  if (ts.isObjectLiteralExpression(current)) {
    const properties = objectProperties(
      current,
      "compile-time object",
      code,
    );
    const value: Record<string, JsonValue> = {};

    for (const [name, property] of properties) {
      defineJsonProperty(
        value,
        name,
        evaluateStaticValue(property.expression, context, activeSymbols, code, childDepth),
      );
    }

    return value;
  }

  if (ts.isIdentifier(current)) {
    const symbol = context.checker.getSymbolAtLocation(current);
    const declaration = symbol?.declarations?.find(
      (candidate): candidate is ts.VariableDeclaration =>
        ts.isVariableDeclaration(candidate) && ts.isIdentifier(candidate.name),
    );

    if (
      !symbol ||
      !declaration ||
      declaration.getSourceFile() !== context.site.sourceFile ||
      !declaration.initializer ||
      (declaration.parent.flags & ts.NodeFlags.Const) === 0
    ) {
      fail(
        code,
        current,
        `compile-time reference ${JSON.stringify(current.text)} must name an initialized const in the same source file`,
      );
    }

    if (activeSymbols.has(symbol)) {
      fail(
        code,
        current,
        `compile-time const ${JSON.stringify(current.text)} is recursive`,
      );
    }

    const nextActive = new Set(activeSymbols);
    nextActive.add(symbol);
    return evaluateStaticValue(
      declaration.initializer,
      context,
      nextActive,
      code,
      childDepth,
    );
  }

  fail(
    code,
    current,
    "value is not compile-time serializable; use literals, literal arrays/objects, same-file const references, or constant enum members",
  );
}

function validateInputValue(
  value: JsonValue,
  type: InputType,
  node: ts.Node,
  budget: DefinitionContext["budget"],
  depth = 0,
): void {
  consumeDefinitionBudget(budget, node, DIAGNOSTIC_CODE.invalidExample, depth);

  switch (type.kind) {
    case "string":
      if (typeof value !== "string") {
        invalidExampleValue(node, "string");
      }
      return;
    case "boolean":
      if (typeof value !== "boolean") {
        invalidExampleValue(node, "boolean");
      }
      return;
    case "number":
      if (typeof value !== "number" || !Number.isFinite(value)) {
        invalidExampleValue(node, "finite number");
      }
      return;
    case "null":
      if (value !== null) {
        invalidExampleValue(node, "null");
      }
      return;
    case "literal":
      if (!primitiveEquals(value, type.value)) {
        invalidExampleValue(node, `literal ${JSON.stringify(type.value)}`);
      }
      return;
    case "enum":
      if (!type.values.some((candidate) => primitiveEquals(value, candidate))) {
        invalidExampleValue(node, `member of enum ${type.name}`);
      }
      return;
    case "array":
      if (!isJsonArray(value)) {
        invalidExampleValue(node, "array");
      }
      value.forEach((item) => {
        validateInputValue(item, type.items, node, budget, depth + 1);
      });
      return;
    case "tuple":
      if (!isJsonArray(value) || value.length !== type.items.length) {
        invalidExampleValue(node, `tuple of length ${String(type.items.length)}`);
      }
      type.items.forEach((itemType, index) => {
        validateInputValue(value[index] as JsonValue, itemType, node, budget, depth + 1);
      });
      return;
    case "object": {
      if (!isJsonObject(value)) {
        invalidExampleValue(node, `object ${type.name}`);
      }

      const keys = Object.keys(value);
      const fields = new Map(type.fields.map((field) => [field.name, field]));

      for (const key of keys) {
        if (!fields.has(key)) {
          invalidExampleValue(node, `object ${type.name} without extra property ${key}`);
        }
      }

      for (const field of type.fields) {
        if (!Object.hasOwn(value, field.name)) {
          if (!field.optional) {
            invalidExampleValue(node, `object ${type.name} with required property ${field.name}`);
          }
          continue;
        }

        validateInputValue(
          value[field.name] as JsonValue,
          field.type,
          node,
          budget,
          depth + 1,
        );
      }
      return;
    }
    case "union":
      if (
        !type.variants.some((variant) =>
          inputValueMatches(value, variant, node, budget, depth),
        )
      ) {
        invalidExampleValue(node, "supported union variant");
      }
  }
}

function inputValueMatches(
  value: JsonValue,
  type: InputType,
  node: ts.Node,
  budget: DefinitionContext["budget"],
  depth: number,
): boolean {
  try {
    validateInputValue(value, type, node, budget, depth + 1);
    return true;
  } catch (error) {
    if (error instanceof DefinitionLimitFailure) {
      throw error;
    }

    if (error instanceof DefinitionFailure) {
      return false;
    }
    throw error;
  }
}

function validateOutputValue(
  value: JsonValue,
  output: OutputSpec,
  node: ts.Node,
  code: number,
): void {
  if (output.kind === "scalar") {
    validateHeadValue(value, output.head, node, code);
    return;
  }

  if (!isJsonObject(value)) {
    invalidDefinitionValue(code, node, `flat output ${output.tsType}`);
  }

  const fields = new Map(output.fields.map((field) => [field.name, field]));

  for (const key of Object.keys(value)) {
    if (!fields.has(key)) {
      invalidDefinitionValue(code, node, `flat output without extra property ${key}`);
    }
  }

  for (const field of output.fields) {
    if (!Object.hasOwn(value, field.name)) {
      invalidDefinitionValue(code, node, `flat output with required property ${field.name}`);
    }

    validateHeadValue(value[field.name] as JsonValue, field.head, node, code);
  }
}

function validateHeadValue(value: JsonValue, head: HeadSpec, node: ts.Node, code: number): void {
  if (head.sourceKind === "boolean") {
    if (typeof value !== "boolean") {
      invalidDefinitionValue(code, node, "boolean output");
    }
    return;
  }

  if (head.sourceKind === "bounded-int" || head.sourceKind === "bounded-number") {
    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      !head.supportDecimal.some((candidate) => Number(candidate) === value)
    ) {
      invalidDefinitionValue(code, node, `${head.sourceKind} support member`);
    }
    return;
  }

  if (!("support" in head)) {
    throw new Error("unsupported output head");
  }

  if (!head.support.some((candidate) => primitiveEquals(value, candidate))) {
    invalidDefinitionValue(code, node, `${head.sourceKind} support member`);
  }
}

function parseTemplate(site: SemaSite, inputs: readonly InputEntry[]): readonly TemplatePart[] {
  const template = site.node.template;

  if (ts.isNoSubstitutionTemplateLiteral(template)) {
    return [{ kind: "text", text: template.text }];
  }

  const parts: TemplatePart[] = [{ kind: "text", text: template.head.text }];

  for (const [index, span] of template.templateSpans.entries()) {
    const input = inputs[index];

    if (!input || !ts.isIdentifier(span.expression) || span.expression.text !== input.name) {
      fail(
        DIAGNOSTIC_CODE.invalidOptions,
        span.expression,
        "resolved inputs do not match the tagged template interpolation order",
      );
    }

    parts.push({ kind: "input", name: input.name });
    parts.push({ kind: "text", text: span.literal.text });
  }

  return parts;
}

function parseConfidence(
  template: readonly TemplatePart[],
  site: SemaSite,
): { readonly confidenceThreshold: number | null; readonly template: readonly TemplatePart[] } {
  let confidenceThreshold: number | null = null;
  let encounteredContent = false;
  let hasBehaviorText = false;
  const rewritten: TemplatePart[] = [];

  for (const part of template) {
    if (part.kind === "input") {
      encounteredContent = true;
      rewritten.push(part);
      continue;
    }

    assertUnicodeScalarString(part.text, site.node.template, DIAGNOSTIC_CODE.invalidConfidence);
    let rewrittenText = "";

    for (const line of splitLines(part.text)) {
      const trimmed = line.content.trim();

      if (!trimmed) {
        rewrittenText += line.full;
        continue;
      }

      if (trimmed.startsWith("@confidence")) {
        if (encounteredContent || confidenceThreshold !== null) {
          fail(
            DIAGNOSTIC_CODE.invalidConfidence,
            site.node.template,
            "@confidence must appear at most once as the first nonblank template line",
          );
        }

        const match = CONFIDENCE_LINE.exec(line.content);

        if (!match?.[1]) {
          fail(
            DIAGNOSTIC_CODE.invalidConfidence,
            site.node.template,
            "malformed @confidence header; expected @confidence(q) with decimal q in [0, 1]",
          );
        }

        confidenceThreshold = Number(match[1]);
        continue;
      }

      encounteredContent = true;
      hasBehaviorText = true;
      rewrittenText += line.full;
    }

    rewritten.push({ kind: "text", text: rewrittenText });
  }

  if (!hasBehaviorText) {
    fail(
      DIAGNOSTIC_CODE.emptyBehavior,
      site.node.template,
      "sema behavioral specification is empty after removing directives and whitespace",
    );
  }

  return { confidenceThreshold, template: rewritten };
}

function inputTypesForSite(site: SemaSite, checker: ts.TypeChecker): ReadonlyMap<string, ts.Type> {
  const result = new Map<string, ts.Type>();
  const template = site.node.template;

  if (!ts.isNoSubstitutionTemplateLiteral(template)) {
    for (const span of template.templateSpans) {
      if (ts.isIdentifier(span.expression)) {
        result.set(span.expression.text, checker.getTypeAtLocation(span.expression));
      }
    }
  }

  return result;
}

function assertAssignable(
  expression: ts.Expression,
  target: ts.Type,
  context: DefinitionContext,
  code: number,
): void {
  const source = context.checker.getTypeAtLocation(expression);

  if (!context.checker.isTypeAssignableTo(source, target)) {
    fail(
      code,
      expression,
      `type ${JSON.stringify(context.checker.typeToString(source))} is not assignable to ${JSON.stringify(context.checker.typeToString(target))}`,
    );
  }
}

function assertStaticExampleInputAssignable(
  expression: ts.Expression,
  target: ts.Type,
  context: DefinitionContext,
): void {
  if (isStaticExampleInputAssignable(expression, target, context, 0)) {
    return;
  }

  const source = context.checker.getTypeAtLocation(expression);
  fail(
    DIAGNOSTIC_CODE.invalidExample,
    expression,
    `type ${JSON.stringify(context.checker.typeToString(source))} is not assignable to ${JSON.stringify(context.checker.typeToString(target))}`,
  );
}

function isStaticExampleInputAssignable(
  expression: ts.Expression,
  target: ts.Type,
  context: DefinitionContext,
  depth: number,
): boolean {
  consumeDefinitionNode(context, expression, DIAGNOSTIC_CODE.invalidExample, depth);
  const source = context.checker.getTypeAtLocation(expression);

  if (context.checker.isTypeAssignableTo(source, target)) {
    return true;
  }

  if (target.isUnion()) {
    return target.types.some((variant) =>
      isStaticExampleInputAssignable(expression, variant, context, depth + 1),
    );
  }

  const unwrapped = unwrapStaticExpressionWithBudget(
    expression,
    context,
    DIAGNOSTIC_CODE.invalidExample,
    depth,
  );
  const current = unwrapped.expression;
  const childDepth = unwrapped.depth + 1;

  if (ts.isArrayLiteralExpression(current)) {
    if (current.elements.some((element) => ts.isSpreadElement(element))) {
      return false;
    }

    if (context.checker.isTupleType(target)) {
      const itemTypes = context.checker.getTypeArguments(target as ts.TypeReference);
      return (
        itemTypes.length === current.elements.length &&
        itemTypes.every((itemType, index) => {
          const item = current.elements[index];
          return (
            !!item &&
            !ts.isOmittedExpression(item) &&
            isStaticExampleInputAssignable(item, itemType, context, childDepth)
          );
        })
      );
    }

    if (
      (target.flags & ts.TypeFlags.Object) !== 0 &&
      context.checker.isArrayLikeType(target)
    ) {
      const itemType = context.checker.getIndexTypeOfType(target, ts.IndexKind.Number);
      return (
        !!itemType &&
        current.elements.every(
          (item) =>
            !ts.isOmittedExpression(item) &&
            isStaticExampleInputAssignable(item, itemType, context, childDepth),
        )
      );
    }

    return false;
  }

  if (!ts.isObjectLiteralExpression(current) || (target.flags & ts.TypeFlags.Object) === 0) {
    return false;
  }

  const sourceProperties = objectProperties(
    current,
    "compile-time object",
    DIAGNOSTIC_CODE.invalidExample,
  );
  const targetProperties = context.checker.getPropertiesOfType(target);

  for (const targetProperty of targetProperties) {
    const sourceProperty = sourceProperties.get(targetProperty.getName());

    if (!sourceProperty) {
      if ((targetProperty.flags & ts.SymbolFlags.Optional) === 0) {
        return false;
      }
      continue;
    }

    const targetPropertyType = context.checker.getTypeOfSymbolAtLocation(
      targetProperty,
      sourceProperty.expression,
    );

    if (
      !isStaticExampleInputAssignable(
        sourceProperty.expression,
        targetPropertyType,
        context,
        childDepth,
      )
    ) {
      return false;
    }
  }

  return [...sourceProperties.keys()].every(
    (name) => context.checker.getPropertyOfType(target, name) !== undefined,
  );
}

function constraintLiteral(
  expression: ts.Expression,
  context: DefinitionContext,
): string | number | boolean | null | undefined {
  if (ts.isStringLiteral(expression) || ts.isNoSubstitutionTemplateLiteral(expression)) {
    assertUnicodeScalarString(expression.text, expression, DIAGNOSTIC_CODE.invalidConstraint);
    return expression.text;
  }

  if (expression.kind === ts.SyntaxKind.TrueKeyword) {
    return true;
  }

  if (expression.kind === ts.SyntaxKind.FalseKeyword) {
    return false;
  }

  if (expression.kind === ts.SyntaxKind.NullKeyword) {
    return null;
  }

  if (ts.isNumericLiteral(expression)) {
    return finiteNumericLiteral(expression, DIAGNOSTIC_CODE.invalidConstraint);
  }

  return constantEnumValue(expression, context, DIAGNOSTIC_CODE.invalidConstraint);
}

function constantEnumValue(
  expression: ts.Expression,
  context: DefinitionContext,
  code: number,
): string | number | undefined {
  let memberName: ts.Node;

  if (ts.isPropertyAccessExpression(expression) && !expression.questionDotToken) {
    memberName = expression.name;
  } else if (
    ts.isElementAccessExpression(expression) &&
    !expression.questionDotToken &&
    (ts.isStringLiteral(expression.argumentExpression) ||
      ts.isNoSubstitutionTemplateLiteral(expression.argumentExpression))
  ) {
    memberName = expression.argumentExpression;
  } else {
    return undefined;
  }

  const symbol =
    context.checker.getSymbolAtLocation(memberName) ??
    context.checker.getSymbolAtLocation(expression);
  const declaration = symbol?.declarations?.find(ts.isEnumMember);

  if (!declaration) {
    return undefined;
  }

  if (declaration.getSourceFile() !== context.site.sourceFile) {
    fail(
      code,
      expression,
      "compile-time enum members must be declared in the same source file",
    );
  }

  const value = context.checker.getConstantValue(declaration);

  if ((typeof value !== "string" && typeof value !== "number") || !isFinitePrimitive(value)) {
    fail(
      code,
      expression,
      "enum member is not a finite string or number constant",
    );
  }

  if (typeof value === "string") {
    assertUnicodeScalarString(value, expression, code);
  }

  return value;
}

function objectProperties(
  object: ts.ObjectLiteralExpression,
  description: string,
  code: number,
): ReadonlyMap<string, { readonly node: ts.ObjectLiteralElementLike; readonly expression: ts.Expression }> {
  const result = new Map<
    string,
    { readonly node: ts.ObjectLiteralElementLike; readonly expression: ts.Expression }
  >();

  for (const property of object.properties) {
    let name: string;
    let expression: ts.Expression;

    if (ts.isPropertyAssignment(property)) {
      name = staticPropertyName(property.name, code);
      expression = property.initializer;
    } else if (ts.isShorthandPropertyAssignment(property)) {
      name = property.name.text;
      expression = property.name;
    } else {
      fail(code, property, `${description} cannot contain spreads, methods, or accessors`);
    }

    if (result.has(name)) {
      fail(code, property, `${description} contains duplicate property ${JSON.stringify(name)}`);
    }

    result.set(name, { node: property, expression });
  }

  return result;
}

function staticPropertyName(name: ts.PropertyName, code: number): string {
  if (ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name)) {
    assertUnicodeScalarString(name.text, name, code);
    return name.text;
  }

  fail(code, name, "computed property names are not compile-time serializable");
}

function symbolForCallable(
  expression: ts.LeftHandSideExpression,
  checker: ts.TypeChecker,
): ts.Symbol | undefined {
  if (ts.isIdentifier(expression)) {
    return checker.getSymbolAtLocation(expression);
  }

  if (ts.isPropertyAccessExpression(expression) && !expression.questionDotToken) {
    return checker.getSymbolAtLocation(expression.name);
  }

  return undefined;
}

function unaryOperator(kind: ts.PrefixUnaryOperator): "!" | "+" | "-" | undefined {
  switch (kind) {
    case ts.SyntaxKind.ExclamationToken:
      return "!";
    case ts.SyntaxKind.PlusToken:
      return "+";
    case ts.SyntaxKind.MinusToken:
      return "-";
    default:
      return undefined;
  }
}

function binaryOperator(
  kind: ts.BinaryOperator,
):
  | "==="
  | "!=="
  | "<"
  | "<="
  | ">"
  | ">="
  | "&&"
  | "||"
  | "+"
  | "-"
  | "*"
  | "/"
  | "%"
  | "**"
  | undefined {
  switch (kind) {
    case ts.SyntaxKind.EqualsEqualsEqualsToken:
      return "===";
    case ts.SyntaxKind.ExclamationEqualsEqualsToken:
      return "!==";
    case ts.SyntaxKind.LessThanToken:
      return "<";
    case ts.SyntaxKind.LessThanEqualsToken:
      return "<=";
    case ts.SyntaxKind.GreaterThanToken:
      return ">";
    case ts.SyntaxKind.GreaterThanEqualsToken:
      return ">=";
    case ts.SyntaxKind.AmpersandAmpersandToken:
      return "&&";
    case ts.SyntaxKind.BarBarToken:
      return "||";
    case ts.SyntaxKind.PlusToken:
      return "+";
    case ts.SyntaxKind.MinusToken:
      return "-";
    case ts.SyntaxKind.AsteriskToken:
      return "*";
    case ts.SyntaxKind.SlashToken:
      return "/";
    case ts.SyntaxKind.PercentToken:
      return "%";
    case ts.SyntaxKind.AsteriskAsteriskToken:
      return "**";
    default:
      return undefined;
  }
}

function evaluateConstantConstraint(
  expression: ConstraintExpression,
  node: ts.Node,
): JsonValue | typeof UNKNOWN_CONSTANT {
  if (expression.node === "literal") {
    return expression.value;
  }

  if (expression.node === "input") {
    return UNKNOWN_CONSTANT;
  }

  if (expression.node === "property") {
    evaluateConstantConstraint(expression.object, node);
    return UNKNOWN_CONSTANT;
  }

  if (expression.node === "index") {
    evaluateConstantConstraint(expression.object, node);
    evaluateConstantConstraint(expression.index, node);
    return UNKNOWN_CONSTANT;
  }

  if (expression.node === "unary") {
    const operand = evaluateConstantConstraint(expression.operand, node);

    if (operand === UNKNOWN_CONSTANT) {
      return UNKNOWN_CONSTANT;
    }

    if (expression.operator === "!") {
      return typeof operand === "boolean" ? !operand : UNKNOWN_CONSTANT;
    }

    if (typeof operand !== "number") {
      return UNKNOWN_CONSTANT;
    }

    return assertFiniteConstraintConstant(
      expression.operator === "-" ? -operand : operand,
      node,
    );
  }

  const left = evaluateConstantConstraint(expression.left, node);
  const right = evaluateConstantConstraint(expression.right, node);

  if (expression.operator === "&&") {
    if (left === false || right === false) {
      return false;
    }

    if (left === true && typeof right === "boolean") {
      return right;
    }

    if (right === true && typeof left === "boolean") {
      return left;
    }

    return UNKNOWN_CONSTANT;
  }

  if (expression.operator === "||") {
    if (left === true || right === true) {
      return true;
    }

    if (left === false && typeof right === "boolean") {
      return right;
    }

    if (right === false && typeof left === "boolean") {
      return left;
    }

    return UNKNOWN_CONSTANT;
  }

  if (left === UNKNOWN_CONSTANT || right === UNKNOWN_CONSTANT) {
    return UNKNOWN_CONSTANT;
  }

  switch (expression.operator) {
    case "===":
      return left === right;
    case "!==":
      return left !== right;
    case "<":
    case "<=":
    case ">":
    case ">=":
      return evaluateConstantComparison(expression.operator, left, right);
    case "+":
      return typeof left === "number" && typeof right === "number"
        ? assertFiniteConstraintConstant(left + right, node)
        : UNKNOWN_CONSTANT;
    case "-":
      return typeof left === "number" && typeof right === "number"
        ? assertFiniteConstraintConstant(left - right, node)
        : UNKNOWN_CONSTANT;
    case "*":
      return typeof left === "number" && typeof right === "number"
        ? assertFiniteConstraintConstant(left * right, node)
        : UNKNOWN_CONSTANT;
    case "/":
      return typeof left === "number" && typeof right === "number"
        ? assertFiniteConstraintConstant(left / right, node)
        : UNKNOWN_CONSTANT;
    case "%":
      return typeof left === "number" && typeof right === "number"
        ? assertFiniteConstraintConstant(left % right, node)
        : UNKNOWN_CONSTANT;
    case "**":
      return typeof left === "number" && typeof right === "number"
        ? assertFiniteConstraintConstant(left ** right, node)
        : UNKNOWN_CONSTANT;
  }
}

function assertFiniteConstraintConstant(value: number, node: ts.Node): number {
  if (!Number.isFinite(value)) {
    fail(
      DIAGNOSTIC_CODE.invalidConstraint,
      node,
      "literal constraint arithmetic must produce a finite number",
    );
  }

  return value;
}

function evaluateConstantComparison(
  operator: "<" | "<=" | ">" | ">=",
  left: JsonValue,
  right: JsonValue,
): boolean | typeof UNKNOWN_CONSTANT {
  if (
    !(
      (typeof left === "string" && typeof right === "string") ||
      (typeof left === "number" && typeof right === "number")
    )
  ) {
    return UNKNOWN_CONSTANT;
  }

  switch (operator) {
    case "<":
      return left < right;
    case "<=":
      return left <= right;
    case ">":
      return left > right;
    case ">=":
      return left >= right;
  }
}

function finiteNumericLiteral(node: ts.NumericLiteral, code: number): number {
  const value = Number(node.getText(node.getSourceFile()).replaceAll("_", ""));

  if (!Number.isFinite(value)) {
    fail(code, node, "numeric literals in definitions must be finite");
  }

  return value;
}

function splitLines(text: string): readonly { readonly content: string; readonly full: string }[] {
  const result: Array<{ readonly content: string; readonly full: string }> = [];
  let start = 0;

  while (start < text.length) {
    let end = start;

    while (end < text.length && text[end] !== "\n" && text[end] !== "\r") {
      end += 1;
    }

    let afterEnd = end;

    if (text[afterEnd] === "\r" && text[afterEnd + 1] === "\n") {
      afterEnd += 2;
    } else if (text[afterEnd] === "\r" || text[afterEnd] === "\n") {
      afterEnd += 1;
    }

    result.push({ content: text.slice(start, end), full: text.slice(start, afterEnd) });
    start = afterEnd;
  }

  return result;
}

function unwrapStaticExpression(expression: ts.Expression): ts.Expression {
  let current = expression;

  while (
    ts.isParenthesizedExpression(current) ||
    ts.isAsExpression(current) ||
    ts.isTypeAssertionExpression(current) ||
    ts.isSatisfiesExpression(current)
  ) {
    current = current.expression;
  }

  return current;
}

function unwrapStaticExpressionWithBudget(
  expression: ts.Expression,
  context: DefinitionContext,
  code: number,
  depth: number,
): { readonly expression: ts.Expression; readonly depth: number } {
  let current = expression;
  let currentDepth = depth;

  while (
    ts.isParenthesizedExpression(current) ||
    ts.isAsExpression(current) ||
    ts.isTypeAssertionExpression(current) ||
    ts.isSatisfiesExpression(current)
  ) {
    current = current.expression;
    currentDepth += 1;
    consumeDefinitionNode(context, current, code, currentDepth);
  }

  return { expression: current, depth: currentDepth };
}

function primitiveEquals(value: JsonValue, expected: JsonValue): boolean {
  return (
    (typeof value === "number" && typeof expected === "number" && value === expected) ||
    value === expected
  );
}

function isJsonObject(
  value: JsonValue,
): value is { readonly [name: string]: JsonValue } {
  return value !== null && !Array.isArray(value) && typeof value === "object";
}

function isJsonArray(value: JsonValue): value is readonly JsonValue[] {
  return Array.isArray(value);
}

function isFinitePrimitive(value: string | number): boolean {
  return typeof value === "string" || Number.isFinite(value);
}

function defineJsonProperty(target: Record<string, JsonValue>, name: string, value: JsonValue): void {
  Object.defineProperty(target, name, {
    configurable: true,
    enumerable: true,
    value,
    writable: true,
  });
}

function invalidExampleValue(node: ts.Node, expected: string): never {
  fail(DIAGNOSTIC_CODE.invalidExample, node, `example value must match ${expected}`);
}

function invalidDefinitionValue(code: number, node: ts.Node, expected: string): never {
  fail(code, node, `definition value must match ${expected}`);
}

function assertUnicodeScalarString(value: string, node: ts.Node, code: number): void {
  if (!isUnicodeScalarString(value)) {
    fail(code, node, "definition strings cannot contain unpaired UTF-16 surrogates");
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

function toDiagnostic(failure: DefinitionFailure, site: SemaSite): DefinitionDiagnostic {
  const start = failure.node.getStart(site.sourceFile);
  const prefix = `${site.location.fileName}:${String(site.location.line)}:${String(site.location.column)}`;
  const siteStart = site.node.getStart(site.sourceFile);
  const siteEnd = site.node.getEnd();
  const previewEnd = Math.min(
    siteEnd,
    siteStart + MAX_DIAGNOSTIC_SITE_CHARACTERS * 2,
  );
  let siteText = truncateDiagnosticText(site.sourceFile.text.slice(siteStart, previewEnd));

  if (previewEnd < siteEnd && !siteText.endsWith("…")) {
    siteText += "…";
  }
  return {
    category: ts.DiagnosticCategory.Error,
    code: failure.code,
    file: site.sourceFile,
    start,
    length: failure.node.getEnd() - start,
    messageText: `${prefix}: ${failure.message} (site: ${siteText})`,
    site,
  };
}

function consumeDefinitionNode(
  context: DefinitionContext,
  node: ts.Node,
  code: number,
  depth: number,
): void {
  consumeDefinitionBudget(context.budget, node, code, depth);
}

function consumeDefinitionBudget(
  budget: DefinitionContext["budget"],
  node: ts.Node,
  code: number,
  depth: number,
): void {
  if (depth > MAX_DEFINITION_DEPTH) {
    throw new DefinitionLimitFailure(
      code,
      node,
      `compile-time definition nesting exceeds the limit of ${String(MAX_DEFINITION_DEPTH)}`,
    );
  }

  budget.remaining -= 1;

  if (budget.remaining < 0) {
    throw new DefinitionLimitFailure(
      code,
      node,
      `compile-time definition work exceeds the limit of ${String(MAX_DEFINITION_NODES)} nodes`,
    );
  }
}

function truncateDiagnosticText(value: string): string {
  let end = 0;
  let characters = 0;

  while (end < value.length && characters < MAX_DIAGNOSTIC_SITE_CHARACTERS) {
    const codePoint = value.codePointAt(end);
    end += codePoint !== undefined && codePoint > 0xffff ? 2 : 1;
    characters += 1;
  }

  return end < value.length ? `${value.slice(0, end)}…` : value;
}

function fail(code: number, node: ts.Node, message: string): never {
  throw new DefinitionFailure(code, node, message);
}
