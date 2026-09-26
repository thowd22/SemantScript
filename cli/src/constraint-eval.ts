/**
 * The v1 constraint predicate AST evaluated the way the trainer does
 * (trainer/src/semantscript_trainer/constraints.py `_evaluate`), so
 * `semantscript explain` reports the same active constraints for an input as
 * the dataset builder and the verifier. The shared vectors in
 * examples/constraints/predicate-vectors.v1.json pin both evaluators.
 */

/** A predicate that cannot be evaluated for this input (the trainer raises the same). */
export class ConstraintEvaluationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConstraintEvaluationError";
  }
}

/** Absent object property or out-of-range index: JavaScript `undefined`. */
const UNDEFINED: unique symbol = Symbol("undefined");
type Value = unknown;

const MAXIMUM_DEPTH = 100;
const MAXIMUM_ARRAY_INDEX = 2 ** 32 - 1;
const BINARY_OPERATORS = new Set([
  "===",
  "!==",
  "<",
  "<=",
  ">",
  ">=",
  "&&",
  "||",
  "+",
  "-",
  "*",
  "/",
  "%",
  "**",
]);

/** Evaluate one predicate AST against a call's inputs; true when the constraint is active. */
export function evaluatePredicate(
  predicate: unknown,
  inputs: Readonly<Record<string, unknown>>,
): boolean {
  const value = evaluate(predicate, inputs, 1);
  if (typeof value !== "boolean") {
    throw new ConstraintEvaluationError(
      "predicate did not evaluate to a boolean",
    );
  }
  return value;
}

/** JavaScript strict equality over JSON values, as the trainer compares outputs. */
export function jsonStrictEqual(left: unknown, right: unknown): boolean {
  return strictEqual(left, right);
}

function node(value: unknown): Readonly<Record<string, unknown>> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new ConstraintEvaluationError("predicate node must be an object");
  }
  return value as Readonly<Record<string, unknown>>;
}

function evaluate(
  expression: unknown,
  inputs: Readonly<Record<string, unknown>>,
  depth: number,
): Value {
  if (depth > MAXIMUM_DEPTH) {
    throw new ConstraintEvaluationError(
      `predicate exceeds maximum depth ${String(MAXIMUM_DEPTH)}`,
    );
  }
  const current = node(expression);
  switch (current["node"]) {
    case "literal":
      return current["value"];
    case "input": {
      const name = String(current["name"]);
      if (!Object.hasOwn(inputs, name)) {
        throw new ConstraintEvaluationError(
          `references missing input ${JSON.stringify(name)}`,
        );
      }
      return inputs[name];
    }
    case "property": {
      const value = evaluate(current["object"], inputs, depth + 1);
      const property = String(current["property"]);
      if (isMapping(value)) {
        return Object.hasOwn(value, property) ? value[property] : UNDEFINED;
      }
      if (
        (Array.isArray(value) || typeof value === "string") &&
        property === "length"
      ) {
        return value.length;
      }
      throw new ConstraintEvaluationError(
        `cannot read property ${JSON.stringify(property)} from this value`,
      );
    }
    case "index": {
      const value = evaluate(current["object"], inputs, depth + 1);
      const index = evaluate(current["index"], inputs, depth + 1);
      return readIndex(value, index);
    }
    case "unary": {
      const operand = evaluate(current["operand"], inputs, depth + 1);
      const operator = current["operator"];
      if (operator === "!") {
        if (typeof operand !== "boolean") {
          throw new ConstraintEvaluationError("unary ! operand is not boolean");
        }
        return !operand;
      }
      if (operator !== "+" && operator !== "-") {
        throw new ConstraintEvaluationError("unsupported unary operator");
      }
      const number = numberOf(operand, `unary ${operator} operand`);
      return finite(operator === "+" ? number : -number);
    }
    case "binary":
      return evaluateBinary(current, inputs, depth);
    default:
      throw new ConstraintEvaluationError(
        `unsupported expression node ${JSON.stringify(current["node"])}`,
      );
  }
}

function evaluateBinary(
  current: Readonly<Record<string, unknown>>,
  inputs: Readonly<Record<string, unknown>>,
  depth: number,
): Value {
  const operator = String(current["operator"]);
  if (!BINARY_OPERATORS.has(operator)) {
    throw new ConstraintEvaluationError("unsupported binary operator");
  }
  const left = evaluate(current["left"], inputs, depth + 1);
  if (operator === "&&" || operator === "||") {
    if (typeof left !== "boolean") {
      throw new ConstraintEvaluationError(
        `${operator} left operand is not boolean`,
      );
    }
    if (operator === "&&" ? !left : left) return left;
    const right = evaluate(current["right"], inputs, depth + 1);
    if (typeof right !== "boolean") {
      throw new ConstraintEvaluationError(
        `${operator} right operand is not boolean`,
      );
    }
    return right;
  }
  const right = evaluate(current["right"], inputs, depth + 1);
  switch (operator) {
    case "===":
      return strictEqual(left, right);
    case "!==":
      return !strictEqual(left, right);
    case "<":
    case "<=":
    case ">":
    case ">=":
      return orderedCompare(left, right, operator);
    default:
      break;
  }
  const a = numberOf(left, `${operator} left operand`);
  const b = numberOf(right, `${operator} right operand`);
  switch (operator) {
    case "+":
      return finite(a + b);
    case "-":
      return finite(a - b);
    case "*":
      return finite(a * b);
    case "/":
      if (b === 0) throw nonFinite();
      return finite(a / b);
    case "%":
      if (b === 0) throw nonFinite();
      return finite(a % b);
    default:
      return finite(a ** b);
  }
}

function readIndex(value: Value, index: Value): Value {
  if (isMapping(value)) {
    const key = propertyKey(index);
    return Object.hasOwn(value, key) ? value[key] : UNDEFINED;
  }
  if (Array.isArray(value) || typeof value === "string") {
    if (index === "length") return value.length;
    const position = arrayIndex(index);
    if (position === undefined || position >= value.length) return UNDEFINED;
    return value[position];
  }
  throw new ConstraintEvaluationError("cannot index this value");
}

function propertyKey(value: Value): string {
  if (typeof value === "string") return value;
  if (typeof value === "number") {
    return String(numberOf(value, "object index"));
  }
  throw new ConstraintEvaluationError("object index is not a string or number");
}

function arrayIndex(value: Value): number | undefined {
  if (typeof value === "string") {
    if (value === "0") return 0;
    if (!/^[1-9][0-9]*$/u.test(value)) return undefined;
    const integer = Number(value);
    return integer < MAXIMUM_ARRAY_INDEX ? integer : undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  if (value < 0 || !Number.isInteger(value)) return undefined;
  return value < MAXIMUM_ARRAY_INDEX ? value + 0 : undefined;
}

function orderedCompare(left: Value, right: Value, operator: string): boolean {
  let comparison: number;
  if (typeof left === "string" && typeof right === "string") {
    // JavaScript compares strings by UTF-16 code units, as the trainer does.
    comparison = left < right ? -1 : left > right ? 1 : 0;
  } else if (typeof left === "number" && typeof right === "number") {
    const a = numberOf(left, "left comparison operand");
    const b = numberOf(right, "right comparison operand");
    comparison = a < b ? -1 : a > b ? 1 : 0;
  } else {
    throw new ConstraintEvaluationError(
      "ordered comparison operands must both be numbers or both be strings",
    );
  }
  switch (operator) {
    case "<":
      return comparison < 0;
    case "<=":
      return comparison <= 0;
    case ">":
      return comparison > 0;
    default:
      return comparison >= 0;
  }
}

function strictEqual(left: Value, right: Value): boolean {
  if (left === UNDEFINED || right === UNDEFINED) return left === right;
  if (typeof left === "number" && typeof right === "number") {
    return (
      numberOf(left, "equality operand") === numberOf(right, "equality operand")
    );
  }
  // Objects and arrays compare by identity, strings by code units, the rest by value.
  return left === right;
}

function numberOf(value: Value, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ConstraintEvaluationError(`${context} is not a finite number`);
  }
  return value;
}

function finite(value: number): number {
  if (!Number.isFinite(value)) throw nonFinite();
  return value;
}

function nonFinite(): ConstraintEvaluationError {
  return new ConstraintEvaluationError("produced a non-finite intermediate");
}

function isMapping(value: Value): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
