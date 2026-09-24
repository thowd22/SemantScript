import ts from "typescript";

export const MAX_BOUNDED_SUPPORT = 100_000;
const MAX_DECIMAL_DIGITS = 1_000;

export interface DecimalRational {
  readonly coefficient: bigint;
  readonly scale: number;
}

export interface DecimalGrid {
  readonly minimum: string;
  readonly maximum: string;
  readonly step: string;
  readonly supportDecimal: readonly string[];
}

export function parseDecimalTypeNode(
  node: ts.TypeNode,
): DecimalRational | undefined {
  const unwrapped = unwrapParenthesizedType(node);

  if (!ts.isLiteralTypeNode(unwrapped)) {
    return undefined;
  }

  const literal = unwrapped.literal;

  if (ts.isNumericLiteral(literal)) {
    return parseDecimalLexeme(literal.getText(literal.getSourceFile()));
  }

  if (
    ts.isPrefixUnaryExpression(literal) &&
    literal.operator === ts.SyntaxKind.MinusToken &&
    ts.isNumericLiteral(literal.operand)
  ) {
    return parseDecimalLexeme(
      `-${literal.operand.getText(literal.getSourceFile())}`,
    );
  }

  return undefined;
}

export function parseDecimalLexeme(
  lexeme: string,
): DecimalRational | undefined {
  const normalized = lexeme.replaceAll("_", "");

  if (normalized.length > MAX_DECIMAL_DIGITS) {
    return undefined;
  }

  const match =
    /^(?<sign>-?)(?:(?<integer>[0-9]+)(?:\.(?<fraction>[0-9]*))?|\.(?<leadingFraction>[0-9]+))(?:[eE](?<exponent>[+-]?[0-9]+))?$/.exec(
      normalized,
    );

  if (!match?.groups) {
    return undefined;
  }

  const integer = match.groups["integer"] ?? "0";
  const fraction =
    match.groups["fraction"] ?? match.groups["leadingFraction"] ?? "";
  const exponent = Number(match.groups["exponent"] ?? "0");

  if (!Number.isSafeInteger(exponent)) {
    return undefined;
  }

  if (Math.abs(exponent) > MAX_DECIMAL_DIGITS) {
    return undefined;
  }

  let coefficient = BigInt(`${integer}${fraction}` || "0");
  let scale = fraction.length - exponent;

  if (match.groups["sign"] === "-") {
    coefficient = -coefficient;
  }

  if (scale < 0) {
    coefficient *= 10n ** BigInt(-scale);
    scale = 0;
  }

  return normalizeDecimal({ coefficient, scale });
}

export function buildDecimalGrid(
  minimum: DecimalRational,
  maximum: DecimalRational,
  step: DecimalRational,
  requireIntegers: boolean,
): DecimalGrid | string {
  if (compareDecimals(minimum, maximum) >= 0) {
    return "minimum must be less than maximum";
  }

  if (step.coefficient <= 0n) {
    return "step must be greater than zero";
  }

  if (requireIntegers && (minimum.scale !== 0 || maximum.scale !== 0)) {
    return "BoundedInt bounds must be integer literals";
  }

  const commonScale = Math.max(minimum.scale, maximum.scale, step.scale);
  const minimumInteger = scaleCoefficient(minimum, commonScale);
  const maximumInteger = scaleCoefficient(maximum, commonScale);
  const stepInteger = scaleCoefficient(step, commonScale);
  const difference = maximumInteger - minimumInteger;

  if (difference % stepInteger !== 0n) {
    return "(maximum - minimum) must be exactly divisible by step";
  }

  const intervalCount = difference / stepInteger;

  if (intervalCount + 1n > BigInt(MAX_BOUNDED_SUPPORT)) {
    return `bounded support exceeds the ${MAX_BOUNDED_SUPPORT.toLocaleString("en-US")} value limit`;
  }

  const supportDecimal: string[] = [];
  const seenNumbers = new Set<number>();

  for (let index = 0n; index <= intervalCount; index += 1n) {
    const value = normalizeDecimal({
      coefficient: minimumInteger + index * stepInteger,
      scale: commonScale,
    });
    const canonical = formatDecimal(value);
    const numeric = Number(canonical);

    if (!Number.isFinite(numeric)) {
      return `support value ${canonical} is not a finite JavaScript number`;
    }

    if (seenNumbers.has(numeric)) {
      return `support value ${canonical} collides after IEEE 754 conversion`;
    }

    seenNumbers.add(numeric);
    supportDecimal.push(canonical);
  }

  return {
    minimum: formatDecimal(minimum),
    maximum: formatDecimal(maximum),
    step: formatDecimal(step),
    supportDecimal,
  };
}

export function formatDecimal(value: DecimalRational): string {
  const normalized = normalizeDecimal(value);

  if (normalized.coefficient === 0n) {
    return "0";
  }

  const negative = normalized.coefficient < 0n;
  const digits = (
    negative ? -normalized.coefficient : normalized.coefficient
  ).toString();

  if (normalized.scale === 0) {
    return `${negative ? "-" : ""}${digits}`;
  }

  const padded = digits.padStart(normalized.scale + 1, "0");
  const split = padded.length - normalized.scale;
  return `${negative ? "-" : ""}${padded.slice(0, split)}.${padded.slice(split)}`;
}

function compareDecimals(
  left: DecimalRational,
  right: DecimalRational,
): number {
  const scale = Math.max(left.scale, right.scale);
  const leftCoefficient = scaleCoefficient(left, scale);
  const rightCoefficient = scaleCoefficient(right, scale);

  if (leftCoefficient < rightCoefficient) {
    return -1;
  }

  if (leftCoefficient > rightCoefficient) {
    return 1;
  }

  return 0;
}

function scaleCoefficient(value: DecimalRational, scale: number): bigint {
  return value.coefficient * 10n ** BigInt(scale - value.scale);
}

function normalizeDecimal(value: DecimalRational): DecimalRational {
  let { coefficient, scale } = value;

  while (scale > 0 && coefficient % 10n === 0n) {
    coefficient /= 10n;
    scale -= 1;
  }

  return { coefficient, scale };
}

function unwrapParenthesizedType(node: ts.TypeNode): ts.TypeNode {
  let current = node;

  while (ts.isParenthesizedTypeNode(current)) {
    current = current.type;
  }

  return current;
}
