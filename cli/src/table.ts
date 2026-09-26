import type { ManifestFunctionSummary } from "./manifest.js";

/** Render aligned plain-text columns for terminal output. */
export function renderTable(
  headers: readonly string[],
  rows: readonly (readonly string[])[],
): string {
  const widths = headers.map((header, index) =>
    Math.max(header.length, ...rows.map((row) => (row[index] ?? "").length)),
  );
  const line = (cells: readonly string[]): string =>
    cells
      .map((cell, index) => cell.padEnd(widths[index] ?? cell.length))
      .join("  ")
      .trimEnd();
  return `${[line(headers), line(widths.map((width) => "-".repeat(width))), ...rows.map(line)].join("\n")}\n`;
}

export function formatRatio(value: number): string {
  return value.toFixed(4);
}

export function shortId(functionId: string): string {
  return functionId.length > 14 ? `${functionId.slice(0, 11)}…` : functionId;
}

/** The per-function verification columns `test` and `releases show` share. */
export const VERIFICATION_HEADERS: readonly string[] = [
  "function",
  "verification",
  "accuracy",
  "ece",
  "brier",
  "pairs",
  "attested",
  "violations",
  "heads",
];

export function verificationCells(
  fn: ManifestFunctionSummary,
): readonly string[] {
  return [
    shortId(fn.id),
    fn.status,
    formatRatio(fn.accuracy),
    formatRatio(fn.ece),
    formatRatio(fn.brier),
    formatRatio(fn.pairConsistency),
    String(fn.attestedCases),
    String(fn.constraintViolations),
    fn.heads.map((head) => formatRatio(head.accuracy)).join("/"),
  ];
}
