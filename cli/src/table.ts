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
