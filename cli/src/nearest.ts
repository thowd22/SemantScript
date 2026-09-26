/**
 * The typed distance `semantscript explain` ranks gold examples and training
 * cases by. It compares the JSON form of two input records leaf by leaf and
 * says nothing about how the model represents them: it is a way to find the
 * labelled cases that look most like the input, not the model's similarity.
 */
export const DISTANCE_DESCRIPTION =
  "mean over the inputs' leaf fields of a per-field distance: numbers |a-b| divided by that field's spread among the candidates and the input, strings 1 minus the Jaccard overlap of their lowercase word sets, booleans, null and other values 0 when equal and 1 otherwise, a field present on one side only 1";

type Leaves = ReadonlyMap<string, unknown>;

export interface Neighbour<T> {
  readonly candidate: T;
  readonly distance: number;
}

/** The `count` candidates nearest to `query`, ties kept in candidate order. */
export function nearest<T>(
  query: unknown,
  candidates: readonly T[],
  inputsOf: (candidate: T) => unknown,
  count: number,
): readonly Neighbour<T>[] {
  if (count <= 0 || candidates.length === 0) return [];
  const queryLeaves = leaves(query);
  const candidateLeaves = candidates.map((candidate) =>
    leaves(inputsOf(candidate)),
  );
  const spreads = numericSpreads([queryLeaves, ...candidateLeaves]);
  return candidates
    .map((candidate, index) => ({
      candidate,
      index,
      distance: distance(
        queryLeaves,
        candidateLeaves[index] ?? new Map(),
        spreads,
      ),
    }))
    .sort(
      (left, right) =>
        left.distance - right.distance || left.index - right.index,
    )
    .slice(0, count)
    .map(({ candidate, distance: value }) => ({
      candidate,
      distance: value,
    }));
}

/** Flatten a JSON value to `/path` leaves; an empty array or object is one leaf. */
export function leaves(value: unknown): Leaves {
  const result = new Map<string, unknown>();
  const visit = (current: unknown, path: string): void => {
    if (Array.isArray(current)) {
      if (current.length === 0) result.set(path, "[]");
      current.forEach((entry, index) => {
        visit(entry, `${path}/${String(index)}`);
      });
      return;
    }
    if (current !== null && typeof current === "object") {
      const entries = Object.entries(current);
      if (entries.length === 0) result.set(path, "{}");
      for (const [key, entry] of entries) visit(entry, `${path}/${key}`);
      return;
    }
    result.set(path, current);
  };
  visit(value, "");
  return result;
}

function numericSpreads(
  records: readonly Leaves[],
): ReadonlyMap<string, number> {
  const minimum = new Map<string, number>();
  const maximum = new Map<string, number>();
  for (const record of records) {
    for (const [path, value] of record) {
      if (typeof value !== "number") continue;
      minimum.set(path, Math.min(minimum.get(path) ?? value, value));
      maximum.set(path, Math.max(maximum.get(path) ?? value, value));
    }
  }
  const spreads = new Map<string, number>();
  for (const [path, low] of minimum) {
    spreads.set(path, (maximum.get(path) ?? low) - low);
  }
  return spreads;
}

function distance(
  left: Leaves,
  right: Leaves,
  spreads: ReadonlyMap<string, number>,
): number {
  const paths = new Set([...left.keys(), ...right.keys()]);
  if (paths.size === 0) return 0;
  let total = 0;
  for (const path of paths) {
    if (!left.has(path) || !right.has(path)) {
      total += 1;
      continue;
    }
    total += leafDistance(left.get(path), right.get(path), spreads.get(path));
  }
  return total / paths.size;
}

function leafDistance(
  left: unknown,
  right: unknown,
  spread: number | undefined,
): number {
  if (typeof left === "number" && typeof right === "number") {
    if (left === right) return 0;
    return spread === undefined || spread === 0
      ? 1
      : Math.min(1, Math.abs(left - right) / spread);
  }
  if (typeof left === "string" && typeof right === "string") {
    if (left === right) return 0;
    const a = words(left);
    const b = words(right);
    const union = new Set([...a, ...b]);
    if (union.size === 0) return 1;
    let shared = 0;
    for (const word of a) if (b.has(word)) shared += 1;
    return 1 - shared / union.size;
  }
  return Object.is(left, right) ? 0 : 1;
}

function words(text: string): ReadonlySet<string> {
  return new Set(text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);
}
