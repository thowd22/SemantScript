/**
 * Persistence and transactions (TASK-8.2): data stays out of the weights.
 *
 * Handlers read rows from a database, hand plain values to sema expressions
 * and let the neural decision gate the transaction. A sema expression never
 * sees the database: its inputs are its interpolated values only, the
 * compiler rejects a client object as an input type, and the runtime
 * serializes plain data. The helpers below work over any client that speaks
 * the node-postgres query shape (`pg` Pool and Client, PGlite).
 */

export interface QueryResultLike<Row = Record<string, unknown>> {
  readonly rows: Row[];
  readonly rowCount?: number | null;
}

/** The query surface the helpers need: node-postgres and PGlite both provide it. */
export interface QueryableClient {
  query<Row = Record<string, unknown>>(
    text: string,
    params?: readonly unknown[],
  ): Promise<QueryResultLike<Row>>;
}

/** A pool that hands out connections; `pg.Pool` has this shape. */
export interface PoolLike {
  connect(): Promise<QueryableClient & { release(): void }>;
}

/** The outcome of a transaction body: commit with a value, or roll back with a reason. */
export class Rollback<T = unknown> {
  constructor(
    readonly reason: string,
    readonly value?: T,
  ) {}
}

/** Return a `Rollback` from a `transactional` body to roll the transaction back with `reason` (and an optional value). */
export function rollback<T = unknown>(reason: string, value?: T): Rollback<T> {
  return new Rollback(reason, value);
}

export interface TransactionOutcome<T> {
  readonly committed: boolean;
  readonly reason?: string;
  readonly value: T | undefined;
}

/**
 * Runs `body` inside one transaction on `client`: BEGIN, then COMMIT when the
 * body returns a value, ROLLBACK when it returns `rollback(reason)` or throws
 * (the error is rethrown after the rollback). A `PoolLike` is checked out and
 * released around the transaction.
 */
export async function transactional<T>(
  client: QueryableClient | PoolLike,
  body: (tx: QueryableClient) => Promise<T | Rollback<T>> | T | Rollback<T>,
): Promise<TransactionOutcome<T>> {
  const leased = isPool(client) ? await client.connect() : undefined;
  const tx: QueryableClient = leased ?? (client as QueryableClient);
  try {
    await tx.query("BEGIN");
    let result: T | Rollback<T>;
    try {
      result = await body(tx);
    } catch (error) {
      await tx.query("ROLLBACK");
      throw error;
    }
    if (result instanceof Rollback) {
      await tx.query("ROLLBACK");
      return { committed: false, reason: result.reason, value: result.value };
    }
    await tx.query("COMMIT");
    return { committed: true, value: result };
  } finally {
    leased?.release();
  }
}

/**
 * The gate a neural decision applies to a transaction: `commit` when the
 * decision satisfies `accept`, else a rollback naming the decision.
 */
export function gate<Decision, T>(
  decision: Decision,
  accept: (decision: Decision) => boolean,
  commit: () => Promise<T> | T,
  reason: string = `decision ${JSON.stringify(decision)} does not permit the write`,
): Promise<T | Rollback<T>> {
  if (!accept(decision)) {
    return Promise.resolve(rollback<T>(reason));
  }
  return Promise.resolve(commit());
}

function isPool(client: QueryableClient | PoolLike): client is PoolLike {
  return (
    typeof (client as PoolLike).connect === "function" &&
    typeof (client as QueryableClient).query !== "function"
  );
}
