import { PGlite } from "@electric-sql/pglite";

/**
 * The service's data lives in Postgres, never in the weights: PGlite runs
 * Postgres 17 in process so the example needs no server. Handlers read rows,
 * hand the decisions plain values, and write only what a decision allows.
 */
export const db = new PGlite();

export async function migrate(): Promise<void> {
  await db.exec(`
    CREATE TABLE IF NOT EXISTS customers (
      id text PRIMARY KEY, tier text NOT NULL, prior_refunds int NOT NULL);
    CREATE TABLE IF NOT EXISTS orders (
      id text PRIMARY KEY, customer_id text NOT NULL REFERENCES customers(id),
      total numeric NOT NULL, age_days int NOT NULL, status text NOT NULL,
      payment_method text NOT NULL, mismatched_address boolean NOT NULL DEFAULT false,
      orders_last_hour int NOT NULL DEFAULT 0, chargebacks int NOT NULL DEFAULT 0,
      shipment_held boolean NOT NULL DEFAULT false);
    CREATE TABLE IF NOT EXISTS refunds (
      order_id text PRIMARY KEY REFERENCES orders(id), decision text NOT NULL,
      method text NOT NULL, risk text NOT NULL, decided_at timestamptz NOT NULL DEFAULT now());
    CREATE TABLE IF NOT EXISTS tickets (
      id text PRIMARY KEY, category text NOT NULL, impacted_users int NOT NULL,
      customer_tier text NOT NULL, priority text, queue text, needs_human boolean);
    CREATE TABLE IF NOT EXISTS fraud_reviews (
      order_id text PRIMARY KEY REFERENCES orders(id), flag text NOT NULL,
      escalation text NOT NULL, reviewed_at timestamptz NOT NULL DEFAULT now());
    INSERT INTO customers VALUES ('c1', 'standard', 1), ('c2', 'enterprise', 4), ('c3', 'standard', 0)
      ON CONFLICT DO NOTHING;
    INSERT INTO orders (id, customer_id, total, age_days, status, payment_method, mismatched_address, orders_last_hour, chargebacks) VALUES
      ('o1', 'c1', 88.5, 12, 'paid', 'card', false, 1, 0),
      ('o2', 'c2', 900, 70, 'paid', 'bank', false, 0, 0),
      ('o3', 'c3', 2600, 3, 'paid', 'card', true, 2, 0),
      ('o4', 'c1', 6200, 1, 'fraudulent', 'voucher', true, 5, 1)
      ON CONFLICT DO NOTHING;
    INSERT INTO tickets (id, category, impacted_users, customer_tier) VALUES
      ('t1', 'outage', 400, 'enterprise'), ('t2', 'billing', 3, 'standard'), ('t3', 'feature', 1, 'standard')
      ON CONFLICT DO NOTHING;
  `);
}
