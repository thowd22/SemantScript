import { type ScalarSemaResult } from "@semantscript/core";
export type RefundDecision = "approve" | "deny" | "review";
export interface Customer {
    readonly priorRefunds: number;
    readonly tier: "enterprise" | "standard";
}
export interface Order {
    readonly ageDays: number;
    readonly status: "fraudulent" | "paid";
    readonly total: number;
}
/**
 * Canonical diagnostic form of the Phase 1 refund task.
 *
 * The policy is stated completely: the first Phase 1 run showed that the short
 * form ("60 days, 30 days, suspicious goes to review") was read three different
 * ways by the teacher, the judge and the baselines. Every rule below is also a
 * deterministic constraint so generated cases are rule-checked at build time.
 *
 * Examples intentionally live outside this source file: benchmark evaluation
 * cases must never become compiler examples or training inputs by accident.
 */
export declare function decideRefund(customer: Customer, order: Order): ScalarSemaResult<RefundDecision>;
