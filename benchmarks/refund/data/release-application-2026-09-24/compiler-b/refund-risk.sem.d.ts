import { type ScalarSemaResult } from "@semantscript/core";
export type RefundRisk = "low" | "medium" | "high";
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
 * Companion function for the shared-encoder experiment (TASK-6.1).
 *
 * It reads the same customer and order records as the refund decision but
 * answers a different question, so a shared encoder must serve two heads with
 * different label spaces. The policy is stated completely and every rule is a
 * constraint, which is what lets the training corpus be labeled from the
 * compiled constraints on real inputs without a language-model teacher. The
 * top rule depends on a single field so that every input has a one-field
 * counterfactual twin, which the constraint-based adversarial teacher needs.
 */
export declare function assessRefundRisk(customer: Customer, order: Order): ScalarSemaResult<RefundRisk>;
