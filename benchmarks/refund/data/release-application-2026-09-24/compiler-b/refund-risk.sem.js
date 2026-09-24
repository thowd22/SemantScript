import { always, sema, } from "@semantscript/core";
import { __sema as __sema } from "@semantscript/core";
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
export function assessRefundRisk(customer, order) {
    return __sema.call("nf_3fb6038f6c6ef1ed5a2e8c674511fcfe8bf9b58d4a1f60af537eb620aaf07dbc", { customer, order });
}
//# sourceMappingURL=refund-risk.sem.js.map