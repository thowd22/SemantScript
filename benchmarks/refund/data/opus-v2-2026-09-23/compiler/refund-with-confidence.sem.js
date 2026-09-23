import { always, never, sema, } from "@semantscript/core";
import { __sema as __sema } from "@semantscript/core";
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
export function decideRefund(customer, order) {
    return __sema.call("nf_65e347f7dd8736c55d82e539be7ad005cedb3ca396dfa2ab89de619f77adfc9c", { customer, order });
}
//# sourceMappingURL=refund-with-confidence.sem.js.map