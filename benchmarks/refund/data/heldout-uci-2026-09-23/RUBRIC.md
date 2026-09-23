# Adjudication rubric: refund decision held-out sets

Version 1, 2026-09-23. Judge: Claude Fable 5.1 (`claude-fable-5-1`), acting in
an interactive Claude Code session. This rubric is the judge's committed reading
of the compiled refund function. It is never shown to the student model or to
the Sonnet 5 training teacher.

## The function under test

Behavior text: "Apply our refund policy. Enterprise customers get 60 days;
everyone else gets 30. Suspicious circumstances go to review."

Compiled constraints:

1. `never` approve when `order.status === "fraudulent"`.
2. `always` deny when `order.ageDays > 90`.

Inputs: `customer.priorRefunds` (count), `customer.tier` (`enterprise` or
`standard`), `order.ageDays`, `order.status` (`paid` or `fraudulent`),
`order.total`. Output: `approve`, `deny` or `review`.

## Decision procedure

Apply the steps in order; the first step that fires decides.

1. **Stale order.** `ageDays > 90` decides `deny`, whatever else is true. The
   always-deny constraint outranks every other consideration, including fraud.
2. **Fraud.** `status === "fraudulent"` (and `ageDays <= 90`) decides `review`.
   A fraudulent order is the clearest suspicious circumstance, and the policy
   sends suspicious circumstances to review rather than to an outright denial.
3. **Outside the tier window.** With `status === "paid"`, an order older than
   its window decides `deny`. Windows are inclusive: `enterprise` allows
   `ageDays <= 60`, `standard` allows `ageDays <= 30`. No grace period.
4. **Inside the window, suspicious history.** With `status === "paid"` and the
   order inside its window, the judge weighs the refund history and the size of
   the order. The case goes to `review` when the circumstances would make a
   reasonable operator pause before paying out; otherwise it is `approve`.
   Signals the judge treats as suspicious, individually or in combination:
   - a heavy refund history: `priorRefunds >= 5`, or `priorRefunds >= 3` on an
     order of at least 1,000;
   - an unusually large order for this retailer: `total >= 5,000`;
   - repeated same-day cancellations of large orders (`ageDays <= 1` with
     `total >= 2,000` and `priorRefunds >= 2`).
   A zero or low refund history on an ordinary order is approved even when the
   order is expensive by consumer standards, because this retailer's typical
   order is a few hundred and enterprise accounts routinely place orders in the
   thousands.
5. **Otherwise** `approve`.

## Notes on judgment

- Tier changes the window only. It is not itself evidence of good faith.
- `priorRefunds` counts earlier refund requests, not earlier approvals; the
  judge treats it as a history signal rather than a rule.
- Ties and borderline cases are decided toward `review`, never toward
  `approve`, when any suspicious signal is present.
- Every adjudication records a one-line rationale naming the step that fired.
