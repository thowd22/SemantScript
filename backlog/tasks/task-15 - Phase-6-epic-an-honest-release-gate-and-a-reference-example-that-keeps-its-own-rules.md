---
id: TASK-15
title: >-
  Phase 6 epic: an honest release gate and a reference example that keeps its
  own rules
status: To Do
assignee: []
created_date: '2026-09-26 06:27'
updated_date: '2026-09-26 11:48'
labels:
  - dx
  - quality
milestone: m-6
dependencies: []
priority: high
ordinal: 65000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 5 made SemantScript installable, debuggable and deployable. Its last review round found the gap that matters most for trust: the Express example's current release (manifest 217d386c, seed 5 of the TASK-14.5 reduced-prompt comparison, refund accuracy 0.924) answers review or approve for paid and fraudulent orders at 100, 110, 120, 150 and 200 days although the expression declares always(() => order.ageDays > 90, 'deny'), and the release gate recorded zero constraint violations. The verifier's constraint check (trainer/src/semantscript_trainer/verification.py, _constraint_violations over _case_records) scores only the training-corpus records (gold, synthetic and adversarial cases the model trained on), so a model that memorises the boundary cases passes while failing the interior of the region. semantscript explain (TASK-14.7) is what exposed it. This epic makes the gate honest, retrains the example under it, and closes the smaller test-honesty follow-ups the Phase 5 reviewers raised. Publishing (TASK-14.1) is deliberately out of scope until the user sets up the registries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The release gate scores every declared constraint on inputs the model never trained on, and a release that breaks a rule on such inputs is refused with the offending inputs named
- [ ] #2 The Express example's current release passes that gate and its README describes the release that is actually on disk
- [x] #3 semantscript test refuses a stale or tampered artifact by default, and every failure remedy names explain or the stub artifact where they are the next step
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-26 AC3 evidence on main fc90858: a copy of the Express release with 4 bytes overwritten in models/adapters/application.onnx makes 'semantscript test --no-bundle --artifact <copy>' exit 1 with SEMA_ARTIFACT_INTEGRITY and a next: line (TASK-15.3); plain 'semantscript test' in examples/express-app finds the build's bundle and replays 3/3 examples per function; diagnostics/remedies.json entries gold-check-example, runtime-not-loaded, artifact-missing, unknown-function, test-example-mismatch and editor-no-artifact name semantscript explain or @semantscript/core/testing (TASK-15.4); generate-remedies --check passes.

2026-09-26 AC1 evidence (TASK-15.1 on main, TASK-15.2 reports): the verifier draws a held-out sample of 512 inputs from the input types and constraint predicates (none a training input, fixed by --seed) and refuses a release over the 1% tolerance naming each constraint and the offending inputs, e.g. train-report-15.2-e16sb-seed2: 'held-out constraint check failed on 20 of 512 sampled inputs (0.039 exceeds the configured tolerance 0.01; seed 2; broken: constraint 0 on 6, constraint 2 on 3, constraint 3 on 8, constraint 5 on 3)' followed by the inputs and the model's answers; the manifest records verification.heldOutConstraints. AC2 stays open: every one of 27 free retrains from the cached datasets fails the gate (best 20 of 512), so the Express example still serves 217d386c and the user must approve teacher spend (USD 0.39 to 2.02 estimated) or rescope TASK-15.2.
<!-- SECTION:NOTES:END -->
