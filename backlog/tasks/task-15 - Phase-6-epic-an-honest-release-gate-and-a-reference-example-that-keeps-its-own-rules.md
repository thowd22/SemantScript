---
id: TASK-15
title: >-
  Phase 6 epic: an honest release gate and a reference example that keeps its
  own rules
status: To Do
assignee: []
created_date: '2026-09-26 06:27'
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
- [ ] #1 The release gate scores every declared constraint on inputs the model never trained on, and a release that breaks a rule on such inputs is refused with the offending inputs named
- [ ] #2 The Express example's current release passes that gate and its README describes the release that is actually on disk
- [ ] #3 semantscript test refuses a stale or tampered artifact by default, and every failure remedy names explain or the stub artifact where they are the next step
<!-- AC:END -->
