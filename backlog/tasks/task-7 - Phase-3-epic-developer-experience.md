---
id: TASK-7
title: 'Phase 3 epic: developer experience'
status: Done
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-25 01:29'
labels:
  - epic
milestone: m-3
dependencies:
  - TASK-6
  - TASK-12
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the primitive pleasant to use in projects that already exist. The adoption path is: npm install, semantscript init, write one sema expression in an existing file, semantscript dev trains it in the background, the editor shows accuracy. A separate framework is not required for adoption.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A developer can go from a fresh .sem.ts file to a running function using only the semantscript CLI
- [x] #2 An existing Express or Next.js app adopts sema with one dependency, one init command and one expression, with no other structural changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed on the subtasks' evidence: TASK-7.1 and 7.6 take a fresh .sem.ts to a running function with the CLI alone (init, build, train, test, run, with documented defaults); TASK-7.5 and 7.6 adopt sema in the committed Express and Next.js examples with one init command, one expression and the plugin entry, the one runtime addition being a loadSemaArtifact() call at startup, which 7.9 made path-free. 7.6's five-minute criterion and 7.9's Docker build remain unmeasured on this machine (no teacher key, no Docker), as their notes record.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 3 delivered the developer experience: the CLI (7.1), the build cache (7.2), the compile-time investigation (7.3), diagnostics (7.4), build-tool plugins for tsc, esbuild, Vite and Next.js (7.5), semantscript init with zero-config defaults (7.6), semantscript dev with incremental retraining and runtime hot-swap (7.7), the editor plugin (7.8) and deployment packaging with measured cold starts (7.9).
<!-- SECTION:FINAL_SUMMARY:END -->
