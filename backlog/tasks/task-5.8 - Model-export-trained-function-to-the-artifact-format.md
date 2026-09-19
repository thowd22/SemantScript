---
id: TASK-5.8
title: 'Model: export trained function to the artifact format'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - model
milestone: m-1
dependencies:
  - TASK-5.7
parent_task_id: TASK-5
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The runtime is Node and must load the model in-process. Export the encoder+head to ONNX and write the manifest defined in the IR/artifact task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Encoder and head export to ONNX and produce identical outputs to the PyTorch model on a fixed test batch
- [ ] #2 Artifact directory matches the documented layout including manifest, calibration and verification stats
- [ ] #3 Artifact loads without the training environment installed
<!-- AC:END -->
