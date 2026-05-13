# R5 Fidelity Audit First Run — <UTC>

- Run root: `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/<UTC>_run`
- Runtime log: `agent/runtime_logs/r5_fidelity_audit/<UTC>_audit_all.log`
- Command: `timeout 120 .envs/wbc/bin/python -m work.recap.r5_fidelity_audit audit --all --output-root agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/<UTC>_run`
- Overall label: `<overall_label from gr00t_recap_fidelity_matrix.json>`
- Report: `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/<UTC>_run/FIX_R2_A1_LOAD_04_R5_FIDELITY_AUDIT_REPORT.md`

## Git tracking note

`agent/artifacts` is an L0 symlink to HDD live storage and is ignored; keep this pointer tracked and leave runtime/artifact bodies in ignored storage.

## Output file hashes

| path | sha256 |
|---|---|
| `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/<UTC>_run/Q1/fidelity_question_manifest.json` | `<sha256>` |
| `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/<UTC>_run/Q1/fidelity_question_report.md` | `<sha256>` |
| `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/<UTC>_run/gr00t_recap_fidelity_matrix.json` | `<sha256>` |
| `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/<UTC>_run/gr00t_recap_fidelity_fact_report_v1.md` | `<sha256>` |
