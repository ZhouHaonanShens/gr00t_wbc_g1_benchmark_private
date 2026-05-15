# R5 GR00T RECAP fidelity first run — 20260513T065300Z

- artifact_root: `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run`
- phase_report: `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run/FIX_R2_A1_LOAD_04_R5_FIDELITY_AUDIT_REPORT.md`
- fact_report: `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run/gr00t_recap_fidelity_fact_report_v1.md`
- matrix_json: `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run/gr00t_recap_fidelity_matrix.json`
- exit_label: `DETACHED_RUNTIME_PATH`
- scope: static audit only; 0 GPU / 0 training / 0 eval.

## Matrix

| Q | repo_presence | active_path_consumption | conclusion |
|---|---|---|---|
| Q1 | `IMPLEMENTED` | `PARTIAL` | RECAP component coverage is separated from active-path consumption evidence. |
| Q2 | `IMPLEMENTED` | `ABSENT` | Value/critic symbols count active only with checkpoint or config consumption evidence. |
| Q3 | `PARTIAL` | `ABSENT` | Advantage embedding is learned only with explicit learned/config evidence. |
| Q4 | `IMPLEMENTED` | `IMPLEMENTED` | Binary indicator threshold evidence must be explicit in literals or training config. |
| Q5 | `IMPLEMENTED` | `ABSENT` | Training-side placement requires placement symbols and active dataset/config wiring. |
| Q6 | `IMPLEMENTED` | `ABSENT` | Dual-loss fidelity is active only when conditional loss appears in training config/artifacts. |
| Q7 | `IMPLEMENTED` | `ABSENT` | Indicator dropout is absent unless explicit dropout code is wired into training config. |
| Q8 | `PARTIAL` | `ABSENT` | Runtime indicator consumption requires serving/rollout evidence, not symbol presence alone. |
| Q9 | `IMPLEMENTED` | `IMPLEMENTED` | A.2-A.5 axis fidelity uses the R2 evidence-grade cell SSOT and never infers A.1 coverage. |

## Verification

- R5 tests: 36 passed (`agent/runtime_logs/r5_fidelity_audit/20260513T065900Z_integrated_r5_pytest_after_testloc.log`)
- R3/R2 regression: 154 passed (`agent/runtime_logs/r5_fidelity_audit/20260513T065100Z_r3_r2_regression.log`)
- compileall: passed (`agent/runtime_logs/r5_fidelity_audit/20260513T065100Z_compileall.log`)
- CLI smoke: passed (`agent/runtime_logs/r5_fidelity_audit/20260513T065100Z_cli_smoke.log`)
- first audit run: passed (`agent/runtime_logs/r5_fidelity_audit/20260513T065300Z_audit_all.log`)
- LOC budget: implementation 659/850, tests 586/600.

## Checksums

- `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run/gr00t_recap_fidelity_matrix.json` — sha256 `49f317f4259ea8187b35ce5d99a2788484829b1cee47566446deb2a45646dd84`
- `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run/gr00t_recap_fidelity_fact_report_v1.md` — sha256 `55f4648b418b7cb8bd5151e24f33baf7f30f0aa788df5ebd2bff76872378ccf5`
- `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run/FIX_R2_A1_LOAD_04_R5_FIDELITY_AUDIT_REPORT.md` — sha256 `1960de2cc4497587ed5df3195d0c3d63984fe10449a58feb9f7dbd4f413b3fe4`
