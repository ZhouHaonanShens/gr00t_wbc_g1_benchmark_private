# FIX_R2_A1_LOAD_08_R7_2_UPLIFT_PROBE_REPORT

## Provenance
- R7.0 recipe JSON sha256: `a9948f64750ea28084bca17056270deb1821512f278068c007b1d945c2a79fc6`
- R7.1 dryrun_report sha256: `1bfa76042d97286e6efa78370d6fa2ae2aa09acc2f189364f90ec83e95dd1c2b`
- R6.1 probe evidence sha256: `4af0098469e985e7921438cf80828d34c44e3684f03638c7bddd89248253af09`

## Counterfactual evolution table
| step | status | condition_sha_equal | l2_diff_max | verdict |
|---:|---|---|---:|---|
| 200 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 400 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 600 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 800 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 1000 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 1200 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 1400 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 1600 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 1800 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |
| 2000 | NOT_RUN_EARLY_STOP | N/A | N/A | N/A |

## Final verdict
verdict=TRAINING_FAILED; reason=crash; next=R7.2_debug

## Trial-1 evidence
- Trial artifact root: `agent/artifacts/recap_substrate_recovery/r7_2_uplift_probe/20260515T095201Z_trial_1`
- Trial report sha256: `201cae76fd5a7ce1a7929de24aedfe8cb38537807d097ec09b7b03c8cdddf0a1`
- Exchange evidence: `agent/exchange/r7_2_uplift_probe_trial_1_20260515T095201Z.json`
- Exchange evidence sha256: `e2c9f0af3acf038c0155f4eafcd9832957a79c081a0f44eea8f48c4ab96a84c7`
- Runtime log: `agent/runtime_logs/r7_2_uplift_probe/trial1_20260515T095201Z.log`
- GPU budget used: `0.046348` seconds on GPU1; <= 4 GPU-hr budget.
- Worker preflight reason: `entrypoint_unresolved`; the R7.2 worker failed closed before training instead of inventing a bespoke loop.

## Scope guard
R7.2 remained adapter-only and did not run R7.3, MuJoCo, C3 learned_value, or C4 advantage_embedding work. No base checkpoint file was modified and no LoRA adapter checkpoint was produced because trial-1 stopped at preflight.
