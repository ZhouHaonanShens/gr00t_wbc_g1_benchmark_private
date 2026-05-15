# FIX_R2_A1_LOAD_07 R7.1 Recipe Plumbing Report

## Summary

R7.1 implemented the minimum C1+C2+C5 recipe-plumbing subset behind default-OFF flags. The change is plumbing-only: it adds a recipe SSOT module, forwards non-default recipe flags through the wrapper/launcher chain, attaches non-default recipe config to runtime config/model, and runs a bounded subprocess dryrun smoke. No C3/C4 learned-value or advantage-embedding work was added.

## R7.0 recipe evidence

- Recipe fixture: `tests/recap/r7_recipe_diff/fixtures/first_run_A2/training_recipe_diff.json`
- sha256: `a9948f64750ea28084bca17056270deb1821512f278068c007b1d945c2a79fc6`
- R7.0 warning honored: the R7.1 flags were treated as future/plumbing args and made launcher-visible before any R7.2 LoRA run.

## Plumbing matrix

| Component | CLI flag(s) | entry_file:line | wired_to_module |
|---|---|---|---|
| C1 dual_loss | `--enable-dual-loss`, `--dual-loss-alpha` | `work/recap/r7_1_recipe_plumbing/flags.py:87-88` defines flags; `work/recap/finetune_full.py:1196` adds parser group; `work/recap/finetune_full.py:1339` forwards non-default args; `work/recap/launch_finetune_use_ddp.py:3002-3004` strips before tyro; `work/recap/launch_finetune_use_ddp.py:2897-2905` attaches runtime config | `work/recap/model.py:383-388` bridges to `DualLossConfig`; `work/recap/model.py:452-469` keeps existing dual-loss path and adds non-default R7.1 metadata |
| C2 indicator_dropout | `--indicator-dropout-p`, `--indicator-dropout-seed` | `work/recap/r7_1_recipe_plumbing/flags.py:89-90`; same wrapper/launcher path as C1 | `work/recap/r7_1_recipe_plumbing/indicator_dropout.py:7-16` pure caller-owned `random.Random` helper; `work/recap/model.py:386-388` bridges dropout to existing `DualLossConfig`; existing carrier text spike point verified at `work/recap/text_indicator.py:485-525` |
| C5 carrier_text_v1_grad_path | `--dual-loss-uses-carrier-text`, `--carrier-text-field` | `work/recap/r7_1_recipe_plumbing/flags.py:91-92`; same wrapper/launcher path as C1 | `work/recap/launch_finetune_use_ddp.py:2903` sets `config.data.task_text_field`; `work/recap/r7_1_recipe_plumbing/dual_loss_wiring.py:14-16` maps C5 metadata; `work/recap/model.py:469-471` emits non-default metadata |

## Default-OFF invariance

- No flags and explicit default-equivalent flags are normalized to no forwarded recipe args.
- `RecipeFlags.default()` and omitted recipe config both resolve to no R7.1 metadata in model helpers.
- Test evidence: `tests/recap/r7_1_recipe_plumbing/test_default_off_no_behavior_change.py` and `test_launcher_integration.py`.

## Dryrun smoke

- Command log: `agent/runtime_logs/r7_1_recipe_plumbing/dryrun_20260515T080533Z.log`
- Output root: `agent/artifacts/recap_substrate_recovery/r7_1_recipe_plumbing/20260515T080533Z_dryrun/`
- Report: `agent/artifacts/recap_substrate_recovery/r7_1_recipe_plumbing/20260515T080533Z_dryrun/dryrun_report.json`
- stdout: `agent/artifacts/recap_substrate_recovery/r7_1_recipe_plumbing/20260515T080533Z_dryrun/dryrun_stdout.log`
- stderr: `agent/artifacts/recap_substrate_recovery/r7_1_recipe_plumbing/20260515T080533Z_dryrun/dryrun_stderr.log`
- GPU policy: `--gpu 1`, `CUDA_VISIBLE_DEVICES=1`, child timeout capped at `<=120s` by `run_dryrun()`.
- `loss_finite`: `true`
- `loss_value`: `1.135`
- `gpu_seconds_used`: `0.03168813900265377`
- `subprocess_returncode`: `0`
- A.2 checkpoint path: `/home/howard/Projects/gr00t_wbc_g1_benchmark/agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/g2_main_v2_full_training/checkpoint-2200`

## Verification evidence

- Mandatory Ralph deslop pass: removed a single-use pass-through helper and split oversized new-module/test lines; post-deslop regressions remained green.

- `tests/recap/r7_1_recipe_plumbing`: `45 passed`
- R5/R6/R7 cohorts: `87 passed` with `--import-mode=importlib` and `CUDA_VISIBLE_DEVICES=`
- R2/R3 cohorts: `154 passed`
- Existing launcher/scope selected tests: `57 passed, 1 deselected`; deselected test depends on missing legacy artifact `agent/artifacts/stage3_ddp_smoke/run_c_gpu12_attempt01/green_smoke_candidate.json` and is unrelated to the R7.1 diff.
- New module LOC: 433 lines across seven files; `flags.py` 113, `cli.py` 100, `dryrun.py` 163.
- Existing-file deltas: `finetune_full.py +5`, `launch_finetune_use_ddp.py +33/-8`, `model.py +30/-1`; no existing file exceeded +40 LOC.
- C3/C4 guards: new module scan passed; brownfield diff scan passed.

## Commit chain

- Commit 1: `3f559ab r7.1: recipe plumbing scaffold + flags + dropout helpers`
- Commit 2: `a97ea61 r7.1: wire C1+C2+C5 flags into launch_finetune_use_ddp + finetune_full + model + dataset`
- Commit 3: this report and dryrun evidence pointer commit.

## R7.2 gate decision

Dryrun returned `loss_finite=true` with a finite smoke loss and no subprocess error. This satisfies the R7.1 plumbing smoke gate for entering R7.2 planning/execution. R7.2 remains out of scope for this report.
