# R7_RECIPE_DIFF_REPORT

- schema_version: `r7_training_recipe_diff_v1`
- base_cell: `A.2`
- generated_at_utc: `20260514T135756Z`
- source_openpi_report: `agent/exchange/openpi_recap_fidelity_fact_report_v1.md`

## Fidelity component summary

| component_id | current | paper-prescribed | diff_action | cli_addition |
|---|---|---|---|---|
| C1_dual_loss | ABSENT | IMPLEMENTED | ADD_LOSS_TERM | `--enable-dual-loss --dual-loss-alpha=0.5` |
| C2_indicator_dropout | ABSENT | IMPLEMENTED | ADD_DATASET_AUG | `--indicator-dropout-p=0.15 --indicator-dropout-seed=0` |
| C3_learned_value | ABSENT | IMPLEMENTED | ADD_LOSS_TERM | `--enable-learned-value-head --value-loss-alpha=0.1` |
| C4_advantage_embedding_active | PARTIAL | IMPLEMENTED | ENABLE_FLAG | `--action-head-advantage-input=enabled --advantage-embedding-dim=16` |
| C5_carrier_text_v1_grad_path | PARTIAL | IMPLEMENTED | ADD_CLI_ARG | `--dual-loss-uses-carrier-text --carrier-text-field=carrier_text_v1` |

## Per-component rationale

### C1_dual_loss — Dual conditional/unconditional loss objective
- cite: `OpenPI fidelity Q6`
- training_arg_name: `enable_dual_loss`
- state: `ABSENT` → `IMPLEMENTED`
- diff_action: `ADD_LOSS_TERM`
- required future args: `--enable-dual-loss --dual-loss-alpha=0.5`
- config_path_diff: `[['model.dual_loss.enabled', 'true'], ['training.dual_loss.alpha', '0.5']]`
- evidence_files: `['agent/exchange/openpi_recap_fidelity_fact_report_v1.md', 'work/recap/dual_loss.py', 'work/recap/launch_finetune_use_ddp.py', 'submodules/Isaac-GR00T/gr00t/configs/finetune_config.py']`
- rationale: dual-loss helper exists but launcher/config do not expose active conditional/unconditional training.

### C2_indicator_dropout — Stochastic indicator omission/dropout at training time
- cite: `OpenPI fidelity Q7`
- training_arg_name: `indicator_dropout_p`
- state: `ABSENT` → `IMPLEMENTED`
- diff_action: `ADD_DATASET_AUG`
- required future args: `--indicator-dropout-p=0.15 --indicator-dropout-seed=0`
- config_path_diff: `[['training.indicator_dropout.p', '0.15'], ['training.indicator_dropout.seed', '0']]`
- evidence_files: `['agent/exchange/openpi_recap_fidelity_fact_report_v1.md', 'work/recap/text_indicator.py', 'work/recap/scripts/34b_recap_numeric_adv_smoke.py', 'work/recap/launch_finetune_use_ddp.py']`
- rationale: dropout helpers or smoke flags exist, but stochastic omission is absent from active training config.

### C3_learned_value — Learned value-function / critic in training loop
- cite: `OpenPI fidelity Q2`
- training_arg_name: `enable_learned_value_head`
- state: `ABSENT` → `IMPLEMENTED`
- diff_action: `ADD_LOSS_TERM`
- required future args: `--enable-learned-value-head --value-loss-alpha=0.1`
- config_path_diff: `[['model.value_head.enabled', 'true'], ['training.value_loss.alpha', '0.1']]`
- evidence_files: `['agent/exchange/openpi_recap_fidelity_fact_report_v1.md', 'work/recap/advantage.py', 'work/recap/launch_finetune_use_ddp.py', 'submodules/Isaac-GR00T/gr00t/configs/finetune_config.py']`
- rationale: static value/advantage labels exist, but no learned value head is active in training.

### C4_advantage_embedding_active — advantage_input numeric sidecar consumed by action head
- cite: `OpenPI fidelity Q3`
- training_arg_name: `action_head_advantage_input`
- state: `PARTIAL` → `IMPLEMENTED`
- diff_action: `ENABLE_FLAG`
- required future args: `--action-head-advantage-input=enabled --advantage-embedding-dim=16`
- config_path_diff: `[['model.action_head.advantage_input.enabled', 'true'], ['model.action_head.advantage_embedding_dim', '16']]`
- evidence_files: `['agent/exchange/openpi_recap_fidelity_fact_report_v1.md', 'work/recap/advantage.py', 'work/recap/lerobot_export/dataset_export.py', 'work/recap/launch_finetune_use_ddp.py']`
- rationale: advantage_input sidecar exists in data paths, but active action-head consumption is not wired.

### C5_carrier_text_v1_grad_path — carrier_text_v1 tokens get gradient signal through dual-loss path
- cite: `OpenPI fidelity Q8 / GR00T R6.1 verdict`
- training_arg_name: `dual_loss_uses_carrier_text`
- state: `PARTIAL` → `IMPLEMENTED`
- diff_action: `ADD_CLI_ARG`
- required future args: `--dual-loss-uses-carrier-text --carrier-text-field=carrier_text_v1`
- config_path_diff: `[['training.dual_loss.uses_carrier_text', 'true'], ['data.task_text_field', 'carrier_text_v1']]`
- evidence_files: `['agent/exchange/openpi_recap_fidelity_fact_report_v1.md', 'work/recap/text_indicator.py', 'work/recap/lerobot_export/dataset_export.py', 'agent/exchange/r6_1_runtime_probe_a2_20260513.md']`
- rationale: carrier text reaches token surfaces, but R6/open-loop evidence does not prove gradient-sensitive action conditioning.

## Composable training recipe

These are required future args for R7.1; R7.0 does not execute them and the current launcher may not accept them yet.

```bash
--enable-dual-loss --dual-loss-alpha=0.5 --indicator-dropout-p=0.15 --indicator-dropout-seed=0 --enable-learned-value-head --value-loss-alpha=0.1 --action-head-advantage-input=enabled --advantage-embedding-dim=16 --dual-loss-uses-carrier-text --carrier-text-field=carrier_text_v1
```
