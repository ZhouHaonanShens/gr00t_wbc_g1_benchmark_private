# OpenPI RECAP fidelity 事实报告 v1

本报告只陈述当前仓库与现有 artifact 可直接支持的本地事实，不把 absent / unclear 等价成 implemented，也不包含修复建议。

## 审计范围与证据落点

- 代码证据：`work/openpi/**`、`work/recap/**`、`submodules/openpi/**` 的指定实现路径。
- 数据与 checkpoint 证据：`agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1`、`agent/artifacts/checkpoints/openpi_libero_variants`、`agent/artifacts/openpi_libero_v21/`。
- 本次新产物：`agent/artifacts/openpi_recap_fidelity_audit_v1`。

### 术语冻结

- 本文所有“存在 / 已实现”默认都拆成 `repo presence` 与 `active-path consumption` 两层。
- `repo presence` 只表示字段、route、checkpoint metadata 或 artifact 在仓库与证据包里存在。
- `active-path consumption` 才表示当前训练主路径或 rollout 主路径真的把它作为 live condition 消费。
- 本文冻结记号 `B0 = omit-control`，它对应现有 paired summary / provenance 里的历史 B control。

## Q1. 现在到底实现的是什么范围？

- 已实现的 RECAP-style 范围：离线 relabel dataset 内含 `recap_m2.value_V / advantage_A / advantage_input / indicator_I`，训练链路把 indicator 作为 prompt 文本 carrier 注入，再走 `pi05_libero` 的连续 action loss。
  - repo evidence：`work/openpi/scripts/materialize_libero_official_8d_recap_relabels.py:368-369`、`work/openpi/prompting/routes.py:186-226`、`submodules/openpi/src/openpi/training/config.py:684-745`
  - artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/data_lineage.json`、`sequence_layout_examples.md`、`loss_objective_report.json`
- 明确未实现：learned value function、dual conditional/unconditional objective、indicator omission/dropout、runtime indicator injection、online self-improvement loop。
  - repo evidence：`agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1/meta/info.json:158`、`submodules/openpi/scripts/train.py:137-193`、`work/openpi/prompting/routes.py:58-82`、`work/openpi/scripts/libero_rollout_eval_v2.py:474-525`
  - artifact：`paper_check_matrix.json`、`inference_path_report.json`
- 评测/发布基础设施：v21 manifest、paired summary、go/no-go、per-episode traces 属于实验外壳，不是算法本体。
  - repo/artifact：`agent/artifacts/openpi_libero_v21/paired_summary_abcx_v21.json`、`go_no_go_v21.json`、`runs/*/per_episode_trace.jsonl`

## Q2. 当前是否存在 learned value function / critic 路径？

- 结论：`active-path absent, repo presence exists for static relabel fields`。dataset contract 写明 `value_source='baseline'`、`critic_dir=null`，C provenance 还写明 `no_value_head=true`。当前可以证明 repo 内存在静态 relabel 的 `value_V` / `advantage_A` 字段，但没有本地 learned critic 训练入口，也没有后续 policy 训练或 rollout 主路径消费 learned value 的证据。
- repo evidence：`agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1/meta/info.json:158`
- artifact：`agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1/meta/info.json`、`agent/artifacts/checkpoints/openpi_libero_variants/recap_only_relabel8d_v2/checkpoint_provenance.json`、`agent/artifacts/openpi_recap_fidelity_audit_v1/paper_check_matrix.json`

## Q3. 当前的 recap_m2.* / advantage label 是怎么来的？

- 结论：来自静态 relabel baseline，不是 learned value function。粒度是 step-level / frame-level。`value_V` 来自按时间步 `t` 聚合的 mean return，`advantage_A = return_G - value_V`，`advantage_input` 是连续缩放值，`indicator_I` 是二值阈值化结果。
- repo evidence：`work/openpi/scripts/materialize_libero_official_8d_recap_relabels.py:368-369`、`work/openpi/scripts/materialize_libero_official_8d_recap_relabels.py:384-406`
- artifact：`agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1/meta/info.json`、`meta/relabel_stats_report.json`、`agent/artifacts/openpi_recap_fidelity_audit_v1/data_lineage.json`

## Q4. 当前是否真的实现了论文里的二值 improvement indicator？

- 结论：`部分实现 / approximated`。本地确有二值 `indicator_I`，但阈值 `epsilon_l` 来自全局 quantile，而不是 task-dependent threshold。本文冻结 `B0 = omit-control`，对应历史 artifact 里的 B control；`C` 喂真实 `indicator_I`；`X` 喂 deterministic shuffled indicator。
- repo evidence：`work/openpi/scripts/materialize_libero_official_8d_recap_relabels.py:384-406`、`work/openpi/prompting/routes.py:186-226`
- artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/data_lineage.json`、`sequence_layout_examples.md`、`token_diff_samples.json`

## Q5. indicator 在模型输入序列里的确切位置是什么？

- 结论：训练期 indicator 作为 `prompt_raw` 后追加的一行文本进入 prompt token prefix；随后模型再拼 action suffix。换句话说，它位于 prompt 文本后部、连续 action expert token 之前。运行期当前并不重放这条路径。
- repo evidence：`work/openpi/prompting/routes.py:186-226`、`work/openpi/scripts/libero_rollout_eval_v2.py:474-525`
- artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/sequence_layout_examples.md`、`token_diff_samples.json`、`inference_path_report.json`

## Q6. 当前 loss 到底是不是论文里的 conditional + unconditional 目标？

- 结论：`不是`。当前训练目标是单一连续 action flow-matching MSE，没有 dual conditional/unconditional loss，也没有第二个 value/token objective。`all_data_used` 只能判为 `unclear`。
- repo evidence：`submodules/openpi/scripts/train.py:137-193`、`submodules/openpi/src/openpi/training/config.py:684-745`、`submodules/openpi/src/openpi/training/data_loader.py:148-149`
- artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/loss_objective_report.json`

## Q7. 训练时是否有 indicator omission / dropout？

- 结论：`ABSENT for stochastic omission path`。本地只看到 `B0 = omit-control` 这条固定 control，也就是历史 artifact 里的 B control；没有 C 训练期随机 omission/dropout，也没有 CFG-like 双分支。
- repo evidence：`work/openpi/prompting/routes.py:58-82`
- artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/loss_objective_report.json`

## Q8. 当前 continuous action expert 是否真的“看见了” indicator？

- 结论：训练期 `看见`，运行期 `看不见`。训练期 prompt token 进 prefix，suffix action tokens 通过联合前向与 attention 使用这些 prompt tokens；且样本 token 长度 21/24 远低于 `max_token_len=180`，在这些示例里没有被截断。运行期 rollout 则只喂 `task_description`，不带 indicator carrier。
- repo evidence：`submodules/openpi/src/openpi/training/config.py:684-745`、`work/openpi/scripts/libero_rollout_eval_v2.py:474-525`
- artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/sequence_layout_examples.md`、`loss_objective_report.json`、`inference_path_report.json`

## Q9. 当前 B0 / C / X 的差异到底只剩下什么？

- 结论：训练数据边界本身被 provenance 证明是相同的；显式差异集中在 prompt route / consumer_mode 语义。本文冻结 `B0 = omit-control`，它对应历史 artifact 里的 B control；`C = real indicator`；`X = deterministic shuffled indicator`。但运行期三者最终都走同一个 `task_description` feeding 路径，这说明当前 `repo presence` 不能直接上升成 rollout `active-path consumption`。
- repo evidence：`agent/artifacts/checkpoints/openpi_libero_variants/fixedadv_relabel8d_control_v1/checkpoint_provenance.json:13-18,51-55`、`agent/artifacts/checkpoints/openpi_libero_variants/recap_only_relabel8d_v2/checkpoint_provenance.json:13-18,51-55`、`agent/artifacts/checkpoints/openpi_libero_variants/recap_shuffledadv_diag_v1/checkpoint_provenance.json:13-18,51-55`、`work/openpi/scripts/libero_rollout_eval_v2.py:474-525`
- artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/data_lineage.json`、`sequence_layout_examples.md`、`token_diff_samples.json`、`inference_path_report.json`

## Q10. C checkpoint 对 label 的 counterfactual sensitivity 是什么？

- 结论：当前 runtime 没有 label injection 机制，所以本次只能做同一 C checkpoint 的 prompt-level counterfactual probe。结果上，`removed_labels` 相对 `real_labels` 的平均 `l2_norm` 明显大于 `fixed_labels`，而 `shuffled_labels` 依样本而定；这说明 checkpoint 对 prompt carrier 不是完全不敏感，但当前 rollout 主路径没有消费它。
- 关键统计：`fixed=0.006570469161400977`，`shuffled=0.0035571193314021046`，`removed=0.04025790506202992`。
- repo evidence：`work/openpi/scripts/libero_rollout_eval_v2.py:474-525`、`work/openpi/prompting/routes.py:186-226`
- artifact：`agent/artifacts/openpi_recap_fidelity_audit_v1/counterfactual_sensitivity_report.json`

## Q11. 为什么 B0 会赢？

- 结论：在 strong v21 authority 上，`B0 = omit-control` 这条 inherited control path 赢的是 headline primary metric `success_rate@0.50_budget`，同时 throughput-like 指标也高于 C；并非只有 headline 一项。按 task 看，历史 B control 在 task 0 上 `1.0 vs 0.875`，在 task 1 上 `0.75 vs 0.75`。C 还有 `timeout_rate=0.020833...`，B 为 0。这个结果只能说明当前 detached runtime path 下，B0 比 C 更强，不能直接升级成 clean RECAP verdict。
- repo evidence：`work/openpi/scripts/libero_rollout_eval_v21.py:710-748`（per-episode trace 字段与 50%/75%/timeout 计算）`、`work/openpi/scripts/libero_rollout_eval_v21.py:1037-1078`（summary/trace 落盘 authority）
- artifact：`agent/artifacts/openpi_libero_v21/paired_summary_abcx_v21.json`、`go_no_go_v21.json`、`agent/artifacts/openpi_recap_fidelity_audit_v1/metric_breakdown_report.json`

## Q12. 为什么 C 没有拉开 X？

- 结论：在 v21 strong primary metric 上，`C-X delta=-0.020833...`，置信区间跨 0；task 级上 C task0=`0.875`、X task0=`1.0`，task1 也接近。结合 runtime 主路径没有 indicator injection，这构成一个强支持的风险解释，训练期 carrier 与 rollout 主路径脱钩。当前 `C` 路径负结果因此只能当作 gap evidence，不能写成 clean RECAP verdict。
- repo evidence：`work/openpi/scripts/libero_rollout_eval_v2.py:474-525`、`agent/artifacts/checkpoints/openpi_libero_variants/fixedadv_relabel8d_control_v1/checkpoint_provenance.json:13-18,51-55`、`agent/artifacts/checkpoints/openpi_libero_variants/recap_only_relabel8d_v2/checkpoint_provenance.json:13-18,51-55`、`agent/artifacts/checkpoints/openpi_libero_variants/recap_shuffledadv_diag_v1/checkpoint_provenance.json:13-18,51-55`
- artifact：`agent/artifacts/openpi_libero_v21/paired_summary_abcx_v21.json`、`agent/artifacts/openpi_recap_fidelity_audit_v1/metric_breakdown_report.json`、`inference_path_report.json`

## 审计结论（本地事实口径）

- headline winner：`B0 = omit-control`，对应 checkpoint `fixedadv_relabel8d_control_v1` 与 paired summary 里的历史 B control，primary metric=`success_rate@0.50_budget`。
- 当前仓库实现的是“静态 baseline relabel → prompt 文本 carrier → pi05_libero 连续 action loss”的 OpenPI 适配版，而不是论文原版的 learned value-derived advantage + dual objective + omission + runtime carrier 一致性路径。
- 当前证据必须拆开写成 `repo presence` 与 `active-path consumption`。前者已被多份 artifact 支持，后者在 runtime indicator carrier 上仍未闭环。
- 当前 `C` 路径负结果不是 clean RECAP verdict，只是 detached runtime path 下的 inherited negative result。
- 本轮审计已把这些事实完整落盘到 `agent/artifacts/openpi_recap_fidelity_audit_v1/` 与多份 exchange 文档中，可直接进入后续 paper-to-code gap 诊断。
