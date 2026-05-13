# R6.1/R6.2 forced runtime probe — A.2

- Status: `COMPLETE`
- Final runtime artifact root: `agent/artifacts/recap_substrate_recovery/r6_runtime_indicator_probe/20260513T135700Z_runtime/A.2/`
- Runtime log: `agent/runtime_logs/r6_1_runtime_probe_20260513T115831Z/probe_A2_final.log`
- Cell: `A.2`
- GPU: `1`
- Seed: `20000`
- max_steps_per_episode: `200`
- forced: `true`
- counterfactual: `true`
- Final successful run GPU minutes: `0.273`
- Cumulative R6.1 GPU minutes including failed/debug attempts: `4.463`

## Static/report citations

- R6.0 matrix sha256: `6cfe1c5460ed0b34c6447785700874f563a55e374c4454c11f6c7852c6d3817b`
- R6.0 report sha256: `96b7f0e407895dcebc657f472e99096d2b013adabe327975f49f4f6e62a61b90`
- R5 matrix sha256: `49f317f4259ea8187b35ce5d99a2788484829b1cee47566446deb2a45646dd84`
- R5 report sha256: `1960de2cc4497587ed5df3195d0c3d63984fe10449a58feb9f7dbd4f413b3fe4`
- Phase A report sha256: `c5b1dc1ef98a761a198b214a9d6e28f66fefa03f675eb807cb23ebc664e84777`

## Runtime output hashes

- `prompt_at_tokenizer_step0.txt`: `ee89c07a5d4a4a29c57be66297b5d3cead7b19e4b6dce8d51f744e8db3635b65`
- `runtime_trace.json`: `16404b218e123b1919ddc948b74b0eff4ecf44a774beae9585e4250255a4cf92`
- `counterfactual.json`: `ac350da405b0966c2d11bd08e3780c2f529ed9659153942b15293e16c3eb55c1`
- `FIX_R2_A1_LOAD_06_R6_RUNTIME_PROBE_REPORT.md`: `d4935a646b8383b2b8e98ffda30b6dcae5b35cd4c69a6cfdbd472f2d657ec66f`

## Decision facts

- Positive prompt indicator substring present: `true`
- Positive runtime verdict: `INDICATOR_PRESENT`
- Positive condition sha256: `12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126`
- Negative condition sha256: `12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126`
- condition_sha_equal: `true`
- first_5_actions_l2_diff: `(5.8529205746449264e-05, 0.00043130195695328943, 8.08885251383229e-05, 0.0003840019854801646, 9.919943576319756e-05)`
- counterfactual_verdict: `INDICATOR_INVARIANT`
- Exit decision: `branch=algorithm_fidelity_gap; estimated_fix_horizon=weeks`

## Scope guard

Only these final A.2 runtime files were emitted in the final runtime root:

1. `prompt_at_tokenizer_step0.txt`
2. `runtime_trace.json`
3. `counterfactual.json`
4. `FIX_R2_A1_LOAD_06_R6_RUNTIME_PROBE_REPORT.md`

No video, pickle, dataset writeback, checkpoint mutation, A.3/A.4/A.5 probe, or R2.1 30-episode rerun was performed.
