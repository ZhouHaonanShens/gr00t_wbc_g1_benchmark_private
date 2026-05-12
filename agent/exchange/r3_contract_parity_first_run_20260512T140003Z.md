# R3 Contract Parity First Audit Run — 20260512T140003Z

- Run root: `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run`
- Runtime log: `agent/runtime_logs/r3_contract_parity_ultragoal_20260512_234010/20260512T140003Z_audit_all.log`
- Command: `timeout 300 .envs/main/bin/python -m work.recap.r3_contract_parity audit --all --output-root agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run`
- Cells: A.2, A.3, A.4, A.5
- Aggregate verdict: EXIT-COLD / all PASS
- Runtime literal: `runtime_invocations = []` present in all cell reports and summary.

## Git tracking note

`agent/artifacts` is an L0 symlink to HDD live storage and is ignored; Git refuses pathspecs beyond that symlink. This committed pointer records the first-run artifact paths and hashes while the canonical artifacts remain in `agent/artifacts`.

## Output file hashes

| path | sha256 |
|---|---|
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.2/cell_parity_manifest.json` | `21c5663346022ef00026a647b8d50aee6790d0d17b73c0636a196ff42ec09806` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.2/cell_parity_report.md` | `ee55f6a758c7c1d6e54dfbbad1a4a52754b0e56d57366fbb10be65f29b421a74` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.3/cell_parity_manifest.json` | `8bf89dfa86a2b804e13059f4807557069667162e0111a9911ba87b0d46e10512` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.3/cell_parity_report.md` | `29d0e8eca07910db03b3c13de0d5f8ae28c5f2341c6387f2f08ade1f8b71af0b` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.4/cell_parity_manifest.json` | `b507cdd1497efb55af28ace773567d9d474d2d9e73746b655dc669a39b366432` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.4/cell_parity_report.md` | `eea5a6b611ac29fbdd2295d99baf11b2751a70b4fa3c90a9623960f588da02bf` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.5/cell_parity_manifest.json` | `96a6d48e63ec04994425a8b1126557f15a2d195f9283497efa63b5023ddd7c6e` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/A.5/cell_parity_report.md` | `6b0a9105d6b56f7ad84c045bc6b1914e9adb03c61ce7c0ca6e715815ff5591df` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/r3_parity_summary.json` | `611c147557b2521fadb44fb62b2e64b9d4c063660654d3ce821cbb6077121167` |
| `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/r3_parity_summary.md` | `31dda0a6985ede4790457290d29ec9488938559802eea27bd6158c1214f530b3` |
