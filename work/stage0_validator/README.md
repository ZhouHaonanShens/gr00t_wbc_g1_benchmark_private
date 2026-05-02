# Stage0 validator implementation

`work.stage0_validator.core` is the responsibility-bounded implementation for
Stage0 artifact validation.  `agent/run/stage0_artifact_validator.py` is kept as
a thin compatibility/public CLI wrapper for existing scripts and imports.

Migration rule: keep the active referee semantics stable.  Stage1 runtime lanes
should continue to treat the wrapper path as the active entrypoint unless an
independent verifier records migration parity PASS.
