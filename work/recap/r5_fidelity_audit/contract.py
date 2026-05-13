from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Verdict = Literal["IMPLEMENTED", "PARTIAL", "ABSENT", "UNCLEAR"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
IMPLEMENTED: Verdict = "IMPLEMENTED"; PARTIAL: Verdict = "PARTIAL"; ABSENT: Verdict = "ABSENT"; UNCLEAR: Verdict = "UNCLEAR"
HIGH: Confidence = "HIGH"; MEDIUM: Confidence = "MEDIUM"; LOW: Confidence = "LOW"
ALLOWED_VERDICTS: tuple[Verdict, ...] = (IMPLEMENTED, PARTIAL, ABSENT, UNCLEAR)
ALLOWED_CONFIDENCE: tuple[Confidence, ...] = (HIGH, MEDIUM, LOW)
QUESTION_MANIFEST_FILENAME = "fidelity_question_manifest.json"
QUESTION_REPORT_FILENAME = "fidelity_question_report.md"
FULL_REPORT_FILENAME = "gr00t_recap_fidelity_fact_report_v1.md"
MATRIX_REPORT_FILENAME = "gr00t_recap_fidelity_matrix.json"

class R5AuditError(RuntimeError):
    """Raised when R5 static audit inputs or CLI arguments are invalid."""

@dataclass(frozen=True)
class FidelityQuestion:
    qid: str
    title: str
    repo_files_to_inspect: tuple[str, ...]
    artifact_paths_to_inspect: tuple[str, ...]
    analyzer_name: str
    high_risk_active_path: bool = False

    @property
    def intent(self) -> str: return self.title
    @property
    def evidence_files(self) -> tuple[str, ...]: return self.repo_files_to_inspect
    @property
    def evidence_artifacts(self) -> tuple[str, ...]: return self.artifact_paths_to_inspect

FidelityQuestionSpec = FidelityQuestion

@dataclass(frozen=True)
class FidelityQuestionResult:
    question: FidelityQuestion
    repo_presence: Verdict
    active_path_consumption: Verdict
    confidence: Confidence
    evidence_files: tuple[str, ...]
    evidence_artifacts: tuple[str, ...]
    conclusion: str
    details: Mapping[str, Any] = field(default_factory=dict)

def _q(qid: str, title: str, files: tuple[str, ...], artifacts: tuple[str, ...], analyzer: str, high: bool = False) -> FidelityQuestion:
    return FidelityQuestion(qid, title, files, artifacts, analyzer, high)

FIDELITY_QUESTIONS: tuple[FidelityQuestion, ...] = (
    _q("Q1", "RECAP component coverage in work/recap/", ("work/recap/model.py", "work/recap/advantage.py", "work/recap/dual_loss.py", "work/recap/critic_vlm/", "work/recap/runtime_prompt.py"), ("agent/exchange/openpi_recap_fidelity_fact_report_v1.md",), "_analyze_q1"),
    _q("Q2", "learned value function / critic active in training path?", ("work/recap/critic_vlm/", "work/recap/critic_promotion.py", "work/recap/dual_loss.py", "work/recap/advantage.py"), ("agent/artifacts/recap_substrate_recovery/r2/.../experiment_cfg/",), "_analyze_q2", True),
    _q("Q3", "advantage_embedding source: learned or static relabel?", ("work/recap/advantage.py", "work/recap/label_dataset.py", "work/recap/scripts/3E_checkpoint_advantage_embedding_diff.py"), ("agent/artifacts/recap_substrate_recovery/r1_2_variant_audit/",), "_analyze_q3"),
    _q("Q4", "binary improvement indicator: threshold method?", ("work/recap/text_indicator.py", "work/recap/phase_thresholds.py", "work/recap/label_policy.py"), ("agent/artifacts/recap_substrate_recovery/r2_data_dossier/",), "_analyze_q4"),
    _q("Q5", "indicator placement in input sequence (training side)", ("work/recap/runtime_prompt.py", "work/recap/text_indicator.py", "work/recap/model.py"), ("agent/artifacts/recap_substrate_recovery/r2/.../experiment_cfg/",), "_analyze_q5"),
    _q("Q6", "loss objective: single MSE vs dual conditional/unconditional?", ("work/recap/dual_loss.py", "work/recap/finetune_full.py", "work/recap/launch_finetune_use_ddp.py", "submodules/gr00t/"), ("agent/artifacts/recap_substrate_recovery/r2/.../experiment_cfg/",), "_analyze_q6", True),
    _q("Q7", "indicator omission / dropout at training time?", ("work/recap/text_indicator.py", "work/recap/label_policy.py", "work/recap/dataset.py"), ("agent/artifacts/recap_substrate_recovery/r2_data_dossier/",), "_analyze_q7"),
    _q("Q8", "does the action head see indicator at runtime / rollout?", ("work/recap/policy.py", "work/recap/model.py", "work/recap/gr00t_main_recap.py", "work/recap/scripts/3D_recap_eval.py"), ("agent/artifacts/recap_substrate_recovery/r2/.../r2_1_authentic_eval/",), "_analyze_q8", True),
    _q("Q9", "A.2/A.3/A.4/A.5 axis-level differences active vs static", ("work/recap/r1_repro/ckpt_variant_audit.py", "work/recap/r2_authentic_eval/exclusion.py"), ("agent/artifacts/recap_substrate_recovery/fix_r2_a1_load/20260512T023640Z_phase_a/FIX_R2_A1_LOAD_01_PHASE_A_REPORT.md", ".reference/records/recap_a1_exclusion_record.md"), "_analyze_q9", True),
)
QUESTION_BY_ID: dict[str, FidelityQuestion] = {q.qid: q for q in FIDELITY_QUESTIONS}

def require_question(qid_or_question: str | FidelityQuestion) -> FidelityQuestion:
    if isinstance(qid_or_question, FidelityQuestion): return qid_or_question
    qid = str(qid_or_question).strip().upper()
    if qid in QUESTION_BY_ID: return QUESTION_BY_ID[qid]
    raise R5AuditError(f"unknown R5 fidelity question: {qid_or_question!r}")
