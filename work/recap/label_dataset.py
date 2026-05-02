#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

ITER_TAG = "recap_iter_000"
DATASET_DIR_REL = "agent/artifacts/recap_datasets"
CRITICS_DIR_REL = "agent/artifacts/critics"
RUNTIME_LOGS_REL = "agent/runtime_logs"

TOTAL_TIMEOUT_S = 180

VALUE_BASELINE = "t_mean_return"
VALUE_SOURCE = "critic"

EPSILON_STRATEGY = "quantile"
EPSILON_VALUE = 0.0
EPSILON_QUANTILE = 0.7

PROGRESS_JSON_NAME = "progress.json"
CLAIMS_SUBDIR_NAME = "_claims"
PROGRESS_SCHEMA_VERSION = "recap-m2-progress-v1"
FINALIZE_CLAIM_NAME = "__finalize__.claim.json"


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return cast(Path, fn(from_path=__file__))


@contextlib.contextmanager
def _tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(Path(log_path), header=str(header)):
        yield


def _install_alarm_timeout(timeout_s: float | None) -> None:
    if timeout_s is None:
        return
    try:
        t = int(float(timeout_s))
    except Exception:
        return
    if t <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Timed out after {t}s")

    _ = signal.signal(signal.SIGALRM, _handler)
    _ = signal.alarm(t)


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        try:
            _ = signal.alarm(0)
        except Exception:
            pass


def _git_head_and_dirty(repo_root: Path) -> tuple[str, bool]:
    head = "unknown"
    dirty = False
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception:
        head = "unknown"
    try:
        s = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        dirty = bool(str(s).strip())
    except Exception:
        dirty = False
    return str(head), bool(dirty)


def _resolve_dataset_dir(
    repo_root: Path, *, dataset_dir_rel: str, iter_tag: str
) -> Path:
    base = Path(str(dataset_dir_rel))
    base_abs = base if base.is_absolute() else (repo_root / base)

    candidate_iter = base_abs / str(iter_tag)
    if (candidate_iter / "episodes.jsonl").is_file() and (
        candidate_iter / "transitions.jsonl"
    ).is_file():
        return candidate_iter

    if (base_abs / "episodes.jsonl").is_file() and (
        base_abs / "transitions.jsonl"
    ).is_file():
        return base_abs

    return candidate_iter


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, obj: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, sort_keys=True)
        _ = f.write("\n")
    _ = tmp.replace(path)


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        raw: object = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(raw).__name__}")
    out: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"Expected str key in {path}, got {type(key).__name__}")
        out[key] = value
    return out


def _sanitize_token(value: str) -> str:
    out: list[str] = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append(f"_{ord(ch):02x}_")
    token = "".join(out).strip()
    return token or "item"


def _coerce_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like, got bool ({context})")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Invalid float-like str {value!r} ({context})") from e
    raise ValueError(f"Expected float-like, got {type(value).__name__} ({context})")


def _episode_shard_name(index: int, episode_id: str) -> str:
    return f"{int(index):06d}_{_sanitize_token(episode_id)}"


def _coerce_pid(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            pid = int(value)
        except ValueError:
            return None
        return pid if pid > 0 else None
    return None


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _claim_payload(kind: str, *, episode_id: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": str(kind),
        "hostname": str(socket.gethostname()),
        "pid": int(os.getpid()),
        "created_at": _now_iso(),
    }
    if episode_id is not None:
        payload["episode_id"] = str(episode_id)
    return payload


def _maybe_remove_stale_claim(path: Path) -> bool:
    try:
        payload = _read_json_object(path)
    except FileNotFoundError:
        return True
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return True

    hostname = payload.get("hostname")
    pid = _coerce_pid(payload.get("pid"))
    if (
        isinstance(hostname, str)
        and hostname == socket.gethostname()
        and not _pid_is_alive(pid or -1)
    ):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return True
    return False


def _try_acquire_claim(path: Path, *, payload: Mapping[str, object]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if _maybe_remove_stale_claim(path):
                continue
            return False

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
                _ = f.write("\n")
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            raise
        return True


def _release_claim(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _build_single_episode_dataset(
    dataset: Mapping[str, object],
    *,
    episode: dict[str, object],
    transitions: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "episodes": [dict(episode)],
        "transitions_by_episode": {
            str(episode["episode_id"]): [dict(tr) for tr in transitions]
        },
        "n_episodes": 1,
        "n_transitions": len(transitions),
        "dataset_dir": dataset.get("dataset_dir"),
    }


def _group_prelabels_by_episode(
    prelabels: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for rec in prelabels:
        episode_id = str(rec.get("episode_id", ""))
        out.setdefault(episode_id, []).append(dict(rec))
    return out


def _phase1_requires_global_prelabel_pass(value_source: str) -> bool:
    return str(value_source) in {"baseline", "critic"}


def _build_global_prelabels_by_episode(
    *,
    dataset: Mapping[str, object],
    build_m2_prelabels_fn: Callable[..., list[dict[str, object]]],
    value_baseline: str,
    value_source: str,
    critic_dir: str | None,
    code_version: str,
) -> dict[str, list[dict[str, object]]]:
    full_prelabels = build_m2_prelabels_fn(
        dataset,
        value_baseline=str(value_baseline),
        value_source=str(value_source),
        critic_dir=critic_dir,
        code_version_default=str(code_version),
    )
    return _group_prelabels_by_episode(full_prelabels)


def _build_resume_compatibility(
    *,
    code_version: str,
    iter_tag: str,
    dataset_dir: Path,
    value_baseline: str,
    value_source: str,
    critic_dir: Path | None,
    critic_config_metadata: Mapping[str, object],
    epsilon_strategy: str,
    epsilon_value: float,
    epsilon_quantile: float,
    n_total_episodes: int,
    n_transitions: int,
) -> dict[str, object]:
    return {
        "code_version": str(code_version),
        "iter_tag": str(iter_tag),
        "dataset_dir": str(dataset_dir),
        "value_baseline": str(value_baseline),
        "value_source": str(value_source),
        "critic_dir": str(critic_dir) if critic_dir is not None else None,
        "critic_backend_kind": critic_config_metadata.get("critic_backend_kind"),
        "critic_include_t": critic_config_metadata.get("critic_include_t"),
        "epsilon_strategy": str(epsilon_strategy),
        "epsilon_value": float(epsilon_value),
        "epsilon_quantile": float(epsilon_quantile),
        "n_total_episodes": int(n_total_episodes),
        "n_transitions": int(n_transitions),
    }


def _build_progress_manifest(
    *,
    compatibility: Mapping[str, object],
    processed_episode_ids: list[str],
    n_total_episodes: int,
    phase: str,
    epsilon_l: float | None = None,
    labels_path: Path | None = None,
    stats_path: Path | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": str(PROGRESS_SCHEMA_VERSION),
        "phase": str(phase),
        "completed": bool(phase == "completed"),
        "processed_episode_ids": list(processed_episode_ids),
        "n_processed_episodes": int(len(processed_episode_ids)),
        "n_total_episodes": int(n_total_episodes),
        "last_updated": _now_iso(),
        "compatibility": dict(compatibility),
    }
    if epsilon_l is not None:
        payload["epsilon_l"] = float(epsilon_l)
    if labels_path is not None:
        payload["labels_path"] = str(labels_path)
    if stats_path is not None:
        payload["stats_path"] = str(stats_path)
    return payload


def _processed_episode_ids_from_disk(
    episode_ids_in_order: list[str], shard_paths_by_episode: Mapping[str, Path]
) -> list[str]:
    return [
        episode_id
        for episode_id in episode_ids_in_order
        if shard_paths_by_episode[episode_id].is_file()
    ]


def _clear_restart_state(
    *,
    progress_path: Path,
    labels_path: Path,
    stats_path: Path,
    claims_dir: Path,
    private_prelabels_dir: Path,
) -> None:
    for path in (progress_path, labels_path, stats_path):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    if claims_dir.exists():
        shutil.rmtree(claims_dir)
    if private_prelabels_dir.exists():
        shutil.rmtree(private_prelabels_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="32_recap_label_dataset.py",
        description=(
            "RECAP offline labeler (M2): read M1 dataset jsonl/npz, compute G/V/A/I, and write m2_labels outputs."
        ),
    )
    p.add_argument("--iter-tag", type=str, default=str(ITER_TAG))
    p.add_argument("--dataset-dir-rel", type=str, default=str(DATASET_DIR_REL))
    p.add_argument("--runtime-logs-rel", type=str, default=str(RUNTIME_LOGS_REL))
    p.add_argument(
        "--total-timeout-s",
        type=float,
        default=float(TOTAL_TIMEOUT_S),
        help="Hard timeout fuse (best-effort) for the whole script.",
    )
    p.add_argument(
        "--value-baseline",
        type=str,
        choices=["mean_return", "t_mean_return"],
        default=str(VALUE_BASELINE),
    )
    p.add_argument(
        "--value-source",
        type=str,
        choices=["baseline", "critic"],
        default=str(VALUE_SOURCE),
        help=(
            "Where to source the value function for labeling. "
            "When set to 'critic', 32 will auto-dispatch by critic artifact metadata (default: baseline)."
        ),
    )
    p.add_argument(
        "--critic-dir",
        type=str,
        default=None,
        help=(
            "Path to a saved critic directory. If omitted in mainline mode, infer from iter_tag. "
            "32 auto-detects old state-only vs versioned multimodal backends from artifact metadata."
        ),
    )
    p.add_argument(
        "--epsilon-strategy",
        type=str,
        choices=["const", "quantile"],
        default=str(EPSILON_STRATEGY),
    )
    p.add_argument("--epsilon-value", type=float, default=float(EPSILON_VALUE))
    p.add_argument("--epsilon-quantile", type=float, default=float(EPSILON_QUANTILE))
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from compatible `m2_labels/progress.json` and private prelabel shards if present.",
    )
    p.add_argument(
        "--force-restart",
        action="store_true",
        help="Delete existing `m2_labels` resume state/public outputs for this dataset and start over.",
    )

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        p.add_argument(
            "--check-npz-keys",
            dest="check_npz_keys",
            action="store_true",
            default=True,
            help="Validate that arrays/*.npz files do not contain forbidden video/* keys (default: enabled).",
        )
        p.add_argument(
            "--no-check-npz-keys",
            dest="check_npz_keys",
            action="store_false",
            help="Disable NPZ key validation.",
        )
    else:
        p.add_argument(
            "--check-npz-keys",
            action=bool_action,
            default=True,
            help="Validate that arrays/*.npz files do not contain forbidden video/* keys.",
        )
    return p


def _resolve_critic_dir(repo_root: Path, critic_dir: str) -> Path:
    p = Path(str(critic_dir))
    p_abs = p if p.is_absolute() else (repo_root / p)
    return p_abs.resolve()


def _infer_critic_dir_from_iter_tag(repo_root: Path, *, iter_tag: str) -> Path:
    critics_root = (repo_root / CRITICS_DIR_REL).resolve()
    if not critics_root.is_dir():
        raise FileNotFoundError(f"Critics root not found: {critics_root}")
    candidates = [p for p in critics_root.glob(f"critic_{iter_tag}_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            "No critic directory matched iter_tag for mainline labeling: "
            f"iter_tag={iter_tag!r} critics_root={critics_root}"
        )
    for candidate in sorted(candidates):
        if candidate.name.endswith("_k0"):
            return candidate.resolve()
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def _load_critic_config_metadata(critic_dir: Path) -> dict[str, object]:
    config_path = critic_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"critic config.json not found under critic_dir: {critic_dir}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"critic config.json must contain a JSON object: {config_path}"
        )
    artifact_version = raw.get("artifact_version")
    critic_type = raw.get("critic_type")
    bin_centers = raw.get("bin_centers")
    if (
        artifact_version == "multimodal_distributional_v1"
        and critic_type == "multimodal_distributional_v1"
    ):
        return {
            "critic_backend_kind": "multimodal_distributional_v1",
            "critic_include_t": True,
            "critic_bins": len(bin_centers) if isinstance(bin_centers, list) else None,
            "critic_state_dim": raw.get("state_dim"),
        }
    return {
        "critic_backend_kind": "state_only_dist_bins",
        "critic_include_t": bool(raw.get("include_t", False)),
        "critic_bins": raw.get("bins"),
        "critic_state_dim": raw.get("state_dim"),
    }


def _atomic_update_json(path: Path, updates: dict[str, object]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing stats.json for update: {path}")

    data = _read_json_object(path)
    data.update(dict(updates))
    _write_json(path, data)


class RecapLabelDatasetWorkflow:
    def __init__(self) -> None:
        self.parser: argparse.ArgumentParser | None = None
        self.args: argparse.Namespace | None = None
        self.repo_root: Path | None = None
        self.value_source = str(VALUE_SOURCE)
        self.critic_dir_raw: object | None = None
        self.critic_dir_resolved: Path | None = None
        self.critic_config_metadata: dict[str, object] = {}
        self.iter_tag = str(ITER_TAG)
        self.runtime_dir: Path | None = None
        self.log_path: Path | None = None
        self.dataset_dir: Path | None = None
        self.t0_total = 0.0
        self.code_version = "unknown"
        self.read_m1_dataset: Any | None = None
        self.build_m2_prelabels: Any | None = None
        self.compute_m2_epsilon_l: Any | None = None
        self.finalize_m2_prelabels: Any | None = None
        self.list_m2_private_prelabel_shard_paths: Any | None = None
        self.m2_labels_dir: Any | None = None
        self.m2_private_prelabel_shard_path: Any | None = None
        self.m2_private_prelabels_dir: Any | None = None
        self.merge_m2_private_prelabel_shards: Any | None = None
        self.write_m2_label_outputs: Any | None = None
        self.write_m2_private_prelabel_shard_jsonl: Any | None = None
        self.labels_name = "labels.jsonl"
        self.stats_name = "stats.json"
        self.dataset: Mapping[str, object] | None = None
        self.episodes: list[dict[str, object]] = []
        self.transitions_by_episode: dict[str, list[dict[str, object]]] = {}
        self.episode_ids_in_order: list[str] = []
        self.episode_by_id: dict[str, dict[str, object]] = {}
        self.n_transitions = 0
        self.n_total_episodes = 0
        self.labels_dir: Path | None = None
        self.private_prelabels_dir: Path | None = None
        self.claims_dir: Path | None = None
        self.progress_path: Path | None = None
        self.labels_path: Path | None = None
        self.stats_path: Path | None = None
        self.shard_paths_by_episode: dict[str, Path] = {}
        self.compatibility: dict[str, object] = {}
        self.progress_manifest: dict[str, object] | None = None
        self.processed_episode_ids: list[str] = []
        self.global_prelabels_by_episode: dict[str, list[dict[str, object]]] | None = (
            None
        )

    def run(self) -> int:
        self.parser = _build_parser()
        if self._handle_help_request():
            return 0
        self._prepare_run_request()
        assert self.log_path is not None
        with _tee_stdio(self.log_path, header="32_recap_label_dataset"):
            _install_alarm_timeout(self._total_timeout_or_none())
            try:
                self._load_run_metadata_and_services()
                self._load_dataset_state()
                if self._prepare_resume_state():
                    return 0
                self._materialize_global_prelabels_if_needed()
                self._materialize_missing_episode_shards()
                if self._phase1_incomplete():
                    return 0
                return self._finalize_labels()
            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt -> stop early")
                return 130
            finally:
                _clear_alarm_timeout()

    def _handle_help_request(self) -> bool:
        assert self.parser is not None
        if not any(a in ("-h", "--help") for a in sys.argv[1:]):
            return False
        try:
            _ = self.parser.parse_args()
        except SystemExit as e:
            if int(getattr(e, "code", 0) or 0) != 0:
                raise
        return True

    def _prepare_run_request(self) -> None:
        assert self.parser is not None
        self.repo_root = _repo_root()
        self.args = self.parser.parse_args()
        if bool(getattr(self.args, "resume", False)) and bool(
            getattr(self.args, "force_restart", False)
        ):
            self.parser.error("--resume and --force-restart are mutually exclusive")
        self.value_source = str(
            getattr(self.args, "value_source", VALUE_SOURCE) or VALUE_SOURCE
        )
        self.critic_dir_raw = getattr(self.args, "critic_dir", None)
        self.iter_tag = str(getattr(self.args, "iter_tag", "") or ITER_TAG)
        self._resolve_critic_request()
        assert self.repo_root is not None
        self.runtime_dir = (
            self.repo_root
            / str(getattr(self.args, "runtime_logs_rel", RUNTIME_LOGS_REL))
            / self.iter_tag
        )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.runtime_dir / "label.log"
        self.dataset_dir = _resolve_dataset_dir(
            self.repo_root,
            dataset_dir_rel=str(getattr(self.args, "dataset_dir_rel", DATASET_DIR_REL)),
            iter_tag=str(self.iter_tag),
        )
        self.t0_total = time.monotonic()

    def _resolve_critic_request(self) -> None:
        assert self.repo_root is not None
        if self.value_source != "critic":
            self.critic_dir_resolved = None
            self.critic_config_metadata = {}
            return
        if self.critic_dir_raw:
            self.critic_dir_resolved = _resolve_critic_dir(
                self.repo_root, str(self.critic_dir_raw)
            )
        else:
            self.critic_dir_resolved = _infer_critic_dir_from_iter_tag(
                self.repo_root, iter_tag=str(self.iter_tag)
            )
        assert self.critic_dir_resolved is not None
        if not self.critic_dir_resolved.exists():
            raise FileNotFoundError(
                f"critic_dir does not exist: {self.critic_dir_resolved} (from {self.critic_dir_raw!r})"
            )
        if not self.critic_dir_resolved.is_dir():
            raise NotADirectoryError(
                f"critic_dir is not a directory: {self.critic_dir_resolved} (from {self.critic_dir_raw!r})"
            )
        self.critic_config_metadata = _load_critic_config_metadata(
            self.critic_dir_resolved
        )
        if not bool(self.critic_config_metadata.get("critic_include_t", False)):
            raise ValueError(
                "Continuous mainline requires critic include_t=true, "
                f"but got {self.critic_config_metadata!r} from {self.critic_dir_resolved / 'config.json'}"
            )
        paths_mod = importlib.import_module("work.demo_utils.paths")
        maybe_reexec_into_wbc_venv = getattr(paths_mod, "maybe_reexec_into_wbc_venv")
        maybe_reexec_into_wbc_venv(self.repo_root)

    def _total_timeout_or_none(self) -> float | None:
        assert self.args is not None
        timeout_s = float(getattr(self.args, "total_timeout_s", 0.0) or 0.0)
        return timeout_s or None

    def _load_run_metadata_and_services(self) -> None:
        assert self.repo_root is not None
        assert self.args is not None
        head, dirty = _git_head_and_dirty(self.repo_root)
        self.code_version = f"{head}{'-dirty' if dirty else ''}" if head else "unknown"
        print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
        print("[INFO] python:", sys.version.replace("\n", " "))
        print("[INFO] sys.executable:", sys.executable)
        print("[INFO] repo_root:", str(self.repo_root))
        print("[INFO] git_head:", str(head), "dirty:", bool(dirty))
        print("[INFO] iter_tag:", str(self.iter_tag))
        print("[INFO] dataset_dir:", str(self.dataset_dir))
        print("[INFO] runtime_dir:", str(self.runtime_dir))
        print("[INFO] value_baseline:", str(getattr(self.args, "value_baseline")))
        print("[INFO] value_source:", str(self.value_source))
        print(
            "[INFO] critic_dir:",
            str(self.critic_dir_resolved)
            if self.critic_dir_resolved is not None
            else "(none)",
        )
        if self.critic_config_metadata:
            print(
                "[INFO] critic_config:",
                json.dumps(
                    self.critic_config_metadata, ensure_ascii=True, sort_keys=True
                ),
            )
        print(
            "[INFO] epsilon:",
            f"strategy={str(getattr(self.args, 'epsilon_strategy'))}",
            f"epsilon_value={float(getattr(self.args, 'epsilon_value'))}",
            f"epsilon_quantile={float(getattr(self.args, 'epsilon_quantile'))}",
        )
        print(
            "[INFO] check_npz_keys:", bool(getattr(self.args, "check_npz_keys", True))
        )
        dataset_reader_mod = importlib.import_module("work.recap.dataset_reader")
        self.read_m1_dataset = getattr(dataset_reader_mod, "read_m1_dataset")
        labeler_mod = importlib.import_module("work.recap.labeler")
        self.build_m2_prelabels = getattr(labeler_mod, "build_m2_prelabels")
        self.compute_m2_epsilon_l = getattr(labeler_mod, "compute_m2_epsilon_l")
        self.finalize_m2_prelabels = getattr(labeler_mod, "finalize_m2_prelabels")
        label_writer_mod = importlib.import_module("work.recap.label_writer")
        self.list_m2_private_prelabel_shard_paths = getattr(
            label_writer_mod, "list_m2_private_prelabel_shard_paths"
        )
        self.m2_labels_dir = getattr(label_writer_mod, "m2_labels_dir")
        self.m2_private_prelabel_shard_path = getattr(
            label_writer_mod, "m2_private_prelabel_shard_path"
        )
        self.m2_private_prelabels_dir = getattr(
            label_writer_mod, "m2_private_prelabels_dir"
        )
        self.merge_m2_private_prelabel_shards = getattr(
            label_writer_mod, "merge_m2_private_prelabel_shards"
        )
        self.write_m2_label_outputs = getattr(
            label_writer_mod, "write_m2_label_outputs"
        )
        self.write_m2_private_prelabel_shard_jsonl = getattr(
            label_writer_mod, "write_m2_private_prelabel_shard_jsonl"
        )
        self.labels_name = str(
            getattr(label_writer_mod, "LABELS_JSONL_NAME", "labels.jsonl")
        )
        self.stats_name = str(
            getattr(label_writer_mod, "STATS_JSON_NAME", "stats.json")
        )

    def _load_dataset_state(self) -> None:
        assert self.args is not None
        assert self.dataset_dir is not None
        assert self.read_m1_dataset is not None
        assert self.m2_labels_dir is not None
        assert self.m2_private_prelabels_dir is not None
        assert self.m2_private_prelabel_shard_path is not None
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset directory: {self.dataset_dir} (expected M1 outputs under this directory)"
            )
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.dataset_dir}")
        self.dataset = self.read_m1_dataset(
            str(self.dataset_dir),
            check_npz_keys=bool(getattr(self.args, "check_npz_keys", True)),
        )
        dataset = self.dataset
        assert dataset is not None
        episodes_raw = dataset.get("episodes")
        transitions_by_episode_raw = dataset.get("transitions_by_episode")
        if not isinstance(episodes_raw, list):
            raise ValueError(
                f"dataset['episodes'] must be a list, got: {type(episodes_raw).__name__}"
            )
        if not isinstance(transitions_by_episode_raw, dict):
            raise ValueError(
                "dataset['transitions_by_episode'] must be a dict, got: "
                f"{type(transitions_by_episode_raw).__name__}"
            )
        self.episodes = cast(list[dict[str, object]], episodes_raw)
        self.transitions_by_episode = cast(
            dict[str, list[dict[str, object]]], transitions_by_episode_raw
        )
        self.episode_ids_in_order = [
            cast(str, ep["episode_id"])
            for ep in self.episodes
            if isinstance(ep.get("episode_id"), str)
        ]
        self.episode_by_id = {}
        for ep in self.episodes:
            episode_id_obj = ep.get("episode_id")
            if isinstance(episode_id_obj, str) and episode_id_obj:
                self.episode_by_id[episode_id_obj] = ep
        self.n_transitions = int(cast(int, dataset.get("n_transitions", 0)) or 0)
        self.n_total_episodes = int(
            cast(int, dataset.get("n_episodes", len(self.episode_ids_in_order)))
        )
        print("[INFO] input transitions line count:", int(self.n_transitions))
        print("[INFO] input episode count:", int(self.n_total_episodes))
        if self.n_transitions <= 0:
            raise ValueError("No transitions found in transitions.jsonl")
        self.labels_dir = cast(Path, self.m2_labels_dir(str(self.dataset_dir)))
        self.private_prelabels_dir = cast(
            Path, self.m2_private_prelabels_dir(str(self.dataset_dir))
        )
        self.claims_dir = self.labels_dir / CLAIMS_SUBDIR_NAME
        self.progress_path = self.labels_dir / PROGRESS_JSON_NAME
        self.labels_path = self.labels_dir / self.labels_name
        self.stats_path = self.labels_dir / self.stats_name
        self.shard_paths_by_episode = {
            episode_id: cast(
                Path,
                self.m2_private_prelabel_shard_path(
                    str(self.dataset_dir),
                    shard_name=_episode_shard_name(index, episode_id),
                ),
            )
            for index, episode_id in enumerate(self.episode_ids_in_order)
        }

    def _prepare_resume_state(self) -> bool:
        assert self.args is not None
        assert self.dataset_dir is not None
        assert self.progress_path is not None
        assert self.labels_path is not None
        assert self.stats_path is not None
        assert self.claims_dir is not None
        assert self.private_prelabels_dir is not None
        assert self.list_m2_private_prelabel_shard_paths is not None
        existing_private_shards = list(
            cast(
                list[Path],
                self.list_m2_private_prelabel_shard_paths(str(self.dataset_dir)),
            )
        )
        expected_shard_paths = set(self.shard_paths_by_episode.values())
        extra_shard_paths = [
            path for path in existing_private_shards if path not in expected_shard_paths
        ]
        if bool(getattr(self.args, "force_restart", False)):
            print("[INFO] force_restart: clearing existing m2 state")
            _clear_restart_state(
                progress_path=self.progress_path,
                labels_path=self.labels_path,
                stats_path=self.stats_path,
                claims_dir=self.claims_dir,
                private_prelabels_dir=self.private_prelabels_dir,
            )
            existing_private_shards = []
            extra_shard_paths = []
        existing_state_detected = bool(
            self.progress_path.exists()
            or self.labels_path.exists()
            or self.stats_path.exists()
            or existing_private_shards
            or (self.claims_dir.exists() and any(self.claims_dir.iterdir()))
        )
        if (
            existing_state_detected
            and not bool(getattr(self.args, "resume", False))
            and not bool(getattr(self.args, "force_restart", False))
        ):
            raise ValueError(
                "Existing resumable labeling state detected under m2_labels/. "
                "Use --resume to continue or --force-restart to clear it."
            )
        self.compatibility = _build_resume_compatibility(
            code_version=str(self.code_version),
            iter_tag=str(self.iter_tag),
            dataset_dir=self.dataset_dir,
            value_baseline=str(getattr(self.args, "value_baseline")),
            value_source=str(self.value_source),
            critic_dir=self.critic_dir_resolved,
            critic_config_metadata=self.critic_config_metadata,
            epsilon_strategy=str(getattr(self.args, "epsilon_strategy")),
            epsilon_value=float(getattr(self.args, "epsilon_value")),
            epsilon_quantile=float(getattr(self.args, "epsilon_quantile")),
            n_total_episodes=self.n_total_episodes,
            n_transitions=self.n_transitions,
        )
        if extra_shard_paths:
            extra_preview = ", ".join(path.name for path in extra_shard_paths[:5])
            more = (
                ""
                if len(extra_shard_paths) <= 5
                else f" (+{len(extra_shard_paths) - 5} more)"
            )
            raise ValueError(
                "Found stale/unexpected private prelabel shards for this dataset. "
                f"Use --force-restart to clear them: {extra_preview}{more}"
            )
        self.progress_manifest = None
        if self.progress_path.is_file():
            self.progress_manifest = _read_json_object(self.progress_path)
            if self.progress_manifest.get("compatibility") != self.compatibility:
                raise ValueError(
                    "Existing progress manifest is incompatible with this run. "
                    "Use --force-restart to clear stale state."
                )
        self.processed_episode_ids = _processed_episode_ids_from_disk(
            self.episode_ids_in_order, self.shard_paths_by_episode
        )
        print(
            "[INFO] resume_state:",
            f"resume={bool(getattr(self.args, 'resume', False))}",
            f"processed_episodes={len(self.processed_episode_ids)}/{len(self.episode_ids_in_order)}",
        )
        if (
            self.progress_manifest is not None
            and self.progress_manifest.get("completed") is True
            and self.labels_path.is_file()
            and self.stats_path.is_file()
            and len(self.processed_episode_ids) == len(self.episode_ids_in_order)
        ):
            print("[INFO] existing completed progress detected; nothing to do")
            return True
        _write_json(
            self.progress_path,
            _build_progress_manifest(
                compatibility=self.compatibility,
                processed_episode_ids=self.processed_episode_ids,
                n_total_episodes=len(self.episode_ids_in_order),
                phase=(
                    "ready_to_finalize"
                    if len(self.processed_episode_ids) == len(self.episode_ids_in_order)
                    else "prelabeling"
                ),
            ),
        )
        self._timeout_manual_check()
        return False

    def _materialize_global_prelabels_if_needed(self) -> None:
        assert self.args is not None
        assert self.build_m2_prelabels is not None
        assert self.dataset is not None
        self.global_prelabels_by_episode = None
        missing_episode_ids = [
            episode_id
            for episode_id in self.episode_ids_in_order
            if episode_id not in self.processed_episode_ids
        ]
        if not missing_episode_ids or not _phase1_requires_global_prelabel_pass(
            self.value_source
        ):
            return
        if self.value_source == "baseline":
            print(
                "[INFO] baseline prelabel pass: computing full-dataset prelabels once for semantic fidelity"
            )
        elif self.value_source == "critic":
            print(
                "[INFO] critic prelabel pass: computing full-dataset prelabels once before episode shard materialization"
            )
        self.global_prelabels_by_episode = _build_global_prelabels_by_episode(
            dataset=self.dataset,
            build_m2_prelabels_fn=self.build_m2_prelabels,
            value_baseline=str(getattr(self.args, "value_baseline")),
            value_source=str(self.value_source),
            critic_dir=(
                str(self.critic_dir_resolved)
                if self.critic_dir_resolved is not None
                else None
            ),
            code_version=str(self.code_version),
        )

    def _materialize_missing_episode_shards(self) -> None:
        assert self.args is not None
        assert self.dataset is not None
        assert self.dataset_dir is not None
        assert self.claims_dir is not None
        assert self.progress_path is not None
        assert self.build_m2_prelabels is not None
        assert self.write_m2_private_prelabel_shard_jsonl is not None
        for index, episode_id in enumerate(self.episode_ids_in_order):
            shard_path = self.shard_paths_by_episode[episode_id]
            if shard_path.is_file():
                continue
            claim_path = self.claims_dir / (
                _episode_shard_name(index, episode_id) + ".claim.json"
            )
            if not _try_acquire_claim(
                claim_path,
                payload=_claim_payload("episode", episode_id=episode_id),
            ):
                if shard_path.is_file():
                    print(
                        "[INFO] shard already materialized while claiming:", episode_id
                    )
                else:
                    print("[INFO] skip claimed episode:", episode_id)
                continue
            try:
                if shard_path.is_file():
                    print("[INFO] shard already exists after claim:", episode_id)
                    continue
                episode_prelabels = self._build_episode_prelabels(episode_id)
                if not episode_prelabels:
                    raise ValueError(
                        f"No prelabels generated for episode_id={episode_id}"
                    )
                _ = self.write_m2_private_prelabel_shard_jsonl(
                    str(self.dataset_dir),
                    shard_name=_episode_shard_name(index, episode_id),
                    records=episode_prelabels,
                )
                self.processed_episode_ids = _processed_episode_ids_from_disk(
                    self.episode_ids_in_order, self.shard_paths_by_episode
                )
                _write_json(
                    self.progress_path,
                    _build_progress_manifest(
                        compatibility=self.compatibility,
                        processed_episode_ids=self.processed_episode_ids,
                        n_total_episodes=len(self.episode_ids_in_order),
                        phase=(
                            "ready_to_finalize"
                            if len(self.processed_episode_ids)
                            == len(self.episode_ids_in_order)
                            else "prelabeling"
                        ),
                    ),
                )
                print(
                    "[INFO] wrote prelabel shard:",
                    episode_id,
                    f"records={len(episode_prelabels)}",
                    str(shard_path),
                )
            finally:
                _release_claim(claim_path)

    def _build_episode_prelabels(self, episode_id: str) -> list[dict[str, object]]:
        assert self.args is not None
        assert self.dataset is not None
        assert self.build_m2_prelabels is not None
        if self.global_prelabels_by_episode is not None:
            return list(self.global_prelabels_by_episode.get(episode_id, []))
        if self.value_source == "critic":
            raise RuntimeError(
                "critic phase-1 must use the full-dataset global prelabel pass before shard writes"
            )
        return cast(
            list[dict[str, object]],
            self.build_m2_prelabels(
                _build_single_episode_dataset(
                    self.dataset,
                    episode=self.episode_by_id[episode_id],
                    transitions=self.transitions_by_episode.get(episode_id, []),
                ),
                value_baseline=str(getattr(self.args, "value_baseline")),
                value_source=str(self.value_source),
                critic_dir=(
                    str(self.critic_dir_resolved)
                    if self.critic_dir_resolved is not None
                    else None
                ),
                code_version_default=str(self.code_version),
            ),
        )

    def _phase1_incomplete(self) -> bool:
        assert self.progress_path is not None
        self.processed_episode_ids = _processed_episode_ids_from_disk(
            self.episode_ids_in_order, self.shard_paths_by_episode
        )
        if len(self.processed_episode_ids) == len(self.episode_ids_in_order):
            return False
        _write_json(
            self.progress_path,
            _build_progress_manifest(
                compatibility=self.compatibility,
                processed_episode_ids=self.processed_episode_ids,
                n_total_episodes=len(self.episode_ids_in_order),
                phase="prelabeling",
            ),
        )
        print(
            "[INFO] phase-1 incomplete:",
            f"processed={len(self.processed_episode_ids)}/{len(self.episode_ids_in_order)}",
            "resume later to continue or let other workers finish remaining episodes",
        )
        return True

    def _finalize_labels(self) -> int:
        assert self.args is not None
        assert self.dataset_dir is not None
        assert self.claims_dir is not None
        assert self.progress_path is not None
        assert self.labels_path is not None
        assert self.stats_path is not None
        assert self.merge_m2_private_prelabel_shards is not None
        assert self.compute_m2_epsilon_l is not None
        assert self.finalize_m2_prelabels is not None
        assert self.write_m2_label_outputs is not None
        finalize_claim_path = self.claims_dir / FINALIZE_CLAIM_NAME
        if not _try_acquire_claim(
            finalize_claim_path, payload=_claim_payload("finalize")
        ):
            print("[INFO] finalize already claimed by another worker; exiting cleanly")
            return 0
        try:
            _write_json(
                self.progress_path,
                _build_progress_manifest(
                    compatibility=self.compatibility,
                    processed_episode_ids=self.processed_episode_ids,
                    n_total_episodes=len(self.episode_ids_in_order),
                    phase="finalizing",
                ),
            )
            merged_prelabels = cast(
                list[dict[str, object]],
                self.merge_m2_private_prelabel_shards(str(self.dataset_dir)),
            )
            if len(merged_prelabels) != self.n_transitions:
                raise ValueError(
                    "Merged prelabel count mismatch: "
                    f"transitions={self.n_transitions} merged_prelabels={len(merged_prelabels)}"
                )
            epsilon_used = float(
                self.compute_m2_epsilon_l(
                    merged_prelabels,
                    epsilon_strategy=str(getattr(self.args, "epsilon_strategy")),
                    epsilon_value=float(getattr(self.args, "epsilon_value")),
                    epsilon_quantile=float(getattr(self.args, "epsilon_quantile")),
                )
            )
            labels = cast(
                list[dict[str, object]],
                self.finalize_m2_prelabels(merged_prelabels, epsilon_l=epsilon_used),
            )
            n_labels = int(len(labels))
            print("[INFO] output labels line count:", int(n_labels))
            if n_labels != self.n_transitions:
                raise ValueError(
                    f"Label count mismatch: transitions={self.n_transitions} labels={n_labels}"
                )
            if not labels:
                raise ValueError("No labels generated")
            print(
                "[INFO] epsilon used:",
                f"strategy={str(getattr(self.args, 'epsilon_strategy'))}",
                f"epsilon_l={float(epsilon_used)}",
            )
            stats = self.write_m2_label_outputs(
                str(self.dataset_dir),
                labels,
                epsilon_strategy=str(getattr(self.args, "epsilon_strategy")),
                epsilon_value=float(epsilon_used),
            )
            advantage_mod = importlib.import_module("work.recap.advantage")
            compute_sign_scales = getattr(
                advantage_mod, "compute_sign_aware_advantage_scales"
            )
            build_contract = getattr(advantage_mod, "build_advantage_contract_metadata")
            sign_scale_summary = compute_sign_scales(
                [
                    _coerce_float(
                        label.get("advantage_A"),
                        context="label.advantage_A",
                    )
                    for label in labels
                ],
                context=(
                    f"label_dataset.{str(getattr(self.args, 'iter_tag'))}"
                    ".continuous_advantage_contract"
                ),
            )
            positive_scale_raw = sign_scale_summary.get("positive_scale")
            negative_scale_abs_raw = sign_scale_summary.get("negative_scale_abs")
            contract = build_contract(
                source_iter_tag=str(getattr(self.args, "iter_tag")),
                n_samples=len(labels),
                positive_scale=(
                    float(positive_scale_raw)
                    if positive_scale_raw is not None
                    else None
                ),
                negative_scale_abs=(
                    float(negative_scale_abs_raw)
                    if negative_scale_abs_raw is not None
                    else None
                ),
                critic_dir=(
                    str(self.critic_dir_resolved)
                    if self.critic_dir_resolved is not None
                    else None
                ),
                critic_include_t=bool(
                    self.critic_config_metadata.get("critic_include_t", False)
                ),
                advantage_stats={
                    "value_source": str(self.value_source),
                    **self.critic_config_metadata,
                },
                sign_scale_summary=dict(sign_scale_summary),
            )
            contract_path = (
                self.dataset_dir / "m2_labels" / "continuous_advantage_contract.json"
            )
            _write_json(contract_path, contract)
            pos_ratio = stats.get("pos_ratio")
            print("[INFO] indicator_I positive ratio:", pos_ratio)
            _atomic_update_json(
                self.stats_path,
                {
                    "value_source": str(self.value_source),
                    "critic_dir": (
                        str(self.critic_dir_resolved)
                        if self.critic_dir_resolved is not None
                        else None
                    ),
                    **self.critic_config_metadata,
                },
            )
            _write_json(
                self.progress_path,
                _build_progress_manifest(
                    compatibility=self.compatibility,
                    processed_episode_ids=self.processed_episode_ids,
                    n_total_episodes=len(self.episode_ids_in_order),
                    phase="completed",
                    epsilon_l=float(epsilon_used),
                    labels_path=self.labels_path,
                    stats_path=self.stats_path,
                ),
            )
            print("[INFO] labels_path:", str(self.labels_path))
            print("[INFO] stats_path:", str(self.stats_path))
            print("[INFO] contract_path:", str(contract_path))
            return 0
        finally:
            _release_claim(finalize_claim_path)

    def _timeout_manual_check(self) -> None:
        assert self.args is not None
        if hasattr(signal, "SIGALRM"):
            return
        elapsed = time.monotonic() - self.t0_total
        if float(
            getattr(self.args, "total_timeout_s", 0.0) or 0.0
        ) > 0 and elapsed > float(getattr(self.args, "total_timeout_s")):
            raise TimeoutError(f"Timed out after {int(elapsed)}s (manual check)")


def main() -> int:
    return RecapLabelDatasetWorkflow().run()


if __name__ == "__main__":
    raise SystemExit(main())


class RecapLabelDatasetScriptApp:
    def run(self) -> int:
        return main()
