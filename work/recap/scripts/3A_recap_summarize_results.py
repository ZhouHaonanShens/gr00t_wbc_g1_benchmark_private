#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_P3A_DIR_REL = Path("agent/artifacts/p3A")
DEFAULT_RECAP_DATASETS_DIR_REL = Path("agent/artifacts/recap_datasets")
DEFAULT_CRITICS_DIR_REL = Path("agent/artifacts/critics")
DEFAULT_PLOTS_DIR_REL = Path("agent/artifacts/plots")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{i}, got {type(obj).__name__}"
                )
            items.append(obj)
    return items


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    tmp.replace(path)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for w, c in zip(widths, cells):
            parts.append(c.ljust(w))
        return "| " + " | ".join(parts) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out: list[str] = []
    out.append(sep)
    out.append(fmt_row(headers))
    out.append(sep)
    for r in rows:
        out.append(fmt_row(r))
    out.append(sep)
    return "\n".join(out)


def _coerce_bool(x: Any, *, context: str) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "false"):
            return s == "true"
        if s in ("0", "1"):
            return s == "1"
    raise ValueError(f"Expected bool-like value for {context}, got {x!r}")


def _fmt_num(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, (int, float)):
        if isinstance(x, int):
            return str(x)
        s = "%.6g" % float(x)
        if ("e" not in s) and ("E" not in s) and ("." in s):
            s = s.rstrip("0").rstrip(".")
        return s
    return str(x)


def _extract_eval_tags(manifest: dict[str, Any]) -> tuple[str, list[tuple[int, str]]]:
    iters = manifest.get("iterations")
    if not isinstance(iters, list):
        raise ValueError("manifest['iterations'] must be a list")

    baseline_tag: str | None = None
    ft_tags: list[tuple[int, str]] = []

    for it in iters:
        if not isinstance(it, dict):
            continue
        k = it.get("k")
        if not isinstance(k, int):
            continue
        outputs = it.get("outputs")
        if not isinstance(outputs, dict):
            continue
        base = outputs.get("eval_iter_tag_base_advpos")
        if baseline_tag is None and isinstance(base, str) and base.strip():
            baseline_tag = base.strip()
        ft = outputs.get("eval_iter_tag_ft_advpos")
        if isinstance(ft, str) and ft.strip():
            ft_tags.append((k, ft.strip()))
        else:
            raise ValueError(
                f"Missing outputs.eval_iter_tag_ft_advpos for iteration k={k}"
            )

    if baseline_tag is None:
        raise ValueError(
            "Missing baseline eval tag: expected some outputs.eval_iter_tag_base_advpos"
        )

    ft_tags.sort(key=lambda x: x[0])
    return baseline_tag, ft_tags


def _extract_critic_tags(manifest: dict[str, Any]) -> list[tuple[int, str]]:
    iters = manifest.get("iterations")
    if not isinstance(iters, list):
        raise ValueError("manifest['iterations'] must be a list")

    out: list[tuple[int, str]] = []
    for it in iters:
        if not isinstance(it, dict):
            continue
        k = it.get("k")
        tag = it.get("critic_tag")
        if not isinstance(k, int):
            continue
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"Missing critic_tag for iteration k={k}")
        out.append((k, tag.strip()))
    out.sort(key=lambda x: x[0])
    return out


def _compute_success_rate(episodes_jsonl: Path) -> tuple[int, int, float]:
    eps = _iter_jsonl(episodes_jsonl)
    if not eps:
        raise ValueError(f"Empty episodes file: {episodes_jsonl}")
    n = 0
    n_success = 0
    for i, e in enumerate(eps, start=1):
        if "success_episode" not in e:
            raise ValueError(f"Missing 'success_episode' at {episodes_jsonl}:{i}")
        ok = _coerce_bool(
            e.get("success_episode"), context=f"success_episode at {episodes_jsonl}:{i}"
        )
        n += 1
        if ok:
            n_success += 1
    return n, n_success, (float(n_success) / float(n))


def _try_write_pngs(
    *,
    plots_dir: Path,
    success_points: list[tuple[str, float]],
    critic_points: list[tuple[str, float | None]],
) -> list[Path]:
    try:
        matplotlib = __import__("matplotlib")
        getattr(matplotlib, "use")("Agg")
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    except Exception:
        return []

    written: list[Path] = []
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        labels = [p[0] for p in success_points]
        ys = [p[1] for p in success_points]
        xs = list(range(len(labels)))
        fig = plt.figure(figsize=(8, 4.2), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xs, ys, marker="o")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("iteration")
        ax.set_ylabel("success_rate")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = plots_dir / "3A_success_rate_vs_iteration.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)
    except Exception:
        pass

    try:
        labels = [p[0] for p in critic_points]
        ys = [p[1] for p in critic_points]
        xs = list(range(len(labels)))
        fig = plt.figure(figsize=(8, 4.2), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xs, ys, marker="o")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("iteration")
        ax.set_ylabel("critic.final_val_loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = plots_dir / "3A_critic_convergence.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)
    except Exception:
        pass

    return written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize RECAP multi-iteration results: success_rate vs iteration and critic convergence."
        )
    )
    _ = parser.add_argument(
        "--run-id",
        required=True,
        help="Run id under agent/artifacts/p3A/<run_id>/manifest.json",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    run_id = str(args.run_id)

    manifest_path = repo_root / DEFAULT_P3A_DIR_REL / run_id / "manifest.json"
    recap_datasets_dir = repo_root / DEFAULT_RECAP_DATASETS_DIR_REL
    critics_dir = repo_root / DEFAULT_CRITICS_DIR_REL
    plots_dir = repo_root / DEFAULT_PLOTS_DIR_REL

    missing: list[Path] = []
    if not manifest_path.is_file():
        missing.append(manifest_path)

    if missing:
        print("[ERROR] missing required paths:")
        for p in missing:
            print(f"  - {p}")
        return 2

    try:
        manifest_any = _read_json(manifest_path)
        if not isinstance(manifest_any, dict):
            raise ValueError("manifest root must be a JSON object")
        manifest: dict[str, Any] = manifest_any
    except Exception as e:
        print(f"[ERROR] failed to read manifest: {manifest_path}")
        print(f"[ERROR] {e}")
        return 1

    try:
        baseline_eval_tag, ft_eval_tags = _extract_eval_tags(manifest)
        critic_tags = _extract_critic_tags(manifest)
    except Exception as e:
        print(f"[ERROR] manifest schema error: {e}")
        return 1

    success_inputs: list[tuple[str, Path]] = []
    success_inputs.append(
        ("baseline", recap_datasets_dir / baseline_eval_tag / "episodes.jsonl")
    )
    for k, tag in ft_eval_tags:
        success_inputs.append((f"k{k}", recap_datasets_dir / tag / "episodes.jsonl"))

    critic_inputs: list[tuple[int, str, Path]] = []
    for k, ctag in critic_tags:
        critic_inputs.append((k, ctag, critics_dir / ctag / "metrics.json"))

    for _, p in success_inputs:
        if not p.is_file():
            missing.append(p)
    for _, _, p in critic_inputs:
        if not p.is_file():
            missing.append(p)

    if missing:
        print("[ERROR] missing required paths:")
        for p in missing:
            print(f"  - {p}")
        return 2

    success_rows: list[list[str]] = []
    success_points: list[tuple[str, float]] = []
    try:
        for label, episodes_path in success_inputs:
            n, n_success, rate = _compute_success_rate(episodes_path)
            success_rows.append(
                [
                    label,
                    episodes_path.parent.name,
                    str(n),
                    str(n_success),
                    f"{rate:.4f}",
                ]
            )
            success_points.append((label, rate))
    except Exception as e:
        print(f"[ERROR] failed to compute success_rate: {e}")
        return 1

    critic_rows: list[list[str]] = []
    critic_points: list[tuple[str, float | None]] = []
    required_metric_keys = [
        "final_val_loss",
        "converged_epoch",
        "return_G_correlation",
        "n_samples_total",
    ]
    try:
        for k, ctag, metrics_path in critic_inputs:
            m_any = _read_json(metrics_path)
            if not isinstance(m_any, dict):
                raise ValueError(f"metrics root must be a JSON object: {metrics_path}")
            m: dict[str, Any] = m_any
            missing_keys = [kk for kk in required_metric_keys if kk not in m]
            if missing_keys:
                raise ValueError(
                    f"Missing keys in {metrics_path}: {', '.join(missing_keys)}"
                )

            final_val_loss = m.get("final_val_loss")
            converged_epoch = m.get("converged_epoch")
            return_G_correlation = m.get("return_G_correlation")
            n_samples_total = m.get("n_samples_total")

            critic_rows.append(
                [
                    f"k{k}",
                    ctag,
                    _fmt_num(final_val_loss),
                    _fmt_num(converged_epoch),
                    _fmt_num(return_G_correlation),
                    _fmt_num(n_samples_total),
                ]
            )
            critic_points.append(
                (
                    f"k{k}",
                    float(final_val_loss)
                    if isinstance(final_val_loss, (int, float))
                    else None,
                )
            )
    except Exception as e:
        print(f"[ERROR] failed to read critic metrics: {e}")
        return 1

    success_txt_path = plots_dir / "3A_success_rate_vs_iteration.txt"
    critic_txt_path = plots_dir / "3A_critic_convergence.txt"

    git_sha = (
        manifest.get("git", {}).get("sha")
        if isinstance(manifest.get("git"), dict)
        else None
    )
    run_id_in_manifest = manifest.get("run_id")

    success_table = _format_table(
        ["label", "eval_tag", "n_episodes", "n_success", "success_rate"],
        success_rows,
    )
    critic_table = _format_table(
        [
            "label",
            "critic_tag",
            "final_val_loss",
            "converged_epoch",
            "return_G_correlation",
            "n_samples_total",
        ],
        critic_rows,
    )

    success_text = (
        "\n".join(
            [
                "# 3A success_rate vs iteration",
                f"run_id={run_id}",
                f"run_id_in_manifest={run_id_in_manifest}",
                f"git_sha={git_sha}",
                f"manifest={manifest_path}",
                "",
            ]
        )
        + "\n"
        + success_table
        + "\n"
    )
    critic_text = (
        "\n".join(
            [
                "# 3A critic convergence",
                f"run_id={run_id}",
                f"run_id_in_manifest={run_id_in_manifest}",
                f"git_sha={git_sha}",
                f"manifest={manifest_path}",
                "",
            ]
        )
        + "\n"
        + critic_table
        + "\n"
    )

    try:
        _atomic_write_text(success_txt_path, success_text)
        _atomic_write_text(critic_txt_path, critic_text)
    except Exception as e:
        print(f"[ERROR] failed to write outputs: {e}")
        return 1

    _ = _try_write_pngs(
        plots_dir=plots_dir,
        success_points=success_points,
        critic_points=critic_points,
    )

    print(f"[EVIDENCE] {success_txt_path}")
    print(f"[EVIDENCE] {critic_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
