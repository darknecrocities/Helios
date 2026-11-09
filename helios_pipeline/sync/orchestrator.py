"""Post-training sync to Hugging Face Hub, Ollama, and GitHub.

Each target fails independently and reports its status; nothing is retried silently.
Weights are uploaded to HF only when `upload_weights=True`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from helios_pipeline.git_sync.sync import GitSyncEngine

HF_REPO = "Darknecrocities/Helios"


@dataclass
class SyncReport:
    status: dict[str, str] = field(default_factory=dict)


def sync_hf(checkpoints: dict[str, str], report_md: Path, upload_weights: bool) -> str:
    try:
        from huggingface_hub import HfApi

        from scripts.deploy_huggingface import resolve_hf_token

        token = resolve_hf_token()
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(report_md),
            path_in_repo="reports/FIVE_MINUTE_PRETRAIN_REPORT.md",
            repo_id=HF_REPO,
            commit_message="docs: time-boxed pretraining report",
        )
        for tier, ckpt in checkpoints.items():
            files = ["manifest.json", "config.json"] + (
                ["model.safetensors"] if upload_weights else []
            )
            for name in files:
                api.upload_file(
                    path_or_fileobj=str(Path(ckpt) / name),
                    path_in_repo=f"continuous/{tier}/{name}",
                    repo_id=HF_REPO,
                    commit_message=f"chore: sync {tier} continuous checkpoint {name}",
                )
        return "ok"
    except Exception as exc:  # report, don't swallow
        return f"failed: {type(exc).__name__}: {exc}"


def sync_ollama(push: bool) -> str:
    if shutil.which("ollama") is None:
        return "skipped: ollama CLI not installed"
    try:
        from scripts.deploy_family_ollama import deploy_family

        tags = deploy_family(push=push)
        return f"ok ({len(tags)} tags, push={push})"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"


def sync_git(files: list[Path], message: str) -> str:
    try:
        ok = GitSyncEngine().sync_files(files, message, push=True)
        if not ok:
            return "failed: commit or push returned error"
        pr = subprocess.run(
            ["gh", "pr", "comment", "13", "--body", f"Automated sync: {message}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "ok" if pr.returncode == 0 else f"pushed; PR comment failed: {pr.stderr.strip()}"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"


def run_sync(
    targets: str = "all",
    checkpoints: dict[str, str] | None = None,
    report_md: Path | None = None,
    upload_weights: bool = False,
) -> SyncReport:
    """targets: 'all' or comma list of hf,ollama,git."""
    want = {"hf", "ollama", "git"} if targets == "all" else set(targets.split(","))
    rep = SyncReport()
    ckpts = checkpoints or {}
    report = report_md or Path("docs/FIVE_MINUTE_PRETRAIN_REPORT.md")

    if "hf" in want and report.exists():
        rep.status["hf"] = sync_hf(ckpts, report, upload_weights)
    elif "hf" in want:
        rep.status["hf"] = f"skipped: report {report} does not exist"

    if "ollama" in want:
        rep.status["ollama"] = sync_ollama(push=True)

    if "git" in want:
        files = [p for p in [report, Path("eval/reports/pretrain_timeboxed.json")] if p.exists()]
        if files:
            rep.status["git"] = sync_git(
                files, "chore(pipeline): record time-boxed pretraining run"
            )
        else:
            rep.status["git"] = "skipped: no generated reports to commit"

    out_path = Path("eval/reports/last_sync.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep.status, indent=2))
    return rep
