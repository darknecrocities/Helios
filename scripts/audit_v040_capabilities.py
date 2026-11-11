#!/usr/bin/env python3
"""Reproduce the Helios v0.4.0 capability audit.

The audit is intentionally conservative: it reports what is *verifiable* from
the local checkpoints, the Ollama daemon, and the Hugging Face Hub. It does not
upgrade a target into a capability.

Sections
--------
1. Host hardware and free disk.
2. Local checkpoint integrity (Core / Pro / Flagship and v0.4.0 lineage).
3. Ollama advertised capabilities and optional live probes.
4. Hugging Face publication state (branch steps and file sizes).
5. Consolidated capability matrix.

Usage
-----
    python scripts/audit_v040_capabilities.py
    python scripts/audit_v040_capabilities.py --json-out reports/audit.json
    python scripts/audit_v040_capabilities.py --live
    python scripts/audit_v040_capabilities.py --fail-on-integrity

Exit codes
----------
0  Audit completed (findings may still be present).
2  Integrity failure and ``--fail-on-integrity`` was supplied.
3  Audit could not run (missing drive, invalid arguments).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DRIVE_ROOT = Path("/Volumes/One Touch/Helios")
DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_HF_REPO = "Darknecrocities/Helios"
TIERS = ("core", "pro", "flagship")

# A 1x1 red PNG. Used only to check whether a runtime accepts image input.
RED_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAF"
    "gQJ/5J8bWQAAAABJRU5ErkJggg=="
)


@dataclass
class HostAudit:
    """Host identity and memory available to the training/serving process."""

    machine: str = ""
    platform: str = ""
    processor: str = ""
    cpu_count: int = 0
    memory_gib: float = 0.0
    internal_free_gib: float = 0.0
    drive_present: bool = False
    drive_free_gib: float = 0.0
    torch: str = ""
    mps: bool = False
    cuda: bool = False


@dataclass
class CheckpointAudit:
    """Integrity and shape of one tier's local ``latest`` checkpoint."""

    tier: str
    path: str
    exists: bool = False
    mode: str = "unknown"
    declared_step: int = 0
    effective_step: int = 0
    config_max_seq_len: int = 0
    trained_seq_len: int = 0
    tensor_count: int = 0
    parameter_count: int = 0
    keys_prefix: str = ""
    integrity: str = "missing"
    actual_sha256: str = ""
    expected_sha256: str = ""
    model_matches_step: str = ""
    state_matches_step: str = ""
    consistent_step: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class LineageAudit:
    """State of the conflict-free v0.4.0 continuation lineage."""

    path: str
    present: bool = False
    tiers: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class OllamaTierAudit:
    """What Ollama advertises, and optionally what it actually does."""

    tier: str
    tag: str = ""
    found: bool = False
    family: str = ""
    parameter_size: str = ""
    parameter_count: int = 0
    context_length: int = 0
    capabilities: list[str] = field(default_factory=list)
    template_has_tools: bool = False
    live_tool_call: str = "not_run"
    live_tool_arguments: dict[str, Any] = field(default_factory=dict)
    live_vision: str = "not_run"
    live_vision_response: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class HFTierAudit:
    """Publication state for one tier on the Hub."""

    tier: str
    branch: str = ""
    branch_config_present: bool = False
    branch_model_bytes: int = 0
    continuous_steps: int = 0
    drive_steps: int = 0
    stale: bool = False
    continuous_model_bytes: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class HFAudit:
    """Hub-level metadata and per-tier publication state."""

    repo: str = DEFAULT_HF_REPO
    reachable: bool = False
    sha: str = ""
    last_modified: str = ""
    branches: list[str] = field(default_factory=list)
    tiers: dict[str, HFTierAudit] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Complete audit report serialized by ``--json-out``."""

    generated_at: str
    host: HostAudit
    checkpoints: dict[str, CheckpointAudit]
    lineage: LineageAudit
    ollama: dict[str, OllamaTierAudit]
    huggingface: HFAudit
    matrix: dict[str, dict[str, str]]
    findings: list[str]


def human_gib(value: int | float) -> float:
    """Convert bytes to GiB, rounded for display."""
    return round(float(value) / (1024**3), 2)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty dict when unavailable or invalid."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
    token: str | None = None,
) -> dict[str, Any]:
    """Fetch JSON over HTTP, raising a readable error on failure."""
    data = None
    headers = {"User-Agent": "helios-v040-audit/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return parsed


def request_json_list(
    url: str,
    timeout: float = 30.0,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a JSON list over HTTP."""
    headers = {"User-Agent": "helios-v040-audit/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list from {url}")
    return [item for item in parsed if isinstance(item, dict)]


def safetensors_metadata(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    """Read tensor names and shapes from a safetensors header without loading weights."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    return keys, shapes


def count_parameters(shapes: dict[str, list[int]]) -> int:
    """Sum tensor element counts from shapes."""
    total = 0
    for shape in shapes.values():
        count = 1
        for dim in shape:
            count *= int(dim)
        total += count
    return total


def trained_sequence_length(manifest: dict[str, Any]) -> int:
    """Extract the actual training sequence length from a manifest."""
    match = re.search(r"seq=(\d+)", str(manifest.get("hardware_profile", "")))
    if match:
        return int(match.group(1))
    telemetry = manifest.get("training_telemetry")
    if isinstance(telemetry, dict):
        return int(telemetry.get("seq_len", 0) or 0)
    return int(manifest.get("seq_len", 0) or 0)


def discover_consistent_step(
    tier_root: Path,
    latest_model_sha: str,
    latest_state_sha: str,
    expected_latest_sha: str,
    max_candidates: int = 10,
) -> tuple[str, str, str]:
    """Locate which step directories supplied a mixed ``latest`` folder.

    Returns ``(model_matches_step, state_matches_step, consistent_step)``.
    Only the newest ``max_candidates`` step directories are inspected to keep
    the audit bounded.
    """
    candidates = sorted(
        (path for path in tier_root.glob("step_*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )[:max_candidates]
    model_matches = ""
    state_matches = ""
    consistent = ""
    for step_dir in candidates:
        model_path = step_dir / "model.safetensors"
        state_path = step_dir / "state.pt"
        manifest = read_json(step_dir / "manifest.json")
        if model_path.is_file():
            step_model_sha = sha256_file(model_path)
            if step_model_sha == latest_model_sha:
                model_matches = step_dir.name
            if step_model_sha == str(manifest.get("sha256_digest", "")) and not consistent:
                consistent = step_dir.name
        if state_path.is_file() and (
            step_dir.name == model_matches
            or str(manifest.get("sha256_digest", "")) == expected_latest_sha
        ):
            if sha256_file(state_path) == latest_state_sha:
                state_matches = step_dir.name
    return model_matches, state_matches, consistent


def audit_checkpoint(tier: str, tier_root: Path) -> CheckpointAudit:
    """Audit one tier's ``latest`` checkpoint and detect mixed copies."""
    latest = tier_root / "latest"
    result = CheckpointAudit(tier=tier, path=str(latest))
    if not latest.is_dir():
        result.notes.append("latest directory is missing")
        return result

    manifest = read_json(latest / "manifest.json")
    config = read_json(latest / "config.json")
    model_path = latest / "model.safetensors"
    state_path = latest / "state.pt"
    if not model_path.is_file() or not state_path.is_file():
        result.notes.append("model.safetensors or state.pt is missing")
        return result

    result.exists = True
    result.declared_step = int(manifest.get("total_steps", 0) or 0)
    result.effective_step = result.declared_step
    result.config_max_seq_len = int(config.get("max_seq_len", 0) or 0)
    result.trained_seq_len = trained_sequence_length(manifest)
    result.mode = str(manifest.get("mode", "")) or "unknown"
    result.expected_sha256 = str(manifest.get("sha256_digest", ""))
    result.actual_sha256 = sha256_file(model_path)
    result.integrity = (
        "ok" if result.expected_sha256 and result.actual_sha256 == result.expected_sha256 else "mismatch"
    )

    try:
        keys, shapes = safetensors_metadata(model_path)
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        result.notes.append(f"safetensors header unreadable: {exc}")
        return result

    result.tensor_count = len(keys)
    result.parameter_count = count_parameters(shapes)
    if any(key.startswith("block.") for key in keys):
        result.keys_prefix = "block."
    elif any(key.startswith("layers.") for key in keys):
        result.keys_prefix = "layers."
    else:
        result.keys_prefix = keys[0] if keys else ""
    if result.mode == "unknown":
        result.mode = "block_slice" if result.keys_prefix == "block." else "full"

    if result.integrity != "ok":
        latest_state_sha = sha256_file(state_path)
        model_step, state_step, consistent_step = discover_consistent_step(
            tier_root,
            result.actual_sha256,
            latest_state_sha,
            result.expected_sha256,
        )
        result.model_matches_step = model_step
        result.state_matches_step = state_step
        result.consistent_step = consistent_step
        if model_step and state_step and model_step != state_step:
            result.notes.append(
                f"mixed latest: model.safetensors is {model_step}, state.pt is {state_step}"
            )
        elif model_step:
            result.notes.append(f"model.safetensors matches {model_step}")
        if consistent_step:
            result.notes.append(f"{consistent_step} is internally consistent")
            match = re.search(r"step_(\d+)", consistent_step)
            if match:
                result.effective_step = int(match.group(1))
    else:
        result.consistent_step = "latest"

    if result.keys_prefix == "block.":
        result.notes.append(
            "single-block slice; full-tier export requires replication or full retraining"
        )
    return result


def audit_lineage(lineage_root: Path) -> LineageAudit:
    """Inspect the v0.4.0 continuation lineage without requiring it to be complete."""
    result = LineageAudit(path=str(lineage_root), present=lineage_root.is_dir())
    if not result.present:
        result.notes.append("lineage has not been written yet")
        return result

    lineage_file = read_json(lineage_root / "lineage.json")
    if lineage_file:
        result.notes.append(f"lineage.json declares {lineage_file.get('lineage', 'unknown')}")
    for tier in TIERS:
        latest = lineage_root / f"{tier}_continuous" / "latest"
        manifest = read_json(latest / "manifest.json")
        if not manifest:
            result.notes.append(f"{tier}: latest manifest missing or incomplete")
            continue
        result.tiers[tier] = {
            "path": str(latest),
            "total_steps": manifest.get("total_steps", 0),
            "parent_step": manifest.get("parent_step", 0),
            "parent_sha256": str(manifest.get("parent_sha256", ""))[:16],
            "steps": manifest.get("steps", 0),
            "tokens": manifest.get("tokens", 0),
            "seconds": manifest.get("seconds", 0),
            "heldout_loss_before": manifest.get("heldout_loss_before"),
            "heldout_loss_after": manifest.get("heldout_loss_after"),
            "heldout_improvement_pct": manifest.get("heldout_improvement_pct"),
            "target_context_length": manifest.get("target_context_length"),
            "context_extension_status": manifest.get("context_extension_status"),
            "integrity": "declared",
        }
    return result


def audit_ollama(base_url: str, live: bool) -> dict[str, OllamaTierAudit]:
    """Query Ollama for tags, capabilities, and optional live behavior."""
    audits: dict[str, OllamaTierAudit] = {}
    try:
        tags = request_json(f"{base_url}/api/tags")
    except Exception as exc:
        for tier in TIERS:
            audits[tier] = OllamaTierAudit(tier=tier, notes=[f"Ollama unreachable: {exc}"])
        return audits

    available = {
        str(item.get("name", "")) for item in tags.get("models", []) if isinstance(item, dict)
    }
    for tier in TIERS:
        result = OllamaTierAudit(tier=tier)
        candidates = [f"parejasarronkian/Helios:{tier}", f"helios:{tier}"]
        tag = next((candidate for candidate in candidates if candidate in available), "")
        if not tag:
            result.notes.append("tag not found")
            audits[tier] = result
            continue
        result.tag = tag
        result.found = True
        try:
            show = request_json(f"{base_url}/api/show", {"model": tag})
        except Exception as exc:
            result.notes.append(f"show failed: {exc}")
            audits[tier] = result
            continue

        details = show.get("details") if isinstance(show.get("details"), dict) else {}
        result.family = str(details.get("family", ""))
        result.parameter_size = str(details.get("parameter_size", ""))
        model_info = show.get("model_info") if isinstance(show.get("model_info"), dict) else {}
        result.parameter_count = int(model_info.get("general.parameter_count", 0) or 0)
        context_values = [
            int(value)
            for key, value in model_info.items()
            if key.endswith(".context_length") and isinstance(value, (int, float))
        ]
        result.context_length = max(context_values, default=0)
        result.capabilities = [str(item) for item in show.get("capabilities", [])]
        template = str(show.get("template", ""))
        result.template_has_tools = ".Tools" in template or "tools" in template.lower()
        if result.family in {"gemma2", "gemma3", "llama", "qwen2"}:
            result.notes.append(
                f"served through the {result.family} base model; not Helios-native weights"
            )
        if "tools" in result.capabilities and not live:
            result.notes.append("tools capability is advertised; live call not probed")
        if "vision" in result.capabilities and not live:
            result.notes.append("vision capability is advertised; live probe not run")

        if live and "tools" in result.capabilities:
            probe = {
                "model": tag,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "Call get_weather for Manila. Use the tool, do not explain.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            }
            try:
                response = request_json(f"{base_url}/api/chat", probe, timeout=180)
                message = response.get("message", {})
                calls = message.get("tool_calls") if isinstance(message, dict) else None
                if calls:
                    result.live_tool_call = "verified"
                    first = calls[0] if isinstance(calls, list) else {}
                    function = first.get("function", {}) if isinstance(first, dict) else {}
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    result.live_tool_arguments = arguments if isinstance(arguments, dict) else {}
                else:
                    result.live_tool_call = "refused_or_text_only"
            except Exception as exc:
                result.live_tool_call = f"error: {exc}"

        if live and "vision" in result.capabilities:
            probe = {
                "model": tag,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "What color is this image? Answer with one word.",
                        "images": [base64.b64encode(RED_PIXEL_PNG).decode("ascii")],
                    }
                ],
            }
            try:
                response = request_json(f"{base_url}/api/chat", probe, timeout=180)
                message = response.get("message", {})
                content = str(message.get("content", "")) if isinstance(message, dict) else ""
                result.live_vision = "accepted" if content.strip() else "empty_response"
                result.live_vision_response = content.strip()[:120]
            except Exception as exc:
                result.live_vision = f"error: {exc}"

        audits[tier] = result
    return audits


def hf_tree(repo: str, revision: str, token: str | None) -> dict[str, int]:
    """Return a mapping of repo path to byte size for one revision."""
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    try:
        items = request_json_list(url, token=token)
    except Exception:
        return {}
    sizes: dict[str, int] = {}
    for item in items:
        path = item.get("path")
        if not isinstance(path, str):
            continue
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        size = lfs.get("size", item.get("size", 0))
        try:
            sizes[path] = int(size or 0)
        except (TypeError, ValueError):
            sizes[path] = 0
    return sizes


def audit_huggingface(
    repo: str,
    checkpoints: dict[str, CheckpointAudit],
    token: str | None,
) -> HFAudit:
    """Compare Hub branch metadata with the local drive checkpoints."""
    result = HFAudit(repo=repo)
    try:
        info = request_json(f"https://huggingface.co/api/models/{repo}", token=token)
        result.reachable = True
        result.sha = str(info.get("sha", ""))
        result.last_modified = str(info.get("lastModified", ""))
    except Exception as exc:
        result.notes.append(f"Hub metadata unavailable: {exc}")
        return result

    try:
        refs = request_json(f"https://huggingface.co/api/models/{repo}/refs", token=token)
        branches = refs.get("branches", [])
        if isinstance(branches, list):
            result.branches = [str(item.get("name", "")) for item in branches if isinstance(item, dict)]
    except Exception as exc:
        result.notes.append(f"Hub refs unavailable: {exc}")

    main_tree = hf_tree(repo, "main", token)

    for tier in TIERS:
        tier_result = HFTierAudit(tier=tier, branch=tier if tier in result.branches else "")
        if tier_result.branch:
            branch_tree = hf_tree(repo, tier_result.branch, token)
            tier_result.branch_model_bytes = branch_tree.get("model.safetensors", 0)
            tier_result.branch_config_present = "config.json" in branch_tree
        try:
            manifest = request_json(
                f"https://huggingface.co/{repo}/resolve/main/continuous/{tier}/manifest.json",
                token=token,
            )
            tier_result.continuous_steps = int(manifest.get("total_steps", 0) or 0)
        except Exception as exc:
            tier_result.notes.append(f"continuous manifest unavailable: {exc}")

        local = checkpoints.get(tier)
        tier_result.drive_steps = local.effective_step if local else 0
        tier_result.stale = bool(
            tier_result.continuous_steps and tier_result.drive_steps
            and tier_result.continuous_steps < tier_result.drive_steps
        )
        tier_result.continuous_model_bytes = main_tree.get(
            f"continuous/{tier}/model.safetensors", 0
        )
        if tier_result.stale:
            tier_result.notes.append(
                f"Hub step {tier_result.continuous_steps} is behind drive step "
                f"{tier_result.drive_steps}"
            )
        if tier_result.branch_model_bytes:
            tier_result.notes.append(
                f"branch model.safetensors is {human_gib(tier_result.branch_model_bytes):.2f} GiB"
            )
        result.tiers[tier] = tier_result
    return result


def build_matrix(
    checkpoints: dict[str, CheckpointAudit],
    ollama: dict[str, OllamaTierAudit],
    lineage: LineageAudit,
    mcp_available: bool,
) -> dict[str, dict[str, str]]:
    """Build the conservative capability matrix shown to the user."""
    matrix: dict[str, dict[str, str]] = {}
    for tier in TIERS:
        checkpoint = checkpoints[tier]
        serving = ollama[tier]
        caps = set(serving.capabilities)
        trained_seq = checkpoint.trained_seq_len or checkpoint.config_max_seq_len
        header_ctx = serving.context_length or checkpoint.config_max_seq_len
        if checkpoint.mode == "block_slice":
            weights = "block_slice_only"
        elif checkpoint.tensor_count > 20:
            weights = "full"
        else:
            weights = "unknown"
        if lineage.tiers.get(tier):
            weights = f"{weights}+continuation_demo"
        base_only = checkpoint.mode == "block_slice"
        matrix[tier] = {
            "text": "available" if serving.found else "unavailable",
            "native_weights": weights,
            "coding": "target_only" if lineage.tiers.get(tier) else "unverified",
            "tools": (
                "runtime_verified_base_model"
                if serving.live_tool_call == "verified" and base_only
                else "verified"
                if serving.live_tool_call == "verified"
                else "advertised_only"
                if "tools" in caps
                else "unavailable"
            ),
            "mcp": "client_available" if mcp_available else "runtime_missing",
            "vision": (
                "runtime_verified_base_model"
                if serving.live_vision == "accepted" and base_only
                else "accepted_by_runtime"
                if serving.live_vision == "accepted"
                else "advertised_only"
                if "vision" in caps
                else "unavailable"
            ),
            "long_context": (
                f"trained_{trained_seq}_header_{header_ctx}"
                if trained_seq and header_ctx
                else "unverified"
            ),
        }
    return matrix


def audit_host() -> HostAudit:
    """Collect host, memory, and disk facts used in the feasibility analysis."""
    result = HostAudit(
        machine=platform.machine(),
        platform=platform.platform(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 0,
    )
    if platform.system() == "Darwin":
        try:
            result.machine = (
                subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
                or result.machine
            )
            result.memory_gib = human_gib(
                int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            )
        except (subprocess.SubprocessError, ValueError):
            pass
    if not result.memory_gib:
        try:
            result.memory_gib = human_gib(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass

    try:
        result.internal_free_gib = human_gib(shutil.disk_usage(Path.home()).free)
    except OSError:
        pass
    result.drive_present = DRIVE_ROOT.is_dir()
    if result.drive_present:
        try:
            result.drive_free_gib = human_gib(shutil.disk_usage(DRIVE_ROOT).free)
        except OSError:
            pass

    try:
        import torch

        result.torch = torch.__version__
        result.mps = bool(torch.backends.mps.is_available())
        result.cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    return result


def print_report(report: AuditReport) -> None:
    """Render the audit in a compact, human-readable form."""
    host = report.host
    print("Helios v0.4.0 capability audit")
    print("=" * 78)
    print(
        f"Host: {host.machine} | {host.platform} | {host.cpu_count} cores | "
        f"{host.memory_gib:.1f} GiB RAM"
    )
    print(
        f"Disk: internal {host.internal_free_gib:.1f} GiB free | One Touch "
        f"{'present' if host.drive_present else 'missing'} "
        f"({host.drive_free_gib:.1f} GiB free)"
    )
    print(f"Runtime: torch {host.torch or 'unavailable'} | mps={host.mps} | cuda={host.cuda}")

    print("\nCheckpoints (local drive)")
    for tier in TIERS:
        item = report.checkpoints[tier]
        if not item.exists:
            print(f"  {tier:<9} MISSING {item.path}")
            continue
        step_label = f"{item.declared_step}"
        if item.effective_step and item.effective_step != item.declared_step:
            step_label = f"{item.declared_step}->{item.effective_step}"
        print(
            f"  {tier:<9} step={step_label:<13} {item.mode:<11} "
            f"tensors={item.tensor_count:<3} params={item.parameter_count / 1e6:7.2f}M "
            f"trained_seq={item.trained_seq_len or '?':<7} integrity={item.integrity}"
        )
        for note in item.notes:
            print(f"            note: {note}")

    lineage = report.lineage
    print(f"\nv0.4.0 lineage: {'present' if lineage.present else 'not written yet'} ({lineage.path})")
    for tier, item in lineage.tiers.items():
        print(
            f"  {tier:<9} step={item.get('total_steps', 0):<7} "
            f"parent={item.get('parent_step', 0):<7} "
            f"heldout {item.get('heldout_loss_before')} -> {item.get('heldout_loss_after')} "
            f"({item.get('heldout_improvement_pct')}%)"
        )

    print("\nOllama")
    for tier in TIERS:
        item = report.ollama[tier]
        if not item.found:
            print(f"  {tier:<9} not found")
            continue
        print(
            f"  {item.tag:<34} family={item.family:<8} params={item.parameter_size:<7} "
            f"ctx={item.context_length:<8} caps={','.join(item.capabilities) or 'none'}"
        )
        if item.live_tool_call != "not_run":
            print(f"            live tool: {item.live_tool_call} {item.live_tool_arguments}")
        if item.live_vision != "not_run":
            print(f"            live vision: {item.live_vision} {item.live_vision_response}")
        for note in item.notes:
            print(f"            note: {note}")

    print("\nHugging Face")
    hf = report.huggingface
    print(f"  repo={hf.repo} reachable={hf.reachable} sha={hf.sha[:12]} modified={hf.last_modified}")
    for tier in TIERS:
        item = hf.tiers.get(tier)
        if not item:
            continue
        state = "STALE" if item.stale else "current"
        print(
            f"  {tier:<9} branch={item.branch or 'missing':<8} "
            f"continuous={item.continuous_steps:<7} drive={item.drive_steps:<7} {state}"
        )
    for note in hf.notes:
        print(f"  note: {note}")

    print("\nCapability matrix (verified, not aspirational)")
    columns = ("text", "native_weights", "coding", "tools", "mcp", "vision", "long_context")
    print("  tier      " + " ".join(f"{column:<22}" for column in columns))
    for tier in TIERS:
        row = report.matrix[tier]
        print(f"  {tier:<9} " + " ".join(f"{row[column]:<22}" for column in columns))

    print("\nFindings")
    if report.findings:
        for finding in report.findings:
            print(f"  - {finding}")
    else:
        print("  - none")


def build_findings(report: AuditReport) -> list[str]:
    """Derive the short list of findings that must not be lost in the JSON."""
    findings: list[str] = []
    for tier in TIERS:
        checkpoint = report.checkpoints[tier]
        if checkpoint.integrity == "mismatch":
            findings.append(
                f"{tier}: latest checkpoint is a mixed copy "
                f"(model={checkpoint.model_matches_step or 'unknown'}, "
                f"state={checkpoint.state_matches_step or 'unknown'}); "
                f"use {checkpoint.consistent_step or 'a verified step'} as parent"
            )
        elif checkpoint.mode == "block_slice":
            findings.append(
                f"{tier}: only a single-block slice is present; this is not a full "
                f"{tier} model and cannot be exported as one without replication/retraining"
            )
        if checkpoint.trained_seq_len and checkpoint.trained_seq_len < 1024:
            findings.append(
                f"{tier}: trained at {checkpoint.trained_seq_len} tokens; any larger "
                f"context advertised by Ollama is a header value, not a trained capability"
            )
    for tier in TIERS:
        serving = report.ollama[tier]
        if "tools" in serving.capabilities and serving.live_tool_call != "verified":
            findings.append(
                f"{tier}: Ollama advertises tools through its template, but no verified "
                f"Helios-native tool call exists"
            )
        elif "tools" in serving.capabilities:
            findings.append(
                f"{tier}: tool call succeeds at runtime through the base model, not through "
                f"trained Helios weights"
            )
        if "vision" in serving.capabilities:
            findings.append(
                f"{tier}: Ollama exposes vision through the base model, not through a "
                f"trained Helios vision adapter"
            )
    for tier in TIERS:
        hf = report.huggingface.tiers.get(tier)
        if hf and hf.stale:
            findings.append(
                f"{tier}: Hub continuous checkpoint is stale "
                f"({hf.continuous_steps} vs drive {hf.drive_steps})"
            )
    if not report.huggingface.reachable:
        findings.append("Hugging Face Hub could not be audited; publication is unverified")
    if report.matrix["core"]["mcp"] == "runtime_missing":
        findings.append("MCP is not installed as a runtime client; tool metadata is not MCP")
    return findings


def main() -> int:
    """Parse arguments, run every audit section, and print/serialize the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=DRIVE_ROOT)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--live", action="store_true", help="Run tool/vision probes against Ollama")
    parser.add_argument(
        "--fail-on-integrity",
        action="store_true",
        help="Exit 2 when any checkpoint fails its published SHA-256",
    )
    args = parser.parse_args()

    if not args.drive_root.is_dir():
        print(f"Drive root not found: {args.drive_root}", file=sys.stderr)
        return 3

    checkpoints = {
        tier: audit_checkpoint(tier, args.drive_root / "checkpoints" / f"{tier}_continuous")
        for tier in TIERS
    }
    lineage = audit_lineage(args.drive_root / "checkpoints" / "v040_lineage")
    ollama = audit_ollama(args.ollama_url, args.live)
    hf_token = os.environ.get("HF_TOKEN")
    huggingface = audit_huggingface(args.hf_repo, checkpoints, hf_token)

    try:
        import importlib.util

        mcp_available = importlib.util.find_spec("mcp") is not None
    except (ImportError, ValueError):
        mcp_available = False

    report = AuditReport(
        generated_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        host=audit_host(),
        checkpoints=checkpoints,
        lineage=lineage,
        ollama=ollama,
        huggingface=huggingface,
        matrix=build_matrix(checkpoints, ollama, lineage, mcp_available),
        findings=[],
    )
    report.findings = build_findings(report)

    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json_out}")

    if args.fail_on_integrity and any(
        item.integrity == "mismatch" for item in checkpoints.values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
