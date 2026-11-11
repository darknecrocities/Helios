"""Resume-safe v0.4.0 coding/agentic continuation for Helios Core, Pro, and Flagship.

This runner never writes into the existing ``<tier>_continuous`` lineage. It:

1. Selects a verified parent checkpoint (Pro uses step_00005145 because its
   ``latest`` folder is a mixed copy; Core/Flagship use the verified ``latest``).
2. Verifies the parent safetensors SHA-256 against the parent manifest.
3. Loads the block-slice state into ``ProgressiveTierProjectionHead``.
4. Continues training on coding + agentic/MCP records from the One Touch corpus.
5. Saves an atomic ``v040_<tier>_continuous`` checkpoint with parent provenance.

The 1M context target is recorded as a **target** in the manifest. It is not
trained here: the parent was trained at 96 tokens and this machine has 16 GB
unified memory. A 1M-token dense-attention KV cache alone needs ~66-74 GB for
Pro/Flagship (FP16). See docs/research/V040_1M_CAPABILITY_PLAN.md.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from helios.cli.utils import load_or_create_tokenizer
from helios.model.config import HeliosConfig
from scripts.pretrain_all_models import ProgressiveTierProjectionHead
from scripts.pretrain_helios import prepare_token_batches

ONETOUCH = Path("/Volumes/One Touch/Helios")
CORPUS = ONETOUCH / "datasets" / "helios_massive_pretrain_corpus.jsonl"
LINEAGE_ROOT = ONETOUCH / "checkpoints" / "v040_lineage"

CODING_DOMAINS = {
    "coding_and_algorithms",
    "software_engineering_and_architecture",
    "tech_history_places_coding_documentation",
    "technology_and_systems",
}
AGENT_DOMAINS = {"agentic_skills_and_mcp"}

# Pro latest was partially overwritten (state from 5053, safetensors from 5145).
# step_00005145 is the last internally consistent Pro checkpoint.
TIER_SOURCES = {
    "core": "latest",
    "pro": "step_00005145",
    "flagship": "latest",
}

TARGET_CONTEXT = 1_000_000
TARGET_CAPABILITIES = ["coding", "tools", "mcp", "vision", "long_context"]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(tier: str) -> Path:
    """Resolve the verified source checkpoint for a tier."""
    source = ONETOUCH / "checkpoints" / f"{tier}_continuous" / TIER_SOURCES[tier]
    if not (source / "state.pt").is_file():
        raise FileNotFoundError(f"Missing state.pt in {source}")
    return source


def verify_source(source: Path) -> dict[str, Any]:
    """Verify the parent safetensors digest against the parent manifest."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("sha256_digest")
    actual = sha256_file(source / "model.safetensors")
    if expected and expected != actual:
        raise RuntimeError(
            f"Parent checkpoint integrity failure: {source} expected {expected}, got {actual}"
        )
    payload = torch.load(source / "state.pt", map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Empty or invalid state.pt in {source}")
    if any(torch.isnan(t).any() or torch.isinf(t).any() for t in state.values()):
        raise RuntimeError(f"NaN/Inf parameters in parent checkpoint {source}")
    return {
        "manifest": manifest,
        "sha256": actual,
        "step": int(payload.get("step", manifest.get("total_steps", 0))),
        "state": state,
        "config": HeliosConfig.from_dict(json.loads((source / "config.json").read_text())),
    }


def select_records(limit: int, eval_limit: int) -> tuple[list[dict], list[dict], dict]:
    """Select a deterministic coding + agentic/MCP training and held-out split."""
    coding, agents = [], []
    with open(CORPUS, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            domain = str(record.get("domain", ""))
            if domain in CODING_DOMAINS:
                coding.append(record)
            elif domain in AGENT_DOMAINS:
                agents.append(record)

    # 65% coding / 35% agentic, deterministic ordering before slicing.
    coding = coding[:: max(1, len(coding) // max(1, limit))]
    agents = agents[:: max(1, len(agents) // max(1, limit))]
    coding_train = coding[: int(limit * 0.65)]
    agent_train = agents[: limit - len(coding_train)]
    eval_coding = coding[-eval_limit:]
    eval_agents = agents[-max(1, eval_limit // 2) :]
    train = coding_train + agent_train
    heldout = eval_coding + eval_agents
    stats = {
        "coding_pool": len(coding),
        "agent_pool": len(agents),
        "train_records": len(train),
        "heldout_records": len(heldout),
        "coding_train": len(coding_train),
        "agent_train": len(agent_train),
    }
    return train, heldout, stats


def mean_loss(module: ProgressiveTierProjectionHead, batches: list[dict]) -> float:
    """Compute mean held-out loss without gradient updates."""
    module.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            losses.append(float(module(batch["input_ids"], labels=batch["labels"])["loss"].item()))
    module.train()
    valid = [value for value in losses if math.isfinite(value)]
    return round(sum(valid) / len(valid), 6) if valid else float("nan")


def write_checkpoint(
    tier: str,
    module: ProgressiveTierProjectionHead,
    config: HeliosConfig,
    tokenizer: Any,
    parent: dict[str, Any],
    telemetry: dict[str, Any],
    total_steps: int,
) -> Path:
    """Atomically write a new lineage checkpoint and return its step directory."""
    tier_root = LINEAGE_ROOT / f"{tier}_continuous"
    final_dir = tier_root / f"step_{total_steps:08d}"
    staging = LINEAGE_ROOT / "staging" / f"{tier}_{total_steps}_{os.getpid()}"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing lineage checkpoint: {final_dir}")
    staging.mkdir(parents=True, exist_ok=True)

    state = {k: v.detach().cpu().contiguous().clone() for k, v in module.state_dict().items()}
    for tensor in state.values():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise RuntimeError(f"NaN/Inf detected before saving {tier}")

    torch.save(
        {
            "model_state_dict": state,
            "step": total_steps,
            "parent_step": parent["step"],
            "parent_sha256": parent["sha256"],
            "lineage": f"v0.4.0-{tier}-coding-agents",
            "training_telemetry": telemetry,
            "config": config.to_dict(),
        },
        staging / "state.pt",
    )
    save_file(state, str(staging / "model.safetensors"))
    (staging / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    tokenizer.save_pretrained(staging)

    model_sha = sha256_file(staging / "model.safetensors")
    manifest = {
        "tier": tier,
        "lineage": f"v0.4.0-{tier}-coding-agents",
        "mode": "resumed_block_slice_coding_agents",
        "device": "cpu",
        "parent_checkpoint": str(parent["path"]),
        "parent_step": parent["step"],
        "parent_sha256": parent["sha256"],
        "total_steps": total_steps,
        "sha256_digest": model_sha,
        "health_status": "HEALTHY",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "target_context_length": TARGET_CONTEXT,
        "target_capabilities": TARGET_CAPABILITIES,
        "context_extension_status": "target_only_not_trained",
        "dataset": "coding + agentic/MCP (One Touch 134,728-record corpus)",
        **telemetry,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (staging / "SHA256SUMS").write_text(f"{model_sha}  model.safetensors\n", encoding="utf-8")

    tier_root.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final_dir)
    latest = tier_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(final_dir, latest)
    return final_dir


def train_tier(
    tier: str,
    seconds: float,
    seq_len: int,
    batch_size: int,
    lr: float,
    data_limit: int,
    eval_limit: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Resume a tier from its verified parent and run a bounded continuation."""
    source = resolve_source(tier)
    parent = verify_source(source)
    parent["path"] = source
    config = parent["config"]
    config.max_seq_len = seq_len
    config.model_version = f"0.4.0-{tier}-coding-agents"

    module = ProgressiveTierProjectionHead(config, active_vocab=parent["state"]["embed_tokens.weight"].shape[0])
    module.load_state_dict(parent["state"], strict=True)
    module.train()

    tokenizer = load_or_create_tokenizer()
    train_records, eval_records, data_stats = select_records(data_limit, eval_limit)
    train_batches = prepare_token_batches(
        train_records, tokenizer, max_seq_len=seq_len, batch_size=batch_size, device="cpu"
    )
    eval_batches = prepare_token_batches(
        eval_records, tokenizer, max_seq_len=seq_len, batch_size=batch_size, device="cpu"
    )
    if not train_batches:
        raise RuntimeError(f"No training batches produced for {tier}")

    before = mean_loss(module, eval_batches)
    baseline = before
    if dry_run:
        return {
            "tier": tier,
            "parent": str(source),
            "parent_step": parent["step"],
            "parent_sha": parent["sha256"][:16],
            "seq_len": seq_len,
            "train_batches": len(train_batches),
            "eval_batches": len(eval_batches),
            "heldout_loss": before,
            "data": data_stats,
            "dry_run": True,
        }

    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.01)
    start = time.perf_counter()
    steps = tokens = 0
    first = last = float("nan")
    while time.perf_counter() - start < seconds:
        batch = train_batches[steps % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = module(batch["input_ids"], labels=batch["labels"])["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        steps += 1
        tokens += int((batch["labels"] != -100).sum())
        first = float(loss.item()) if steps == 1 else first
        last = float(loss.item())

    elapsed = time.perf_counter() - start
    after = mean_loss(module, eval_batches)
    total_steps = parent["step"] + steps
    telemetry = {
        "steps": steps,
        "seconds": round(elapsed, 2),
        "tokens": tokens,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "learning_rate": lr,
        "initial_loss": round(first, 6),
        "final_loss": round(last, 6),
        "heldout_loss_before": baseline,
        "heldout_loss_after": after,
        "heldout_improvement_pct": round(
            ((baseline - after) / baseline) * 100.0, 4
        )
        if baseline and math.isfinite(baseline)
        else 0.0,
        "data": data_stats,
    }
    final_dir = write_checkpoint(
        tier, module, config, tokenizer, parent, telemetry, total_steps
    )
    telemetry.update(
        {
            "tier": tier,
            "parent": str(source),
            "parent_step": parent["step"],
            "parent_sha": parent["sha256"][:16],
            "checkpoint": str(final_dir),
            "saved": True,
        }
    )
    return telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default="core,pro,flagship")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--data-limit", type=int, default=9000)
    parser.add_argument("--eval-limit", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found: {CORPUS}")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
    results = []
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        print(f"\n=== {tier} ===", flush=True)
        result = train_tier(
            tier,
            args.seconds,
            args.seq_len,
            args.batch_size,
            args.lr,
            args.data_limit,
            args.eval_limit,
            args.dry_run,
        )
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)

    if not args.dry_run:
        LINEAGE_ROOT.mkdir(parents=True, exist_ok=True)
        (LINEAGE_ROOT / "lineage.json").write_text(
            json.dumps(
                {
                    "lineage": "v0.4.0-coding-agents",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "target_context_length": TARGET_CONTEXT,
                    "target_capabilities": TARGET_CAPABILITIES,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nLineage written to {LINEAGE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
