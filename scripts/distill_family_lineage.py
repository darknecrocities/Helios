#!/usr/bin/env python3
"""Distill the Helios family in a capacity cascade, starting with Flagship.

The gate is intentionally strict: a teacher must be a **full** checkpoint with
``9 * n_layers + 3`` tensors and no ``block.*`` keys. The current Flagship
checkpoint has 12 tensors and is a single-block slice, so distillation is
blocked until a full Flagship exists.

Intended cascade
----------------
Flagship (5.80B) -> Pro (3.32B) -> Core (1.13B) -> Beta (303M) -> Alpha (100M)
-> Nano (17.84M).

Each student can become the next smaller teacher. Direct Flagship -> every
student is also supported with ``--teacher-tier flagship``.

Usage
-----
    # Show the schedule and memory estimates (no weights loaded)
    python scripts/distill_family_lineage.py --plan

    # Check whether the current Flagship can teach (it cannot yet)
    python scripts/distill_family_lineage.py --teacher-tier flagship --student-tier pro --dry-run

    # Real distillation on a GPU host with a full teacher
    python scripts/distill_family_lineage.py \
        --teacher-tier flagship --student-tier pro \
        --seconds 600 --seq-len 512 --batch-size 1 --lr 2e-5

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from helios.cli.utils import load_or_create_tokenizer, resolve_device
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from scripts.continue_full_family_lineage import (
    CHECKPOINTS,
    expected_tensor_count,
    training_memory_gib,
    unique_parameter_count,
)
from scripts.continue_full_family_lineage import (
    LINEAGE_ROOT as FULL_LINEAGE_ROOT,
)
from scripts.continue_v040_lineage import select_records
from scripts.pretrain_helios import prepare_token_batches

DISTILL_ROOT = CHECKPOINTS / "v060_distill_lineage"
TIER_ORDER = ["flagship", "pro", "core", "beta", "alpha", "nano"]
PRESETS = {
    "flagship": "flagship-6b",
    "pro": "pro-3b",
    "core": "core-1b",
    "beta": "beta-300m",
    "alpha": "alpha-100m",
    "nano": "nano-20m",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(tier: str) -> Path | None:
    """Find the newest full checkpoint for a tier, preferring distilled/full lineages."""
    candidates = [
        DISTILL_ROOT / f"{tier}_distilled" / "latest",
        FULL_LINEAGE_ROOT / f"{tier}_full" / "latest",
        CHECKPOINTS / f"helios_{tier}",
    ]
    for candidate in candidates:
        if (candidate / "state.pt").is_file():
            return candidate
    return None


def verify_full_checkpoint(tier: str, path: Path) -> tuple[HeliosConfig, dict[str, Any], int]:
    """Verify that a checkpoint is a full model, returning config, payload, and step."""
    config = HeliosConfig.from_pretrained(path)
    expected = expected_tensor_count(config)
    payload = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise SystemExit(f"{tier}: empty state.pt at {path}")
    if any(key.startswith("block.") for key in state):
        raise SystemExit(
            f"TEACHER GATE FAILED: {tier} at {path} is a block slice "
            f"({len(state)} tensors). A distillation teacher must be a full model "
            f"with {expected} tensors and no block.* keys."
        )
    layers = {key.split(".")[1] for key in state if key.startswith("layers.")}
    if len(state) != expected or len(layers) != config.n_layers:
        raise SystemExit(
            f"TEACHER GATE FAILED: {tier} has {len(state)} tensors / {len(layers)} layers; "
            f"expected {expected} tensors / {config.n_layers} layers."
        )
    step = int(payload.get("step", 0) or 0)
    return config, payload, step


def plan() -> None:
    """Print the cascade and the minimum training memory for each pair."""
    print("Helios distillation cascade (teacher -> student)")
    print("=" * 92)
    print(
        "Stage  Teacher        Student        Student unique   Teacher+student+AdamW   Minimum GPU"
    )
    for index, student in enumerate(TIER_ORDER[1:], start=1):
        teacher = TIER_ORDER[index - 1]
        student_cfg = HeliosConfig.from_preset(PRESETS[student])
        teacher_cfg = HeliosConfig.from_preset(PRESETS[teacher])
        student_mem = training_memory_gib(student_cfg)
        teacher_weights = (
            unique_parameter_count(teacher_cfg) + teacher_cfg.vocab_size * teacher_cfg.d_model
        ) * 2 / (1024**3)
        total = student_mem + teacher_weights
        minimum = 24 if total < 20 else 48 if total < 40 else 80 if total < 70 else 141
        print(
            f"{index:<6} {teacher:<14} {student:<14} "
            f"{unique_parameter_count(student_cfg) / 1e6:>14.2f}M "
            f"{total:>23.2f} GiB {minimum:>12} GB"
        )
    print("\nDirect teacher option: --teacher-tier flagship --student-tier <tier>")
    print("The current Flagship checkpoint is a block slice and cannot be a teacher.")


def distill_tier(
    teacher_tier: str,
    student_tier: str,
    seconds: float,
    seq_len: int,
    batch_size: int,
    lr: float,
    temperature: float,
    alpha: float,
    data_limit: int,
    eval_limit: int,
    device_name: str,
    dry_run: bool,
    allow_random_init: bool,
) -> dict[str, Any]:
    """Run logit + cross-entropy distillation from a full teacher to a student."""
    teacher_path = resolve_checkpoint(teacher_tier)
    if teacher_path is None:
        raise SystemExit(f"{teacher_tier}: no checkpoint found")
    teacher_config, teacher_payload, teacher_step = verify_full_checkpoint(
        teacher_tier, teacher_path
    )

    student_path = resolve_checkpoint(student_tier)
    if student_path is None and not allow_random_init:
        raise SystemExit(
            f"{student_tier}: no full student checkpoint. Pass --allow-random-init only "
            "on a GPU host if you intend to train from scratch."
        )
    if student_path is not None:
        student_config, student_payload, student_step = verify_full_checkpoint(
            student_tier, student_path
        )
    else:
        student_config = HeliosConfig.from_preset(PRESETS[student_tier])
        student_payload = {"step": 0}
        student_step = 0

    if teacher_config.vocab_size != student_config.vocab_size:
        raise SystemExit("Teacher and student vocabularies must match for logit distillation")

    teacher_mem = (
        unique_parameter_count(teacher_config) + teacher_config.vocab_size * teacher_config.d_model
    ) * 2 / (1024**3)
    student_mem = training_memory_gib(student_config)
    estimated = round(teacher_mem + student_mem, 2)
    if dry_run:
        return {
            "teacher": teacher_tier,
            "teacher_path": str(teacher_path),
            "teacher_step": teacher_step,
            "student": student_tier,
            "student_path": str(student_path) if student_path else "random_init",
            "student_step": student_step,
            "estimated_memory_gib": estimated,
            "dry_run": True,
        }

    device = resolve_device(device_name) if device_name != "auto" else resolve_device("auto")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram < estimated * 1.25:
            raise SystemExit(
                f"Need about {estimated:.1f} GiB for teacher+student+AdamW; "
                f"this GPU has {vram:.1f} GiB."
            )

    teacher = HeliosForCausalLM(teacher_config)
    teacher.load_state_dict(teacher_payload["model_state_dict"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = HeliosForCausalLM(student_config)
    if student_path is not None:
        student.load_state_dict(student_payload["model_state_dict"], strict=True)
    student.to(device).train()

    tokenizer = load_or_create_tokenizer()
    train_records, eval_records, data_stats = select_records(data_limit, eval_limit)
    train_batches = prepare_token_batches(
        train_records, tokenizer, max_seq_len=seq_len, batch_size=batch_size, device=device
    )
    eval_batches = prepare_token_batches(
        eval_records, tokenizer, max_seq_len=seq_len, batch_size=batch_size, device=device
    )
    if not train_batches:
        raise RuntimeError("No distillation batches produced")

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    start = time.perf_counter()
    steps = tokens = 0
    first_kl = first_ce = last_kl = last_ce = float("nan")
    while time.perf_counter() - start < seconds:
        batch = train_batches[steps % len(train_batches)]
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(input_ids)["logits"]
        student_logits = student(input_ids)["logits"]
        kl = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
        ce = F.cross_entropy(
            student_logits[..., :-1, :].contiguous().view(-1, student_config.vocab_size),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        loss = alpha * kl + (1.0 - alpha) * ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        steps += 1
        tokens += int((labels != -100).sum())
        if steps == 1:
            first_kl, first_ce = float(kl.item()), float(ce.item())
        last_kl, last_ce = float(kl.item()), float(ce.item())

    elapsed = time.perf_counter() - start
    total_steps = student_step + steps
    final_dir = DISTILL_ROOT / f"{student_tier}_distilled" / f"step_{total_steps:08d}"
    staging = DISTILL_ROOT / "staging" / f"{student_tier}_{total_steps}_{os.getpid()}"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {final_dir}")
    staging.mkdir(parents=True, exist_ok=True)

    state = {key: value.detach().cpu().contiguous().clone() for key, value in student.state_dict().items()}
    torch.save(
        {
            "model_state_dict": state,
            "step": total_steps,
            "teacher_tier": teacher_tier,
            "teacher_step": teacher_step,
            "teacher_checkpoint": str(teacher_path),
            "lineage": f"v0.6.0-{student_tier}-distilled-from-{teacher_tier}",
            "training_mode": "logit_distillation",
            "config": student_config.to_dict(),
        },
        staging / "state.pt",
    )
    save_file(state, str(staging / "model.safetensors"))
    (staging / "config.json").write_text(json.dumps(student_config.to_dict(), indent=2), encoding="utf-8")
    tokenizer.save_pretrained(staging)
    model_sha = sha256_file(staging / "model.safetensors")
    manifest = {
        "tier": student_tier,
        "lineage": f"v0.6.0-{student_tier}-distilled-from-{teacher_tier}",
        "mode": "logit_distillation",
        "teacher_tier": teacher_tier,
        "teacher_step": teacher_step,
        "teacher_checkpoint": str(teacher_path),
        "teacher_tensor_count": expected_tensor_count(teacher_config),
        "student_tensor_count": len(state),
        "unique_parameters": unique_parameter_count(student_config),
        "total_steps": total_steps,
        "steps": steps,
        "tokens": tokens,
        "seconds": round(elapsed, 2),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "learning_rate": lr,
        "temperature": temperature,
        "alpha_kl": alpha,
        "initial_kl": round(first_kl, 6),
        "final_kl": round(last_kl, 6),
        "initial_ce": round(first_ce, 6),
        "final_ce": round(last_ce, 6),
        "heldout_batches": len(eval_batches),
        "data": data_stats,
        "sha256_digest": model_sha,
        "health_status": "HEALTHY",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (staging / "SHA256SUMS").write_text(f"{model_sha}  model.safetensors\n", encoding="utf-8")
    (DISTILL_ROOT / f"{student_tier}_distilled").mkdir(parents=True, exist_ok=True)
    os.replace(staging, final_dir)
    latest = DISTILL_ROOT / f"{student_tier}_distilled" / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(final_dir, latest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-tier", default="flagship")
    parser.add_argument("--student-tier", default="pro")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7, help="KL weight; CE weight is 1-alpha")
    parser.add_argument("--data-limit", type=int, default=9000)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    if args.plan:
        plan()
        return

    result = distill_tier(
        args.teacher_tier,
        args.student_tier,
        args.seconds,
        args.seq_len,
        args.batch_size,
        args.lr,
        args.temperature,
        args.alpha,
        args.data_limit,
        args.eval_limit,
        args.device,
        args.dry_run,
        args.allow_random_init,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
