"""Continue *full-parameter* Helios tiers in a new, conflict-free lineage.

Why this exists
---------------
The v0.4.0 continuation runner operates on ``ProgressiveTierProjectionHead``
block slices for Core/Pro/Flagship. That is useful for pipeline validation but
it is not full-model training. This runner only trains tiers whose parent
checkpoint is a real full model:

- Nano  (``helios_nano``,  66 tensors, 17.84M unique params)
- Alpha (``helios_alpha``, 111 tensors, 100.09M unique params)
- Beta  (``helios_beta``,  219 tensors, 303.35M unique params)

Core/Pro/Flagship have only 12-tensor block slices on the One Touch drive.
They are refused unless ``--allow-heavy`` is supplied on a CUDA host with
enough VRAM, because a 16 GB M3 cannot hold their weights, gradients, and
AdamW state at the same time.

Distillation is deliberately not attempted here. A teacher must first pass the
full-model gate (no ``block.*`` keys and the expected tensor count). The
current Flagship checkpoint is a block slice, so it cannot teach anyone.

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

from helios.cli.utils import load_or_create_tokenizer, resolve_device
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from scripts.continue_v040_lineage import select_records
from scripts.pretrain_helios import prepare_token_batches

ONETOUCH = Path("/Volumes/One Touch/Helios")
CHECKPOINTS = ONETOUCH / "checkpoints"
LINEAGE_ROOT = CHECKPOINTS / "v050_full_lineage"

TIERS: dict[str, dict[str, Any]] = {
    "nano": {"preset": "nano-20m", "parent": "helios_nano"},
    "alpha": {"preset": "alpha-100m", "parent": "helios_alpha"},
    "beta": {"preset": "beta-300m", "parent": "helios_beta"},
    "core": {"preset": "core-1b", "parent": None},
    "pro": {"preset": "pro-3b", "parent": None},
    "flagship": {"preset": "flagship-6b", "parent": None},
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tensor_count(config: HeliosConfig) -> int:
    """Expected full state-dict tensor count for a tied-embedding Helios model."""
    return 1 + config.n_layers * 9 + 1 + 1


def unique_parameter_count(config: HeliosConfig) -> int:
    """Advertised unique parameter count (tied embeddings counted once)."""
    return int(config.count_parameters()["total_parameters"])


def state_dict_parameter_count(config: HeliosConfig) -> int:
    """Tensor-sum parameter count when tied embeddings are saved twice."""
    return unique_parameter_count(config) + config.vocab_size * config.d_model


def training_memory_gib(config: HeliosConfig, dtype_bytes: int = 2) -> float:
    """Estimate weights + gradients + AdamW moments for full fine-tuning.

    This is an intentionally optimistic lower bound: it excludes activations,
    attention workspace, allocator fragmentation, and the operating system.
    """
    unique = unique_parameter_count(config)
    state_params = state_dict_parameter_count(config)
    weights = state_params * dtype_bytes
    grads = state_params * dtype_bytes
    adam = unique * 8  # two FP32 moments per unique parameter
    return round((weights + grads + adam) / (1024**3), 2)


def inspect_parent(tier: str) -> tuple[HeliosConfig, dict[str, Any], Path]:
    """Load and verify a full parent checkpoint, or explain why it cannot be used."""
    spec = TIERS[tier]
    parent_dir = CHECKPOINTS / str(spec["parent"])
    state_path = parent_dir / "state.pt"
    if not state_path.is_file():
        raise SystemExit(f"{tier}: no parent state.pt at {state_path}")

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not state:
        raise SystemExit(f"{tier}: empty parent state dict")

    # Prefer the directory config.json: older state.pt payloads omitted their
    # config, which made the tensor-count gate fall back to dataclass defaults.
    config_path = parent_dir / "config.json"
    if config_path.is_file():
        config = HeliosConfig.from_pretrained(parent_dir)
    else:
        config = HeliosConfig.from_dict(payload.get("config", {}))

    observed_layers = {key.split(".")[1] for key in state if key.startswith("layers.")}
    if observed_layers and len(observed_layers) != config.n_layers:
        raise SystemExit(
            f"{tier}: config says {config.n_layers} layers, but state.pt has "
            f"{len(observed_layers)} layer indices"
        )
    if any(key.startswith("block.") for key in state):
        raise SystemExit(
            f"{tier}: parent is a block slice, not a full model. "
            "Use the v0.4.0 block-slice runner for pipeline tests, or train a full model on a GPU."
        )

    expected = expected_tensor_count(config)
    if len(state) != expected:
        raise SystemExit(
            f"{tier}: parent has {len(state)} tensors, expected {expected} for a full model"
        )
    return config, payload, state_path


def write_checkpoint(
    tier: str,
    model: HeliosForCausalLM,
    config: HeliosConfig,
    tokenizer: Any,
    parent_path: Path,
    parent_step: int,
    telemetry: dict[str, Any],
    total_steps: int,
) -> Path:
    """Atomically save a full-model checkpoint into the new lineage."""
    tier_root = LINEAGE_ROOT / f"{tier}_full"
    final_dir = tier_root / f"step_{total_steps:08d}"
    staging = LINEAGE_ROOT / "staging" / f"{tier}_{total_steps}_{os.getpid()}"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {final_dir}")
    staging.mkdir(parents=True, exist_ok=True)

    state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    for tensor in state.values():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise RuntimeError(f"NaN/Inf detected before saving {tier}")

    unique_params = unique_parameter_count(config)
    state_params = sum(tensor.numel() for tensor in state.values())
    expected_tensors = expected_tensor_count(config)
    tensor_count = len(state)
    if tensor_count != expected_tensors:
        raise RuntimeError(f"{tier}: saved {tensor_count} tensors, expected {expected_tensors}")

    torch.save(
        {
            "model_state_dict": state,
            "step": total_steps,
            "parent_step": parent_step,
            "parent_checkpoint": str(parent_path),
            "lineage": f"v0.5.0-{tier}-full",
            "training_mode": "full_parameter_continuation",
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
        "lineage": f"v0.5.0-{tier}-full",
        "mode": "full_parameter_continuation",
        "parent_checkpoint": str(parent_path),
        "parent_step": parent_step,
        "total_steps": total_steps,
        "health_status": "HEALTHY",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "unique_parameters": unique_params,
        "state_dict_parameters": state_params,
        "expected_unique_parameters": unique_params,
        "parameter_match": state_params == state_dict_parameter_count(config),
        "tensor_count": tensor_count,
        "expected_tensor_count": expected_tensors,
        "full_model_verified": tensor_count == expected_tensors
        and not any(key.startswith("block.") for key in state),
        "sha256_digest": model_sha,
        "capabilities": {
            "text": "available",
            "coding": "local_smoke_test_only",
            "tools": "not_trained",
            "mcp": "not_implemented",
            "vision": "not_trained",
            "long_context": f"trained_at_{telemetry.get('seq_len', 0)}_tokens_this_run",
        },
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


def mean_loss(
    model: HeliosForCausalLM,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    """Compute mean held-out loss without gradient updates."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids, labels=labels)["loss"]
            losses.append(float(loss.item()))
    model.train()
    valid = [value for value in losses if math.isfinite(value)]
    return round(sum(valid) / len(valid), 6) if valid else float("nan")


def train_tier(
    tier: str,
    seconds: float,
    seq_len: int,
    batch_size: int,
    lr: float,
    data_limit: int,
    eval_limit: int,
    device_name: str,
    dry_run: bool,
    allow_heavy: bool,
) -> dict[str, Any]:
    """Continue one full tier from its verified parent."""
    if TIERS[tier]["parent"] is None:
        estimate = training_memory_gib(HeliosConfig.from_preset(TIERS[tier]["preset"]))
        message = (
            f"{tier}: no full parent checkpoint exists. A full model needs about "
            f"{estimate:.1f} GiB just for weights/gradients/AdamW before activations. "
            "This 16 GB M3 cannot train it. Run on an 80 GB+ CUDA GPU, or use the "
            "v0.4.0 block-slice runner for pipeline tests only."
        )
        if not allow_heavy:
            raise SystemExit(message)
        if not torch.cuda.is_available():
            raise SystemExit(f"{message} CUDA is not available on this host.")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram < estimate * 1.3:
            raise SystemExit(f"{message} This GPU has {vram:.1f} GiB VRAM.")

    config, parent_payload, parent_path = inspect_parent(tier)
    config.max_seq_len = seq_len
    config.model_version = f"0.5.0-{tier}-full"

    model = HeliosForCausalLM(config)
    model.load_state_dict(parent_payload["model_state_dict"], strict=True)
    device = resolve_device(device_name) if device_name != "auto" else resolve_device("auto")
    model.to(device)
    model.train()

    tokenizer = load_or_create_tokenizer()
    train_records, eval_records, data_stats = select_records(data_limit, eval_limit)
    train_batches = prepare_token_batches(
        train_records, tokenizer, max_seq_len=seq_len, batch_size=batch_size, device=device
    )
    eval_batches = prepare_token_batches(
        eval_records, tokenizer, max_seq_len=seq_len, batch_size=batch_size, device=device
    )
    if not train_batches:
        raise RuntimeError(f"{tier}: no training batches")

    before = mean_loss(model, eval_batches, device)
    unique_params = unique_parameter_count(config)
    state_params = sum(tensor.numel() for tensor in model.state_dict().values())
    tensor_count = len(model.state_dict())
    if dry_run:
        return {
            "tier": tier,
            "parent": str(parent_path),
            "parent_step": int(parent_payload.get("step", 0)),
            "device": str(device),
            "unique_parameters": unique_params,
            "state_dict_parameters": state_params,
            "tensor_count": tensor_count,
            "expected_tensor_count": expected_tensor_count(config),
            "estimated_training_gib": training_memory_gib(config),
            "train_batches": len(train_batches),
            "eval_batches": len(eval_batches),
            "heldout_loss": before,
            "data": data_stats,
            "dry_run": True,
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start = time.perf_counter()
    steps = tokens = 0
    first = last = float("nan")
    while time.perf_counter() - start < seconds:
        batch = train_batches[steps % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
        else:
            loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        tokens += int((batch["labels"] != -100).sum())
        first = float(loss.item()) if steps == 1 else first
        last = float(loss.item())

    elapsed = time.perf_counter() - start
    after = mean_loss(model, eval_batches, device)
    total_steps = int(parent_payload.get("step", 0)) + steps
    telemetry = {
        "steps": steps,
        "seconds": round(elapsed, 2),
        "tokens": tokens,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": str(device),
        "initial_loss": round(first, 6),
        "final_loss": round(last, 6),
        "heldout_loss_before": before,
        "heldout_loss_after": after,
        "heldout_improvement_pct": round(((before - after) / before) * 100.0, 4)
        if before and math.isfinite(before)
        else 0.0,
        "data": data_stats,
        "unique_parameters": unique_params,
        "state_dict_parameters": state_params,
        "tensor_count": tensor_count,
        "expected_tensor_count": expected_tensor_count(config),
        "estimated_training_gib": training_memory_gib(config),
    }
    final_dir = write_checkpoint(
        tier, model, config, tokenizer, parent_path, int(parent_payload.get("step", 0)), telemetry, total_steps
    )
    telemetry.update({"tier": tier, "checkpoint": str(final_dir), "saved": True})
    return telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default="nano,alpha,beta")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--data-limit", type=int, default=9000)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-heavy",
        action="store_true",
        help="Allow Core/Pro/Flagship training only on a CUDA host with enough VRAM",
    )
    args = parser.parse_args()

    if not ONETOUCH.is_dir():
        raise SystemExit(f"One Touch drive not found: {ONETOUCH}")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

    results = []
    for tier in [item.strip() for item in args.tiers.split(",") if item.strip()]:
        if tier not in TIERS:
            raise SystemExit(f"Unknown tier {tier}; valid: {', '.join(TIERS)}")
        print(f"\n=== {tier} ===", flush=True)
        result = train_tier(
            tier,
            args.seconds,
            args.seq_len,
            args.batch_size,
            args.lr,
            args.data_limit,
            args.eval_limit,
            args.device,
            args.dry_run,
            args.allow_heavy,
        )
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)

    if not args.dry_run:
        LINEAGE_ROOT.mkdir(parents=True, exist_ok=True)
        (LINEAGE_ROOT / "lineage.json").write_text(
            json.dumps(
                {
                    "lineage": "v0.5.0-full-family",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "note": "Full-parameter continuations only; Core/Pro/Flagship require cloud GPUs.",
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nLineage written to {LINEAGE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
