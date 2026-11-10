"""H100/A100/L4 High-Throughput Pretraining Runner for Helios Family (2B Tokens Target).

Features:
- Native bfloat16 mixed precision with PyTorch SDPA (Scaled Dot-Product Attention).
- Resilient Auto-Downloader: If corpus or checkpoints are missing from Google Drive,
  it automatically fetches them from Hugging Face Hub (Darknecrocities/Helios-Pretrain-Corpus
  and Darknecrocities/Helios) directly into Drive so training never crashes!
- Continuous zero-loss Google Drive auto-checkpointing (`/content/drive/MyDrive/Helios/`).
- 100% Deduplication Guarantee: loads `curriculum_ledger.json`, uses cryptographic
  hash tracking (`seen_hashes`) and persistent category cursors so zero previously
  trained records are repeated.
- Preemption safety (SIGTERM/SIGINT) flushes all states immediately on compute unit loss.
- Automatic sync to Hugging Face Hub (`Darknecrocities/Helios`).
- Ephemeral cache auto-eviction (/tmp, ~/.cache/huggingface) to prevent Colab disk OOM.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from helios.cli.utils import load_or_create_tokenizer
from helios.model import HeliosBlock, HeliosConfig, HeliosForCausalLM, RMSNorm
TRAIN_TIERS = ["nano", "alpha", "beta", "core", "pro", "flagship"]
TIER_PRESET_MAP = {
    "nano": "nano-20m",
    "alpha": "alpha-100m",
    "beta": "beta-300m",
    "core": "core-1b",
    "pro": "pro-3b",
    "flagship": "flagship-6b",
}
from scripts.colab_dedup import get_deduplicated_record_slice
from scripts.pretrain_helios import load_dataset_records, prepare_token_batches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.h100_runner")

TERMINATION_REQUESTED = False


def sigterm_handler(signum: int, frame: Any) -> None:
    global TERMINATION_REQUESTED
    logger.warning("Caught signal %d (Preemption / Compute Unit Cut). Flushing states safely...", signum)
    TERMINATION_REQUESTED = True


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


def configure_h100_environment() -> tuple[torch.device, torch.dtype]:
    """Detects and configures maximum throughput optimizations for H100/A100/L4 GPUs."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA device detected. Falling back to CPU.")
        return torch.device("cpu"), torch.float32

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("Configuring High-Throughput GPU: %s (VRAM: %.1f GB)", gpu_name, vram_gb)

    # Enable TF32 for matrix multiplications on Ampere/Hopper/Ada
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention) enabled.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info("Using precision: %s", dtype)
    return device, dtype


def clean_ephemeral_cache() -> None:
    """Evicts GPU and filesystem memory caches to prevent Colab disk saturation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for p in [Path("/tmp"), Path.home() / ".cache" / "huggingface", Path.home() / ".cache" / "torch"]:
        if p.exists() and p.is_dir():
            for item in p.glob("*"):
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                except Exception:
                    pass


def ensure_drive_corpus_and_ledger(drive_root: Path) -> tuple[Path, Path]:
    """Ensures the omni-domain corpus and ledger exist in Drive; downloads from HF if missing."""
    datasets_dir = drive_root / "datasets"
    manifests_dir = drive_root / "manifests"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = datasets_dir / "helios_massive_pretrain_corpus.jsonl"
    ledger_path = manifests_dir / "curriculum_ledger.json"

    # 1. Check local repo fallback
    repo_corpus = Path("data/datasets/helios_massive_pretrain_corpus.jsonl")
    if not corpus_path.exists() and repo_corpus.exists():
        logger.info("Copying corpus from repo to Drive: %s", corpus_path)
        import shutil
        shutil.copy2(repo_corpus, corpus_path)

    # 2. If still missing, download directly from Hugging Face Hub dataset repo
    if not corpus_path.exists():
        logger.info("Corpus missing in Drive. Downloading from Hugging Face Hub (Darknecrocities/Helios-Pretrain-Corpus)...")
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        dl_path = hf_hub_download(
            repo_id="Darknecrocities/Helios-Pretrain-Corpus",
            filename="helios_massive_pretrain_corpus.jsonl",
            repo_type="dataset",
            token=token,
        )
        import shutil
        shutil.copy2(dl_path, corpus_path)
        logger.info("Corpus downloaded successfully to %s", corpus_path)

    # 3. Check ledger
    if not ledger_path.exists():
        try:
            logger.info("Ledger missing in Drive. Downloading from Hugging Face Hub...")
            from huggingface_hub import hf_hub_download
            token = os.environ.get("HF_TOKEN")
            dl_path = hf_hub_download(
                repo_id="Darknecrocities/Helios-Pretrain-Corpus",
                filename="curriculum_ledger.json",
                repo_type="dataset",
                token=token,
            )
            import shutil
            shutil.copy2(dl_path, ledger_path)
            logger.info("Ledger downloaded successfully to %s", ledger_path)
        except Exception as e:
            logger.warning("Could not download ledger from HF (%s); starting fresh ledger.", e)

    return corpus_path, ledger_path


def load_trained_tokenizer(drive_root: Path):
    """Load the tokenizer all checkpoints were trained with; refuse to train on a fresh one."""
    tok_dir = drive_root / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    if not (tok_dir / "tokenizer.json").exists():
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        for fname in ("tokenizer.json", "special_tokens_map.json"):
            dl = hf_hub_download("Darknecrocities/Helios", f"tokenizer/{fname}", token=token)
            import shutil
            shutil.copy2(dl, tok_dir / fname)
    tok = load_or_create_tokenizer(tok_dir)
    sample = "Napoleon Bonaparte was a French military leader who rose to prominence during the French Revolution."
    chars_per_token = len(sample) / len(tok.encode(sample))
    if chars_per_token < 1.8:  # trained BPE ~2.7; an untrained byte-level one is ~1.0
        raise RuntimeError(f"Tokenizer looks untrained ({chars_per_token:.2f} chars/token); aborting.")
    logger.info("Loaded trained tokenizer from %s (vocab=%d)", tok_dir, tok.vocab_size)
    return tok


def ensure_tier_weights(tier: str, tier_out: Path) -> None:
    """Ensures checkpoint weights exist; downloads continuous weights from HF if missing."""
    latest_dir = tier_out / "latest"
    latest_st = latest_dir / "model.safetensors"
    if latest_st.exists():
        return

    latest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        logger.info("Fetching existing %s checkpoint from Hugging Face (Darknecrocities/Helios)...", tier)
        for fname in ["model.safetensors", "config.json", "manifest.json", "SHA256SUMS"]:
            try:
                dl = hf_hub_download(
                    repo_id="Darknecrocities/Helios",
                    filename=f"continuous/{tier}/{fname}",
                    repo_type="model",
                    token=token,
                )
                import shutil
                shutil.copy2(dl, latest_dir / fname)
            except Exception:
                pass
        if latest_st.exists():
            logger.info("Successfully fetched %s checkpoint to Drive.", tier)
    except Exception as e:
        logger.warning("Could not fetch %s from HF: %s", tier, e)


def save_checkpoint_to_drive(
    tier: str,
    step: int,
    loss: float,
    model: nn.Module,
    config: HeliosConfig,
    out_dir: Path,
) -> str:
    """Saves SafeTensors checkpoint and manifests to Google Drive."""
    step_dir = out_dir / f"step_{step:08d}"
    latest_dir = out_dir / "latest"
    step_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    unwrapped_model = getattr(model, "_orig_mod", model)
    state_dict = {k: v.detach().cpu().contiguous().clone() for k, v in unwrapped_model.state_dict().items()}
    st_path = step_dir / "model.safetensors"
    save_file(state_dict, str(st_path))

    h = hashlib.sha256()
    with open(st_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    sha256 = h.hexdigest()

    manifest = {
        "tier": tier,
        "step": step,
        "loss": round(float(loss), 4),
        "sha256": sha256,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "COLAB_GPU",
    }
    with open(step_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(step_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Mirror to latest
    for f in [st_path, step_dir / "manifest.json", step_dir / "config.json"]:
        dst = latest_dir / f.name
        with open(f, "rb") as sf, open(dst, "wb") as df:
            df.write(sf.read())
    with open(latest_dir / "SHA256SUMS", "w") as f:
        f.write(f"{sha256}  model.safetensors\n")

    logger.info("Saved %s checkpoint step=%d loss=%.4f (SHA=%s)", tier, step, loss, sha256[:12])
    return sha256


def push_to_huggingface_hub(tier: str, latest_dir: Path, hf_repo: str = "Darknecrocities/Helios") -> None:
    """Synchronizes updated checkpoint to Hugging Face Hub using HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.info("HF_TOKEN not set; skipping Hugging Face Hub push.")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        logger.info("Pushing %s checkpoint to Hugging Face (%s)...", tier, hf_repo)
        api.upload_folder(
            folder_path=str(latest_dir),
            path_in_repo=f"continuous/{tier}",
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"Auto-sync {tier} checkpoint from H100 pretraining runner",
        )
        logger.info("Successfully pushed %s to Hugging Face Hub.", tier)
    except Exception as e:
        logger.warning("Could not push to Hugging Face: %s", e)


def push_ledger_to_hub(ledger_path: Path) -> None:
    """Mirrors the dedup ledger to the HF dataset repo so local One Touch sync stays current."""
    token = os.environ.get("HF_TOKEN")
    if not token or not ledger_path.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=str(ledger_path),
            path_in_repo="curriculum_ledger.json",
            repo_id="Darknecrocities/Helios-Pretrain-Corpus",
            repo_type="dataset",
            commit_message="Auto-sync curriculum ledger from Colab runner",
        )
    except Exception as e:
        logger.warning("Could not push ledger to Hugging Face: %s", e)


def run_h100_pretraining(
    drive_root: Path,
    target_tokens: int = 2_000_000_000,
    batch_size: int = 64,
    block_size: int = 1024,
    enable_compile: bool = True,
    hf_sync: bool = True,
    lr: float = 5e-5,
    warmup_steps: int = 20,
    slice_size: int = 4000,
) -> None:
    """Continuous high-throughput pretraining loop scaling across all 6 tiers."""
    device, dtype = configure_h100_environment()

    checkpoints_dir = drive_root / "checkpoints"
    manifests_dir = drive_root / "manifests"

    corpus_path, ledger_path = ensure_drive_corpus_and_ledger(drive_root)

    tok = load_trained_tokenizer(drive_root)
    ledger: dict[str, Any] = {"category_cursors": {"all": 0}, "category_epochs": {"all": 1}, "seen_hashes": []}
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                ledger = json.load(f)
            logger.info("Resumed curriculum: cursor=%d, seen_hashes=%d",
                        ledger.get("category_cursors", {}).get("all", 0), len(ledger.get("seen_hashes", [])))
        except Exception as e:
            logger.warning("Could not parse ledger (%s); continuing fresh.", e)

    records = load_dataset_records(corpus_path)
    total_records = len(records)
    logger.info("Loaded %d corpus records for high-throughput scaling.", total_records)

    total_tokens_trained = 0
    cycle = 0

    while total_tokens_trained < target_tokens and not TERMINATION_REQUESTED:
        cycle += 1
        logger.info("=== Training Cycle #%d | Progress: %d / %d tokens (%.2f%%) ===",
                    cycle, total_tokens_trained, target_tokens, (total_tokens_trained / target_tokens) * 100)

        for tier in TRAIN_TIERS:
            if TERMINATION_REQUESTED:
                break

            preset_name = TIER_PRESET_MAP[tier]
            cfg = HeliosConfig.from_preset(preset_name)
            cfg.vocab_size = tok.vocab_size
            cfg.max_seq_len = block_size

            tier_out = checkpoints_dir / f"{tier}_continuous"
            ensure_tier_weights(tier, tier_out)

            raw_model = HeliosForCausalLM(cfg)
            n_params = sum(p.numel() for p in raw_model.parameters())
            fp32_master = n_params <= 4_000_000_000  # AdamW fp32 state ~16 B/param must fit in 80GB
            raw_model = raw_model.to(device=device, dtype=torch.float32 if fp32_master else dtype)
            logger.info("%s: %.2fM params, %s weights + %s autocast", tier, n_params / 1e6,
                        "fp32" if fp32_master else "bf16", dtype)

            # Resume existing weights if present
            latest_st = tier_out / "latest" / "model.safetensors"
            current_step = 0
            if latest_st.exists():
                try:
                    from safetensors.torch import load_file
                    raw_model.load_state_dict(load_file(str(latest_st)), strict=False)
                    mf = tier_out / "latest" / "manifest.json"
                    if mf.exists():
                        with open(mf, "r") as f:
                            mdata = json.load(f)
                            current_step = int(mdata.get("step") or mdata.get("total_steps") or 0)
                    logger.info("Resumed %s from step %d", tier, current_step)
                except Exception as ex:
                    logger.warning("Could not resume %s: %s", tier, ex)

            # Optional torch.compile
            model = raw_model
            if enable_compile and torch.cuda.is_available() and hasattr(torch, "compile"):
                try:
                    model = torch.compile(raw_model, mode="reduce-overhead")
                    logger.info("Compiled %s model via TorchDynamo/Inductor.", tier)
                except Exception as e:
                    logger.warning("torch.compile fallback: %s", e)
                    model = raw_model

            # Guaranteed 0% Duplication Record Slice
            batch_recs, start_cur, end_cur = get_deduplicated_record_slice(records, ledger, slice_size=slice_size)

            batches = prepare_token_batches(batch_recs, tok, max_seq_len=block_size, batch_size=batch_size, device=device)
            if not batches:
                continue

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
            model.train()
            running_loss = 0.0
            steps_in_tier = 0
            t0 = time.time()

            for batch in batches:
                if TERMINATION_REQUESTED:
                    break
                input_ids = batch["input_ids"].to(device=device)
                labels = batch["labels"].to(device=device)
                optimizer.zero_grad()

                for g in optimizer.param_groups:  # linear warmup protects resumed weights
                    g["lr"] = lr * min(1.0, (steps_in_tier + 1) / warmup_steps)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    out = model(input_ids, labels=labels)
                loss = out.loss if hasattr(out, "loss") else out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = float(loss.item())
                running_loss += loss_val
                steps_in_tier += 1
                current_step += 1
                tokens_in_step = input_ids.numel()
                total_tokens_trained += tokens_in_step

            elapsed = max(0.001, time.time() - t0)
            tok_s = (steps_in_tier * batch_size * block_size) / elapsed
            avg_loss = running_loss / max(1, steps_in_tier)
            logger.info("[%s] Steps: %d | Avg Loss: %.4f | Speed: %.1f tok/s | Elapsed: %.1fs",
                        tier.upper(), steps_in_tier, avg_loss, tok_s, elapsed)

            save_checkpoint_to_drive(tier, current_step, avg_loss, model, cfg, tier_out)

            # Persist ledger with updated cursor and seen_hashes to Drive
            manifests_dir.mkdir(parents=True, exist_ok=True)
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2)

            if hf_sync:
                push_to_huggingface_hub(tier, tier_out / "latest")
                push_ledger_to_hub(ledger_path)

            del model, raw_model, optimizer
            clean_ephemeral_cache()

    logger.info("Training cycle complete. Total tokens trained: %d", total_tokens_trained)


def main() -> None:
    parser = argparse.ArgumentParser(description="Helios High-Throughput Pretrainer")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/Helios"))
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--slice-size", type=int, default=4000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-hf-sync", action="store_true")
    args = parser.parse_args()

    run_h100_pretraining(
        drive_root=args.drive_root,
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        block_size=args.block_size,
        enable_compile=not args.no_compile,
        hf_sync=not args.skip_hf_sync,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        slice_size=args.slice_size,
    )


if __name__ == "__main__":
    main()
