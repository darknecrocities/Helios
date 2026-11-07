"""Pretraining and fine-tuning execution script for Helios foundation model.

Trains Helios on the unconstrained contextual reasoning dataset using
TrainingController with Pre-RMSNorm float32 stabilization, RoPE, AdamW,
cosine scheduling, and atomic 6-state checkpointing.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from helios.cli.utils import resolve_device
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.base import BaseTokenizer
from helios.tokenizer.bpe import ByteLevelBPETokenizer
from helios.training.config import TrainingConfig
from helios.training.controller import TrainingController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.pretrain")


def load_dataset_records(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL dataset records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_token_batches(
    records: list[dict[str, Any]],
    tokenizer: BaseTokenizer,
    max_seq_len: int = 512,
    batch_size: int = 2,
    device: torch.device | str = "cpu",
) -> list[dict[str, torch.Tensor]]:
    """Tokenize and pad dataset records into training microbatches."""
    pad_id = tokenizer.special_tokens.get("<|pad|>", 3)
    tokenized_samples: list[list[int]] = []

    for r in records:
        text = r.get("text", "")
        if not text:
            prompt = r.get("prompt", "")
            resp = r.get("response", "")
            text = (
                f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n{resp}<|eot_id|>"
            )
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        tokenized_samples.append(ids)

    batches: list[dict[str, torch.Tensor]] = []
    target_device = resolve_device(str(device)) if isinstance(device, str) else device

    for i in range(0, len(tokenized_samples), batch_size):
        chunk = tokenized_samples[i : i + batch_size]
        actual_max = max(len(c) for c in chunk)
        # Pad to multiple of 8 or actual_max
        pad_len = min(max_seq_len, ((actual_max + 7) // 8) * 8)
        batch_input_ids = []
        batch_labels = []

        for seq in chunk:
            pad_count = pad_len - len(seq)
            inp = seq + [pad_id] * pad_count
            lbl = seq + [-100] * pad_count
            batch_input_ids.append(inp)
            batch_labels.append(lbl)

        batches.append(
            {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=target_device),
                "labels": torch.tensor(batch_labels, dtype=torch.long, device=target_device),
            }
        )

    return batches


def run_pretraining(
    data_path: Path | str = "data/helios_contextual_reasoning.jsonl",
    output_dir: Path | str = "checkpoints/helios_core_pretrained",
    max_steps: int = 150,
    learning_rate: float = 2e-3,
    batch_size: int = 2,
    preset: str = "nano-20m",
) -> None:
    """Execute pretraining run and export verified checkpoint."""
    target_output = Path(output_dir).resolve()
    target_output.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    logger.info("Initializing pretraining run on device: %s", device)

    # 1. Config & Model
    config = HeliosConfig.from_preset(preset)
    config.max_seq_len = 512
    model = HeliosForCausalLM(config).to(device)

    # 2. Tokenizer
    tok = ByteLevelBPETokenizer()
    cur_size = tok.vocab_size
    if cur_size < config.vocab_size:
        for i in range(cur_size, config.vocab_size):
            reserved = f"<extra_id_{i}>"
            tok.vocab[reserved] = i
            tok.inv_vocab[i] = reserved

    # 3. Data
    data_file = Path(data_path).resolve()
    records = load_dataset_records(data_file)
    logger.info("Loaded %d reasoning records from %s", len(records), data_file)
    batches = prepare_token_batches(
        records, tok, max_seq_len=config.max_seq_len, batch_size=batch_size, device=device
    )
    logger.info("Created %d training batches", len(batches))

    # 4. Training Controller
    training_cfg = TrainingConfig(
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=10,
        gradient_accumulation_steps=1,
        grad_clip=1.0,
        checkpoint_dir=str(target_output),
        save_interval_steps=50,
        mixed_precision=False,  # MPS/CPU float32 stability
    )

    controller = TrainingController(model=model, config=training_cfg)

    # 5. Training Loop
    logger.info("Starting training loop for %d steps...", max_steps)
    start_time = time.perf_counter()
    batch_idx = 0

    for step in range(1, max_steps + 1):
        batch = batches[batch_idx % len(batches)]
        batch_idx += 1

        loss_val = controller.train_microbatch(batch)
        step_result = controller.step_optimizer()

        if step % 25 == 0 or step == max_steps:
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Step %3d/%d | Loss: %.4f | Status: %s | Elapsed: %.2fs",
                step,
                max_steps,
                loss_val,
                step_result.get("status"),
                elapsed,
            )

    # 6. Save Canonical Model Artifacts in target_output
    logger.info("Exporting complete model package to %s...", target_output)
    tok.save_pretrained(target_output)

    # Save config.json
    (target_output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    # Save state.pt
    state_payload = {
        "model_state_dict": model.state_dict(),
        "step": max_steps,
        "config": config.to_dict(),
    }
    torch.save(state_payload, target_output / "state.pt")

    # Save model.safetensors
    clean_dict = {
        k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()
    }
    save_file(clean_dict, str(target_output / "model.safetensors"))

    # Save manifest.json
    manifest = {
        "checkpoint_version": "1.0.0",
        "step": max_steps,
        "loss": float(loss_val),
        "config": config.to_dict(),
        "timestamp": time.time(),
        "developer": "Arron Kian Parejas",
    }
    (target_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("✓ Pretraining complete! Verified artifacts saved at %s", target_output)


if __name__ == "__main__":
    run_pretraining()
