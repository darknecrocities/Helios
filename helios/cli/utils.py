"""Shared CLI utilities for Helios foundation ecosystem.

Handles hardware device resolution, model instantiation, checkpoint loading,
tokenizer initialization, and formatted console telemetry.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.base import BaseTokenizer, HeliosTokenizer
from helios.tokenizer.bpe import ByteLevelBPETokenizer
from helios.training.checkpoint import CheckpointManager


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve hardware device preferring Apple Silicon MPS, CUDA, then CPU."""
    dev = device_str.lower().strip()
    if dev in ("auto", "default"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if dev == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if dev.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(dev)
        return torch.device("cpu")

    return torch.device("cpu")


def load_model_and_config(
    preset: str = "nano-20m",
    model_path: str | Path | None = None,
    device: torch.device | str = "auto",
) -> tuple[HeliosForCausalLM, HeliosConfig]:
    """Instantiate Helios model from preset and optionally restore checkpoint weights."""
    target_device = resolve_device(str(device)) if isinstance(device, str) else device

    if model_path is not None:
        p = Path(model_path).resolve()
        if p.is_dir() and (p / "manifest.json").exists():
            manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
            config_dict = manifest.get("config", {})
            config = (
                HeliosConfig.from_dict(config_dict)
                if config_dict
                else HeliosConfig.from_preset(preset)
            )
        elif p.is_file() and p.name == "config.json":
            config = HeliosConfig.from_pretrained(p)
        else:
            config = HeliosConfig.from_preset(preset)
    else:
        config = HeliosConfig.from_preset(preset)

    model = HeliosForCausalLM(config).to(target_device)

    if model_path is not None:
        p = Path(model_path).resolve()
        if p.is_dir() and (p / "state.pt").exists():
            ckpt_mgr = CheckpointManager(checkpoint_dir=p.parent)
            ckpt_mgr.load_checkpoint(p, model=model, restore_rng=False)
        elif p.is_file():
            payload = torch.load(p, map_location=target_device, weights_only=False)
            state_dict = (
                payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            )
            model.load_state_dict(state_dict)

    model.eval()
    return model, config


def load_or_create_tokenizer(
    tokenizer_path: str | Path | None = None,
    vocab_size: int = 32000,
) -> BaseTokenizer:
    """Load pretrained tokenizer or initialize ByteLevelBPETokenizer with padded vocab."""
    if tokenizer_path is not None and Path(tokenizer_path).exists():
        return HeliosTokenizer.from_pretrained(tokenizer_path)

    tok = ByteLevelBPETokenizer()
    cur_size = tok.vocab_size
    if cur_size < vocab_size:
        for i in range(cur_size, vocab_size):
            reserved = f"<extra_id_{i}>"
            tok.vocab[reserved] = i
            tok.inv_vocab[i] = reserved

    return tok


def format_card(title: str, fields: list[tuple[str, Any]], width: int = 68) -> str:
    """Format structured key-value pairs into a clean text card."""
    border = "=" * width
    lines = [border, title.center(width), border]
    for key, val in fields:
        if not key:
            lines.append("-" * width)
        else:
            lines.append(f"  {key:<26}: {val}")
    lines.append(border)
    return "\n".join(lines)
