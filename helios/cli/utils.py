"""Shared CLI utilities for Helios foundation ecosystem.

Handles hardware device resolution, model instantiation, checkpoint loading,
model family catalog discovery, tokenizer initialization, and console telemetry.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
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

MODEL_FAMILY_CATALOG: dict[str, dict[str, str]] = {
    "nano": {
        "display_name": "Helios Nano",
        "preset": "nano-20m",
        "version": "v0.3.0-nano",
        "params": "17.84M",
        "context": "16,384 tokens (16K)",
        "rope_theta": "100,000.0",
        "q4_size": "9.4 MB",
        "ram_target": "~61 MB RAM @ 2K | ~92 MB @ 16K (Edge / Raspberry Pi / Any Laptop)",
    },
    "alpha": {
        "display_name": "Helios Alpha",
        "preset": "alpha-100m",
        "version": "v0.3.0-alpha",
        "params": "100.09M",
        "context": "32,768 tokens (32K)",
        "rope_theta": "500,000.0",
        "q4_size": "52.5 MB",
        "ram_target": "~138 MB RAM @ 2K | ~498 MB @ 32K (Everyday Chat & Bilingual QA)",
    },
    "beta": {
        "display_name": "Helios Beta",
        "preset": "beta-300m",
        "version": "v0.3.0-beta",
        "params": "303.35M",
        "context": "65,536 tokens (64K)",
        "rope_theta": "1,000,000.0",
        "q4_size": "159.1 MB",
        "ram_target": "~296 MB RAM @ 2K | ~1.74 GB @ 64K (Math & Deductive Logic)",
    },
    "core": {
        "display_name": "Helios Core",
        "preset": "core-1b",
        "version": "v0.3.0-core",
        "params": "1,128.89M (1.13B)",
        "context": "131,072 tokens (128K)",
        "rope_theta": "5,000,000.0",
        "q4_size": "592.1 MB",
        "ram_target": "~821 MB RAM @ 2K | ~3.71 GB Q8-KV @ 128K (8GB RAM Laptops)",
    },
    "pro": {
        "display_name": "Helios Pro",
        "preset": "pro-3b",
        "version": "v0.3.0-pro",
        "params": "3,319.73M (3.32B)",
        "context": "131,072 tokens (128K)",
        "rope_theta": "5,000,000.0",
        "q4_size": "1.70 GB",
        "ram_target": "~2.12 GB RAM @ 2K | ~7.38 GB Q8-KV @ 128K (8GB-16GB RAM Laptops)",
    },
    "flagship": {
        "display_name": "Helios Flagship",
        "preset": "flagship-6b",
        "version": "v0.3.0-flagship",
        "params": "5,802.04M (5.80B)",
        "context": "262,144 tokens (256K)",
        "rope_theta": "10,000,000.0",
        "q4_size": "2.97 GB",
        "ram_target": "~3.46 GB RAM @ 2K | ~11.2 GB HMA-Q8 @ 256K (16GB RAM Laptops)",
    },
}

PRESET_TO_TIER: dict[str, str] = {
    "nano": "nano",
    "nano-15m": "nano",
    "nano-20m": "nano",
    "nano-25m": "nano",
    "nano-35m": "nano",
    "helios-nano-20m": "nano",
    "alpha": "alpha",
    "alpha-100m": "alpha",
    "helios-alpha-100m": "alpha",
    "beta": "beta",
    "beta-300m": "beta",
    "helios-beta-300m": "beta",
    "core": "core",
    "core-1b": "core",
    "helios-core-1b": "core",
    "pro": "pro",
    "pro-3b": "pro",
    "helios-pro-3b": "pro",
    "flagship": "flagship",
    "flagship-6b": "flagship",
    "helios-flagship-6b": "flagship",
}


def resolve_tier_key(name: str) -> str:
    """Resolve user-supplied model tier or preset name into canonical family key."""
    cleaned = name.strip().lower().replace("_", "-")
    if cleaned.startswith("helios:"):
        cleaned = cleaned.split(":", 1)[1]
    if cleaned in PRESET_TO_TIER:
        return PRESET_TO_TIER[cleaned]
    raise KeyError(
        f"Unknown Helios model '{name}'. Choose from: {', '.join(MODEL_FAMILY_CATALOG.keys())}"
    )


def format_models_catalog(active_preset: str) -> str:
    """Format a structured table of all 6 Helios models for interactive selection."""
    try:
        active_key = resolve_tier_key(active_preset)
    except KeyError:
        active_key = "nano"

    lines = [
        "=" * 78,
        " Helios Model Family — Interactive Model Switcher (/model <tier>)".center(78),
        "=" * 78,
        f"  {'Tier':<10} | {'Parameters':<18} | {'Context':<14} | {'Q4 Size':<9} | Status",
        "-" * 78,
    ]
    for key, info in MODEL_FAMILY_CATALOG.items():
        status = "* ACTIVE" if key == active_key else "Available"
        lines.append(
            f"  {key:<10} | {info['params']:<18} | {info['context']:<14} | "
            f"{info['q4_size']:<9} | {status}"
        )
    lines.extend(
        [
            "-" * 78,
            "  Switch anytime in chat : /model nano | /model alpha | /model beta",
            "                           /model core | /model pro   | /model flagship",
            "  Or launch from terminal: helios chat -m <tier>  (e.g. helios chat -m pro)",
            "=" * 78,
        ]
    )
    return "\n".join(lines)


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

    effective_path = model_path
    if effective_path is None:
        # Check One Touch reasoning checkpoint first, then local pretrained checkpoint
        ext_ckpt = Path("/Volumes/One Touch/Helios/checkpoints/helios_reasoning_trained")
        default_ckpt = Path("checkpoints/helios_core_pretrained").resolve()
        if ext_ckpt.is_dir() and (ext_ckpt / "state.pt").exists():
            effective_path = ext_ckpt
        elif default_ckpt.is_dir() and (default_ckpt / "state.pt").exists():
            effective_path = default_ckpt

    if effective_path is not None:
        p = Path(effective_path).resolve()
        if p.is_dir() and (p / "manifest.json").exists() and model_path is not None:
            manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
            config_dict = manifest.get("config") or manifest.get("architecture", {})
            config = (
                HeliosConfig.from_dict(config_dict)
                if config_dict
                else HeliosConfig.from_preset(preset)
            )
        elif p.is_file() and p.name == "config.json" and model_path is not None:
            config = HeliosConfig.from_pretrained(p)
        else:
            config = HeliosConfig.from_preset(preset)
    else:
        config = HeliosConfig.from_preset(preset)

    model = HeliosForCausalLM(config).to(target_device)

    if effective_path is not None:
        p = Path(effective_path).resolve()
        ckpt_file = (p / "state.pt") if p.is_dir() else p
        if ckpt_file.is_file():
            payload = torch.load(ckpt_file, map_location=target_device, weights_only=False)
            state_dict = (
                payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            )
            if isinstance(state_dict, dict):
                model_sd = model.state_dict()
                compatible_sd = {
                    k: v
                    for k, v in state_dict.items()
                    if k in model_sd and model_sd[k].shape == v.shape
                }
                if len(compatible_sd) == len(model_sd):
                    model.load_state_dict(compatible_sd, strict=True)
                elif model_path is not None:
                    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, config


def load_or_create_tokenizer(
    tokenizer_path: str | Path | None = None,
    vocab_size: int = 32000,
) -> BaseTokenizer:
    """Load pretrained tokenizer or initialize ByteLevelBPETokenizer with padded vocab."""
    effective_path = tokenizer_path
    if effective_path is None:
        default_tok = Path("checkpoints/helios_core_pretrained").resolve()
        if (default_tok / "tokenizer.json").exists():
            effective_path = default_tok

    if effective_path is not None and Path(effective_path).exists():
        return HeliosTokenizer.from_pretrained(effective_path)

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
