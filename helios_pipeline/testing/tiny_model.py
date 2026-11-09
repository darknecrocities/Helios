"""Tiny Helios model factory for fast, deterministic unit and integration tests."""

from __future__ import annotations

import torch

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM


def create_tiny_helios_config(vocab_size: int = 512) -> HeliosConfig:
    """Create a minimal <100K parameter HeliosConfig suitable for fast testing."""
    return HeliosConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )


def create_tiny_helios_model(
    vocab_size: int = 512,
    device: str = "cpu",
) -> HeliosForCausalLM:
    """Instantiate a minimal Helios transformer model for unit tests."""
    cfg = create_tiny_helios_config(vocab_size=vocab_size)
    model = HeliosForCausalLM(cfg)
    model.to(torch.device(device))
    model.eval()
    return model
