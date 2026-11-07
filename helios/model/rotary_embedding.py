"""Rotary Position Embeddings (RoPE) for Helios."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    Splits the last dimension in half: [x1, x2] -> [-x2, x1].

    Args:
        x: Tensor of shape (..., dim) where dim is even.

    Returns:
        Tensor with rotated half dimensions.
    """
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to query or key tensor.

    Broadcasting Invariant:
        cos and sin must be broadcastable against x without requiring
        redundant tensor memory duplication. If cos/sin have shape
        (1, 1, seq_len, head_dim) or (seq_len, head_dim) or (..., seq_len, head_dim),
        they broadcast seamlessly over batch and head dimensions.

    Args:
        x: Input tensor of shape (batch, n_heads, seq_len, head_dim) or similar.
        cos: Cosine frequency table broadcastable to x.
        sin: Sine frequency table broadcastable to x.

    Returns:
        Rotated tensor of identical shape and dtype as x.
    """
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputed and dynamically extensible Rotary Position Embedding (RoPE) layer.

    Encodes relative position information by rotating query and key representations
    in the complex 2D plane.

    Args:
        dim: Dimensionality of each attention head (head_dim). Must be even.
        max_seq_len: Initial maximum context length cached.
        theta: Base frequency constant theta (default: 10000.0).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")

        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.theta: float = theta

        # Frequency bands: theta_i = 1.0 / (theta ** (2i / dim))
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute initial cache lazily (capped at 4096 on startup to avoid 256K RAM spike)
        self.max_seq_len_cached: int = 0
        init_cache_len = min(max_seq_len, 4096)
        self._set_cos_sin_cache(init_cache_len, device=torch.device("cpu"), dtype=torch.float32)

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Precompute cosine and sine tables up to seq_len with Dynamic NTK scaling."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        if seq_len > self.max_seq_len and self.dim > 2:
            # Dynamic NTK-Aware RoPE base scaling beyond nominal max_seq_len
            scale = seq_len / float(self.max_seq_len)
            scaled_theta = self.theta * (scale ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (
                scaled_theta
                ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
            )
        else:
            inv_freq = self.inv_freq.to(device)
        freqs = torch.outer(t, inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

        # Cache as (1, 1, seq_len, dim) for zero-copy broadcasting over (batch, heads, seq, dim)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve sliced cosine and sine tables for the input activation.

        Args:
            x: Input activation tensor (used to infer device, dtype, and seq_len if not given).
            seq_len: Number of tokens in the current sequence (optional).
            start_pos: Starting position index (for KV cache decoding steps).

        Returns:
            Tuple of (cos_slice, sin_slice) shaped (1, 1, seq_len, head_dim).
        """
        if seq_len is None:
            # Assuming x is (batch, n_heads, seq_len, head_dim) or (..., seq_len, head_dim)
            seq_len = x.shape[-2] if x.ndim >= 2 else self.max_seq_len

        required_len = start_pos + seq_len

        # Dynamic cache extension & device/dtype alignment
        device = x.device if x.numel() > 0 else self.cos_cached.device
        dtype = x.dtype if x.numel() > 0 and x.is_floating_point() else self.cos_cached.dtype

        if (
            required_len > self.max_seq_len_cached
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        ):
            new_cache_len = max(required_len, self.max_seq_len_cached * 2)
            self._set_cos_sin_cache(new_cache_len, device=device, dtype=dtype)

        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :]
        return cos, sin

    def extra_repr(self) -> str:
        """Extra representation string for module display."""
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"
