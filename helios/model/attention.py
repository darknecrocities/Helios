"""Grouped-Query Attention (GQA) and Pre-Allocated KV Cache for Helios.

Implements memory-bandwidth optimized attention with 3:1 GQA head ratio,
pre-allocated contiguous fixed buffers, and O(1) in-place caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from helios.model.rotary_embedding import RotaryEmbedding, apply_rotary_emb

if TYPE_CHECKING:
    from helios.model.config import HeliosConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand Key/Value heads to match Query heads in Grouped-Query Attention.

    Args:
        x: Key or Value tensor of shape (batch_size, n_kv_heads, seq_len, head_dim).
        n_rep: Repetition factor (n_heads // n_kv_heads).

    Returns:
        Expanded tensor of shape (batch_size, n_heads, seq_len, head_dim).
    """
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
    )


class KVCache:
    """Pre-allocated contiguous Key-Value Cache for autoregressive generation.

    Eliminates O(T^2) dynamic heap allocations and memory fragmentation by
    pre-allocating contiguous fixed memory buffers of shape
    (max_batch_size, n_kv_heads, max_seq_len, head_dim). Updates execute
    in O(1) time via in-place tensor slice mutation.
    """

    def __init__(
        self,
        max_batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.max_batch_size = max_batch_size
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        self.k_cache = torch.zeros(
            (max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.seq_len: int = 0

    @property
    def cur_len(self) -> int:
        """Current sequence length cursor."""
        return self.seq_len

    @cur_len.setter
    def cur_len(self, value: int) -> None:
        self.seq_len = value

    def update(
        self,
        start_pos: int,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert new key/value representations and return active context view.

        Args:
            start_pos: Starting sequence position for the current slice.
            k_val: New key tensor of shape (batch_size, n_kv_heads, seq_len, head_dim).
            v_val: New value tensor of shape (batch_size, n_kv_heads, seq_len, head_dim).

        Returns:
            Tuple of active key and value tensor views up to (start_pos + seq_len).
        """
        bsz, _n_kv, seq_len, _h_dim = k_val.shape
        if bsz > self.max_batch_size:
            raise ValueError(
                f"Batch size {bsz} exceeds pre-allocated cache max_batch_size {self.max_batch_size}"
            )
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"Position {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Autonomous device & dtype alignment
        if self.k_cache.device != k_val.device or self.k_cache.dtype != k_val.dtype:
            self.k_cache = self.k_cache.to(device=k_val.device, dtype=k_val.dtype)
            self.v_cache = self.v_cache.to(device=v_val.device, dtype=v_val.dtype)
            self.device = k_val.device
            self.dtype = k_val.dtype

        # O(1) in-place slice mutation
        self.k_cache[:bsz, :, start_pos : start_pos + seq_len, :] = k_val
        self.v_cache[:bsz, :, start_pos : start_pos + seq_len, :] = v_val
        self.seq_len = max(self.seq_len, start_pos + seq_len)

        # Return view of valid historical context
        return (
            self.k_cache[:bsz, :, : start_pos + seq_len, :],
            self.v_cache[:bsz, :, : start_pos + seq_len, :],
        )

    def reset(self) -> None:
        """Reset sequence position cursor to zero."""
        self.seq_len = 0

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Migrate internal cache buffers to specified device or dtype."""
        if device is not None:
            self.device = torch.device(device)
            self.k_cache = self.k_cache.to(device=self.device)
            self.v_cache = self.v_cache.to(device=self.device)
        if dtype is not None:
            self.dtype = dtype
            self.k_cache = self.k_cache.to(dtype=self.dtype)
            self.v_cache = self.v_cache.to(dtype=self.dtype)
        return self

    def memory_footprint_mb(self) -> float:
        """Calculate total memory allocated by cache buffers in megabytes."""
        element_size = self.k_cache.element_size()
        total_elements = self.k_cache.numel() + self.v_cache.numel()
        return (total_elements * element_size) / (1024 * 1024)


class Attention(nn.Module):
    """Grouped-Query Attention (GQA) layer with integrated RoPE and KV Cache.

    Computes multi-head attention where n_heads query heads share n_kv_heads
    key/value heads to minimize memory bandwidth bottlenecks.
    """

    def __init__(
        self,
        d_model: int | None = None,
        n_heads: int | None = None,
        n_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        *,
        config: HeliosConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = config.d_model
            n_heads = config.n_heads
            n_kv_heads = config.n_kv_heads
            max_seq_len = getattr(config, "max_seq_len", 2048)
            dropout = getattr(config, "dropout", 0.0)
            rope_theta = getattr(config, "rope_theta", getattr(config, "rope_base", 10000.0))

        if d_model is None or n_heads is None or n_kv_heads is None:
            raise ValueError("Either config or (d_model, n_heads, n_kv_heads) must be specified.")

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout

        # Bias-free linear projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

    @classmethod
    def from_config(cls, config: HeliosConfig) -> Attention:
        """Construct Attention layer from a HeliosConfig instance."""
        return cls(config=config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of Grouped-Query Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            cos: Optional precomputed cosine frequencies for RoPE.
            sin: Optional precomputed sine frequencies for RoPE.
            start_pos: Starting sequence index (offset for autoregressive decoding).
            kv_cache: Optional stateful KVCache buffer.
            attn_mask: Optional custom attention mask tensor.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        bsz, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if cos is None or sin is None:
            cos, sin = self.rotary_emb(q, seq_len=seq_len, start_pos=start_pos)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Ingest into pre-allocated contiguous KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(start_pos, k, v)

        # Broadcast KV heads to match query heads
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        total_k_len = k.shape[2]
        dropout_p = self.dropout_rate if self.training else 0.0

        # Masked attention dispatch
        if seq_len == 1 and kv_cache is not None and attn_mask is None:
            # Single-token decode fast-path: Query at position start_pos attends to all past keys
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
        elif start_pos == 0 and attn_mask is None:
            # Standard prefill / full-sequence causal forward pass
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # Chunked prefill or custom attention mask
            if attn_mask is None:
                q_idx = torch.arange(seq_len, device=x.device)[:, None] + start_pos
                k_idx = torch.arange(total_k_len, device=x.device)[None, :]
                attn_mask = torch.where(
                    k_idx <= q_idx,
                    torch.zeros(1, device=x.device, dtype=x.dtype),
                    torch.full((1,), float("-inf"), device=x.device, dtype=x.dtype),
                )
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            )

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)


# Alias for explicit contract alignment
GroupedQueryAttention = Attention
