"""Helios Transformer Decoder Block.

Combines Pre-RMSNorm normalization, Grouped-Query Attention (with RoPE
and KVCache support), and SwiGLU Feed-Forward network with residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from helios.model.attention import Attention, KVCache
from helios.model.config import HeliosConfig
from helios.model.feed_forward import FeedForward
from helios.model.normalization import RMSNorm


class HeliosBlock(nn.Module):
    """A single decoder-only transformer block for the Helios architecture.

    Data Flow:
      1. h = x + Attention(RMSNorm_1(x), start_pos, kv_cache)
      2. out = h + FeedForward(RMSNorm_2(h))
    """

    def __init__(self, config: HeliosConfig) -> None:
        super().__init__()
        self.config = config

        self.input_layernorm = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps,
        )
        self.attention = Attention(config=config)
        self.post_attention_layernorm = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps,
        )
        self.feed_forward = FeedForward(config=config)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Execute forward pass for the transformer block.

        Args:
            x: Input activations of shape (batch_size, seq_len, d_model).
            start_pos: Starting position index for rotary embeddings and KV cache cursor.
            kv_cache: Optional stateful KVCache for incremental autoregressive decoding.

        Returns:
            Output activations of shape (batch_size, seq_len, d_model).
        """
        # Attention sublayer with Pre-RMSNorm and residual connection
        norm_x = self.input_layernorm(x)
        attn_out = self.attention(norm_x, start_pos=start_pos, kv_cache=kv_cache)
        h = x + attn_out

        # FeedForward sublayer with Pre-RMSNorm and residual connection
        norm_h = self.post_attention_layernorm(h)
        ffn_out = self.feed_forward(norm_h)
        return h + ffn_out
