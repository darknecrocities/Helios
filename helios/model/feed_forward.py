"""SwiGLU Gated Feed-Forward Network (FFN) for Helios."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from helios.model.config import HeliosConfig


def calculate_d_ffn(d_model: int, multiple_of: int = 64) -> int:
    """Calculate SwiGLU hidden dimension with parameter parity and hardware alignment.

    Formula:
        d_ffn = int(2/3 * 4 * d_model) rounded up to nearest multiple of multiple_of.

    Args:
        d_model: Model hidden embedding dimension.
        multiple_of: Tile alignment multiple (default 64 for tensor cores).

    Returns:
        Aligned intermediate feed-forward dimension.
    """
    raw_dim = int(2 * (4 * d_model) / 3)
    return multiple_of * ((raw_dim + multiple_of - 1) // multiple_of)


class FeedForward(nn.Module):
    """SwiGLU Gated Feed-Forward Network.

    Computes:
        SwiGLU(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    Features:
        - Strictly bias-free linear projections.
        - Optimal expansion ratio d_ffn ≈ (8/3) * d_model.
        - Hardware tensor core tile alignment.
        - Optional dropout regularization.
    """

    def __init__(
        self,
        d_model: int | None = None,
        d_ffn: int | None = None,
        dropout: float = 0.0,
        multiple_of: int = 64,
        *,
        config: HeliosConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = config.d_model
            d_ffn = getattr(config, "d_ffn", getattr(config, "d_ff", None))
            dropout = getattr(config, "dropout", 0.0)

        if d_model is None:
            raise ValueError("Either config or d_model must be provided.")

        if d_ffn is None:
            d_ffn = calculate_d_ffn(d_model, multiple_of=multiple_of)

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout_rate = dropout

        # Bias-free linear projections
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @classmethod
    def from_config(cls, config: HeliosConfig) -> FeedForward:
        """Construct FeedForward layer from a HeliosConfig instance."""
        return cls(config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SwiGLU Feed-Forward Network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(gated)
        return self.dropout(out)
