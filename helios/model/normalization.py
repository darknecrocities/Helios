"""Root Mean Square Layer Normalization (RMSNorm) for Helios."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with Float32 Numerical Precision Invariant.

    Normalizes activations across the final dimension without mean centering:
        y = (x / sqrt(mean(x^2) + eps)) * weight

    Numerical Precision Invariant:
        Computes variance and normalization strictly in float32 regardless of input
        dtype (e.g. float16, bfloat16) to prevent overflow or underflow, then casts
        the normalized result back to the original input dtype.

    Args:
        dim: Dimensionality of the input feature vector (e.g. d_model).
        eps: Small numerical constant to prevent division by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))

    def _norm(self, x_fp32: torch.Tensor) -> torch.Tensor:
        """Compute normalization in float32."""
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return x_fp32 * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying RMS normalization.

        Args:
            x: Input tensor of shape (..., dim) in any floating-point dtype.

        Returns:
            Normalized tensor of identical shape and dtype as input x.
        """
        input_dtype = x.dtype
        # Float32 numerical precision invariant
        x_fp32 = x.float()
        normed_fp32 = self._norm(x_fp32)
        # Cast back to input_dtype and scale by learnable weight
        return normed_fp32.to(input_dtype) * self.weight

    def reset_parameters(self) -> None:
        """Reset learnable weight to ones."""
        nn.init.ones_(self.weight)

    def extra_repr(self) -> str:
        """Extra representation string for module display."""
        return f"{self.weight.shape[0]}, eps={self.eps}"
