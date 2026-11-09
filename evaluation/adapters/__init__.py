"""Model adapters for native and external benchmark integration."""

from __future__ import annotations

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.adapters.pytorch_adapter import NativePyTorchAdapter

__all__ = ["HeliosModelAdapter", "NativePyTorchAdapter"]
