"""Helios Foundation Model Architecture Package."""

from helios.model.attention import Attention, GroupedQueryAttention, KVCache, repeat_kv
from helios.model.config import HeliosConfig
from helios.model.feed_forward import FeedForward, calculate_d_ffn
from helios.model.normalization import RMSNorm
from helios.model.parameter_inspector import (
    LayerBreakdown,
    ParameterInspectionReport,
    calculate_theoretical_parameters,
    count_parameters,
    inspect_parameters,
)
from helios.model.rotary_embedding import RotaryEmbedding, apply_rotary_emb, rotate_half
from helios.model.transformer import HeliosForCausalLM
from helios.model.transformer_block import HeliosBlock

__all__ = [
    "HeliosConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "rotate_half",
    "Attention",
    "GroupedQueryAttention",
    "KVCache",
    "repeat_kv",
    "FeedForward",
    "calculate_d_ffn",
    "HeliosBlock",
    "HeliosForCausalLM",
    "LayerBreakdown",
    "ParameterInspectionReport",
    "inspect_parameters",
    "count_parameters",
    "calculate_theoretical_parameters",
]
