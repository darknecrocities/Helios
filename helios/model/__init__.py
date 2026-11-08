"""Helios Foundation Model Architecture Package."""

from helios.model.attention import Attention, GroupedQueryAttention, KVCache, repeat_kv
from helios.model.config import HeliosConfig
from helios.model.feed_forward import FeedForward, calculate_d_ffn
from helios.model.multimodal import (
    CausalLMOutputWithPast,
    HeliosForMultimodalLM,
    HeliosMultiModalProjector,
    HeliosVisionBlock,
    HeliosVisionConfig,
    HeliosVisionEncoder,
    build_multimodal_attention_mask,
)
from helios.model.normalization import RMSNorm
from helios.model.parameter_inspector import (
    LayerBreakdown,
    ParameterInspectionReport,
    ParameterInspector,
    calculate_theoretical_parameters,
    count_parameters,
    inspect_parameters,
)
from helios.model.rotary_embedding import RotaryEmbedding, apply_rotary_emb, rotate_half
from helios.model.speculative import (
    SpeculativeConfig,
    SpeculativeDecodingEngine,
    SpeculativeGenerationOutput,
    SpeculativeMetrics,
)
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
    "ParameterInspector",
    "inspect_parameters",
    "count_parameters",
    "calculate_theoretical_parameters",
    "HeliosVisionConfig",
    "HeliosVisionBlock",
    "HeliosVisionEncoder",
    "HeliosMultiModalProjector",
    "HeliosForMultimodalLM",
    "CausalLMOutputWithPast",
    "build_multimodal_attention_mask",
    "SpeculativeConfig",
    "SpeculativeMetrics",
    "SpeculativeGenerationOutput",
    "SpeculativeDecodingEngine",
]
