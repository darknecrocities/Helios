"""Unit tests for Helios Multimodal Vision-Language Architecture (Feature 45).

Tests ViT patch projection, dynamic 2D bicubic position interpolation,
projector dimension mapping, 2D prefix attention mask invariants,
token sequence merging, and full multimodal forward/backward gradient flow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from helios.model.config import HeliosConfig
from helios.model.multimodal import (
    IMAGE_TOKEN_ID,
    CausalLMOutputWithPast,
    HeliosForMultimodalLM,
    HeliosMultiModalProjector,
    HeliosVisionConfig,
    HeliosVisionEncoder,
    build_multimodal_attention_mask,
)


@pytest.fixture
def tiny_vision_config() -> HeliosVisionConfig:
    return HeliosVisionConfig(
        image_size=28,
        patch_size=14,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )


@pytest.fixture
def tiny_llm_config() -> HeliosConfig:
    return HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=64,
    )


def test_vision_encoder_output_shapes() -> None:
    """Verify ViT patch encoder produces correct token counts for 224x224 and 336x336."""
    config = HeliosVisionConfig(
        image_size=224,
        patch_size=14,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
    )
    encoder = HeliosVisionEncoder(config)

    # 1. Standard resolution 224x224 -> (224/14)^2 = 256 tokens
    x_224 = torch.randn(2, 3, 224, 224)
    out_224 = encoder(x_224)
    assert out_224.shape == (2, 256, 128)
    assert not torch.isnan(out_224).any()

    # 2. High resolution 336x336 -> (336/14)^2 = 576 tokens via 2D bicubic interpolation
    x_336 = torch.randn(1, 3, 336, 336)
    out_336 = encoder(x_336)
    assert out_336.shape == (1, 576, 128)
    assert not torch.isnan(out_336).any()


def test_positional_embedding_bicubic_interpolation() -> None:
    """Verify dynamic 2D bicubic interpolation preserves spatial aspect and continuity."""
    config = HeliosVisionConfig(image_size=28, patch_size=14, hidden_size=32)
    encoder = HeliosVisionEncoder(config)
    assert encoder.position_embeddings.shape == (1, 4, 32)

    # Interpolate to 42x42 -> 3x3 = 9 patches
    interp = encoder.interpolate_pos_encoding(42, 42)
    assert interp.shape == (1, 9, 32)
    assert not torch.isnan(interp).any()


def test_multimodal_projector_dimension_mapping() -> None:
    """Verify 2-layer GELU MLP maps visual latent dimension to LLM hidden dimension."""
    # Core tier mapping: 1024 -> 2048
    core_proj = HeliosMultiModalProjector(vision_hidden_size=1024, llm_d_model=2048)
    v_in = torch.randn(2, 16, 1024)
    out_core = core_proj(v_in)
    assert out_core.shape == (2, 16, 2048)
    assert not torch.isnan(out_core).any()

    # Flagship tier mapping: 1024 -> 4096
    flag_proj = HeliosMultiModalProjector(vision_hidden_size=1024, llm_d_model=4096)
    out_flag = flag_proj(v_in)
    assert out_flag.shape == (2, 16, 4096)
    assert not torch.isnan(out_flag).any()

    # Non-linear activation verification
    zero_out = core_proj(torch.zeros_like(v_in))
    assert not torch.allclose(out_core, zero_out)


def test_multimodal_attention_mask_invariants() -> None:
    """Verify 2D prefix multimodal mask: bidirectional within image, causal for text."""
    seq_len = 8
    img_start = 2
    img_end = 5
    mask = build_multimodal_attention_mask(seq_len, img_start, img_end)

    # Property 1: All visual tokens attend to each other bidirectionally (0.0)
    for q in range(img_start, img_end):
        for k in range(img_start, img_end):
            assert mask[q, k] == 0.0, f"Visual token ({q}, {k}) should be 0.0, got {mask[q, k]}"

    # Property 2: Visual tokens attend causally to preceding text tokens (k < img_start)
    for q in range(img_start, img_end):
        for k in range(img_start):
            assert mask[q, k] == 0.0, f"Visual token {q} should attend to text token {k}"

    # Property 3: Visual tokens CANNOT attend to subsequent response text tokens (k >= img_end)
    for q in range(img_start, img_end):
        for k in range(img_end, seq_len):
            assert mask[q, k] == float("-inf"), (
                f"Visual token {q} must not attend to future token {k}"
            )

    # Property 4: Subsequent text tokens attend causally to all past tokens including vision
    for q in range(img_end, seq_len):
        for k in range(q + 1):
            assert mask[q, k] == 0.0, f"Text token {q} should attend to token {k}"
        for k in range(q + 1, seq_len):
            assert mask[q, k] == float("-inf"), (
                f"Text token {q} must not attend to future token {k}"
            )


def test_token_sequence_merging_case_a(
    tiny_llm_config: HeliosConfig,
    tiny_vision_config: HeliosVisionConfig,
) -> None:
    """Test Case A: Input sequence already contains N placeholder tokens."""
    model = HeliosForMultimodalLM(config=tiny_llm_config, vision_config=tiny_vision_config)
    n_patches = (tiny_vision_config.image_size // tiny_vision_config.patch_size) ** 2  # 4
    assert n_patches == 4

    input_ids = torch.tensor(
        [[10, 11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12]]
    )
    pixels = torch.randn(1, 3, tiny_vision_config.image_size, tiny_vision_config.image_size)

    out = model(input_ids=input_ids, pixel_values=pixels)
    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits.shape == (1, 7, tiny_llm_config.vocab_size)
    assert not torch.isnan(out.logits).any()


def test_token_sequence_merging_case_b(
    tiny_llm_config: HeliosConfig,
    tiny_vision_config: HeliosVisionConfig,
) -> None:
    """Test Case B: Single placeholder token spliced into N visual tokens."""
    model = HeliosForMultimodalLM(config=tiny_llm_config, vision_config=tiny_vision_config)
    n_patches = (tiny_vision_config.image_size // tiny_vision_config.patch_size) ** 2  # 4
    assert n_patches == 4

    # Text: [1, 2, IMAGE_TOKEN_ID, 3] -> expands to [1, 2, vis0, vis1, vis2, vis3, 3] (len 7)
    input_ids = torch.tensor([[1, 2, IMAGE_TOKEN_ID, 3]])
    labels = torch.tensor([[1, 2, -100, 3]])
    pixels = torch.randn(1, 3, tiny_vision_config.image_size, tiny_vision_config.image_size)

    out = model(input_ids=input_ids, pixel_values=pixels, labels=labels)
    assert out.logits.shape == (1, 7, tiny_llm_config.vocab_size)
    assert out.loss is not None
    assert not torch.isnan(out.loss)


def test_multimodal_forward_and_backward_gradient_flow(
    tiny_llm_config: HeliosConfig,
    tiny_vision_config: HeliosVisionConfig,
) -> None:
    """Verify gradients backpropagate end-to-end through LLM, projector, and vision encoder."""
    model = HeliosForMultimodalLM(config=tiny_llm_config, vision_config=tiny_vision_config)

    input_ids = torch.tensor([[5, IMAGE_TOKEN_ID, 6, 7]])
    labels = torch.tensor([[5, -100, 6, 7]])
    pixels = torch.randn(1, 3, tiny_vision_config.image_size, tiny_vision_config.image_size)

    out = model(input_ids=input_ids, pixel_values=pixels, labels=labels)
    loss = out.loss
    assert loss is not None
    loss.backward()

    # 1. Check vision encoder gradients
    has_vision_grad = any(
        p.grad is not None and torch.norm(p.grad) > 0.0 for p in model.vision_encoder.parameters()
    )
    assert has_vision_grad, "Vision encoder received zero gradient"

    # 2. Check projector gradients
    has_proj_grad = any(
        p.grad is not None and torch.norm(p.grad) > 0.0 for p in model.projector.parameters()
    )
    assert has_proj_grad, "Projector received zero gradient"

    # 3. Check language model gradients
    has_llm_grad = any(
        p.grad is not None and torch.norm(p.grad) > 0.0 for p in model.language_model.parameters()
    )
    assert has_llm_grad, "Language model received zero gradient"


def test_text_only_forward_fallback(
    tiny_llm_config: HeliosConfig,
    tiny_vision_config: HeliosVisionConfig,
) -> None:
    """Verify HeliosForMultimodalLM executes text-only forward pass cleanly when pixel_values is None."""
    model = HeliosForMultimodalLM(config=tiny_llm_config, vision_config=tiny_vision_config)
    input_ids = torch.tensor([[10, 20, 30, 40]])

    out = model(input_ids=input_ids, pixel_values=None)
    assert out.logits.shape == (1, 4, tiny_llm_config.vocab_size)
    assert not torch.isnan(out.logits).any()


def test_physical_line_limit_compliance() -> None:
    """Verify helios/model/multimodal.py adheres to strict 500-line ceiling."""
    source_path = Path(__file__).parent.parent.parent / "helios" / "model" / "multimodal.py"
    assert source_path.exists(), f"Missing file: {source_path}"
    with open(source_path, "r", encoding="utf-8") as f:
        line_count = len(f.readlines())
    assert line_count <= 500, f"Physical line ceiling breached: {line_count} > 500 lines"
