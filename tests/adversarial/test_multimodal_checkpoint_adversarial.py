"""Adversarial Verification Suite for Multimodal Architecture & Checkpoint Continuity.

Empirically stress-tests:
1. Dynamic resolution scaling (224x224, 336x336, non-square, arbitrary aspect ratios).
2. Gradient backpropagation through vision encoder, projector, and language backbone.
3. 2D prefix multimodal attention mask invariants, boundary conditions, and causal barriers.
4. Out-of-vocab placeholder token clamping, splicing, prepending, and image_mask routing.
5. Extreme loss spike detection (exact 3.0x threshold, nan, inf, zero/negative baseline).
6. Checkpoint corruption defense (tampered payload, corrupted bytes, NaN weights).
7. All 6 model tiers (Nano through Flagship) tensor dimensions and parameter alignment.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from helios.model.config import HeliosConfig
from helios.model.multimodal import (
    IMAGE_TOKEN_ID,
    HeliosForMultimodalLM,
    HeliosVisionConfig,
    HeliosVisionEncoder,
    build_multimodal_attention_mask,
)
from helios.model.parameter_inspector import calculate_theoretical_parameters
from helios.model.transformer import HeliosForCausalLM
from helios.training.checkpoint import (
    CheckpointCorruptionError,
    CheckpointManager,
    detect_loss_spike,
)
from helios.training.controller import (
    load_multi_tier_checkpoint,
    resume_training_from_checkpoint,
)


@pytest.fixture
def compact_vision_config() -> HeliosVisionConfig:
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
def compact_llm_config() -> HeliosConfig:
    return HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=128,
    )


def test_adversarial_dynamic_resolution_scaling_and_interpolation() -> None:
    """Stress-test dynamic 2D bicubic interpolation on square, non-square, and arbitrary sizes."""
    config = HeliosVisionConfig(
        image_size=224,
        patch_size=14,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    encoder = HeliosVisionEncoder(config)

    # Resolution test matrix: (Height, Width, Expected Patches)
    test_matrix = [
        (224, 224, 256),  # Standard square
        (336, 336, 576),  # High-res square
        (168, 336, 288),  # Landscape wide 1:2
        (336, 224, 384),  # Portrait tall 3:2
        (224, 336, 384),  # Landscape wide 2:3
        (112, 280, 160),  # Extreme non-square 2:5
    ]

    for h, w, exp_patches in test_matrix:
        pixels = torch.randn(2, 3, h, w)
        features = encoder(pixels)
        assert features.shape == (2, exp_patches, 64)
        assert not torch.isnan(features).any(), f"NaN in features for ({h}, {w})"
        assert not torch.isinf(features).any(), f"Inf in features for ({h}, {w})"

        # Verify positional embedding interpolation directly
        pos = encoder.interpolate_pos_encoding(h, w)
        assert pos.shape == (1, exp_patches, 64)
        assert not torch.isnan(pos).any()
        assert not torch.isinf(pos).any()


def test_adversarial_multimodal_gradient_flow_and_freezing(
    compact_llm_config: HeliosConfig,
    compact_vision_config: HeliosVisionConfig,
) -> None:
    """Stress-test full gradient flow and freeze isolation across all 3 sub-architectures."""
    model = HeliosForMultimodalLM(
        config=compact_llm_config, vision_config=compact_vision_config
    )

    # 1. Full end-to-end backprop with loss on prompt and response tokens
    input_ids = torch.tensor([[10, IMAGE_TOKEN_ID, 20, 30]])
    labels = torch.tensor([[10, -100, 20, 30]])
    pixels = torch.randn(1, 3, 28, 28)

    out = model(input_ids=input_ids, pixel_values=pixels, labels=labels)
    assert out.loss is not None
    out.loss.backward()

    # Check non-zero gradient in all three submodules
    vision_grads = [p.grad for p in model.vision_encoder.parameters() if p.grad is not None]
    proj_grads = [p.grad for p in model.projector.parameters() if p.grad is not None]
    llm_grads = [p.grad for p in model.language_model.parameters() if p.grad is not None]

    assert len(vision_grads) == 30 and all(torch.norm(g) > 0 for g in vision_grads)
    assert len(proj_grads) == 4 and all(torch.norm(g) > 0 for g in proj_grads)
    assert len(llm_grads) == 20 and any(torch.norm(g) > 0 for g in llm_grads)

    # 2. Frozen vision encoder stage: only projector and LLM should get gradients
    model.zero_grad(set_to_none=True)
    for p in model.vision_encoder.parameters():
        p.requires_grad = False

    out2 = model(input_ids=input_ids, pixel_values=pixels, labels=labels)
    out2.loss.backward()

    assert all(p.grad is None for p in model.vision_encoder.parameters())
    proj_grads2 = [p.grad for p in model.projector.parameters() if p.grad is not None]
    assert len(proj_grads2) == 4 and all(torch.norm(g) > 0 for g in proj_grads2)


def test_adversarial_multimodal_attention_mask_properties_and_boundaries() -> None:
    """Stress-test 2D prefix attention mask properties, causal barriers, and boundary edges."""
    seq_len = 10
    v_start, v_end = 3, 7
    mask = build_multimodal_attention_mask(seq_len, v_start, v_end)

    # 1. Bidirectional visual token self-attention: (q, k) in [3, 7) must be 0.0
    for q in range(v_start, v_end):
        for k in range(v_start, v_end):
            assert mask[q, k] == 0.0, f"Visual token ({q}, {k}) must be 0.0"

    # 2. Conditioning: visual tokens attend to preceding text tokens (k < 3)
    for q in range(v_start, v_end):
        for k in range(v_start):
            assert mask[q, k] == 0.0, f"Visual token {q} should see prior token {k}"

    # 3. Causal barrier: visual tokens CANNOT see subsequent text tokens (k >= 7)
    for q in range(v_start, v_end):
        for k in range(v_end, seq_len):
            assert mask[q, k] == float("-inf"), f"Visual token {q} leaked future token {k}"

    # 4. Subsequent text tokens attend causally to all past tokens including vision
    for q in range(v_end, seq_len):
        for k in range(q + 1):
            assert mask[q, k] == 0.0
        for k in range(q + 1, seq_len):
            assert mask[q, k] == float("-inf")

    # 5. Boundary Condition: Image at sequence start (0, 4)
    m_start = build_multimodal_attention_mask(seq_len, 0, 4)
    assert m_start[0, 3] == 0.0 and m_start[3, 0] == 0.0
    assert m_start[0, 4] == float("-inf")
    assert m_start[4, 0] == 0.0

    # 6. Boundary Condition: Image at sequence end (6, 10)
    m_end = build_multimodal_attention_mask(seq_len, 6, 10)
    assert m_end[6, 9] == 0.0 and m_end[9, 6] == 0.0
    assert m_end[2, 6] == float("-inf")

    # 7. Invalid boundaries safely fallback to standard causal lower-triangular mask
    causal_ref = torch.where(
        torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :],
        0.0,
        float("-inf"),
    )
    assert torch.equal(build_multimodal_attention_mask(seq_len, 6, 3), causal_ref)
    assert torch.equal(build_multimodal_attention_mask(seq_len, -1, 4), causal_ref)
    assert torch.equal(build_multimodal_attention_mask(seq_len, 2, 15), causal_ref)


def test_adversarial_placeholder_token_handling_and_clamping(
    compact_llm_config: HeliosConfig,
    compact_vision_config: HeliosVisionConfig,
) -> None:
    """Stress-test out-of-vocab placeholder token handling, splicing, prepending, and image_mask."""
    model = HeliosForMultimodalLM(
        config=compact_llm_config, vision_config=compact_vision_config
    )
    n_patches = (compact_vision_config.image_size // compact_vision_config.patch_size) ** 2  # 4
    pixels = torch.randn(1, 3, compact_vision_config.image_size, compact_vision_config.image_size)

    # 1. Out-of-vocab placeholder token ID (IMAGE_TOKEN_ID=32001 >> vocab_size=128)
    assert IMAGE_TOKEN_ID >= compact_llm_config.vocab_size
    # Case B: Single placeholder spliced into 4 visual tokens: [1, 2, IMG, 3] -> len 7
    input_b = torch.tensor([[1, 2, IMAGE_TOKEN_ID, 3]])
    out_b = model(input_ids=input_b, pixel_values=pixels)
    assert out_b.logits.shape == (1, 7, compact_llm_config.vocab_size)
    assert not torch.isnan(out_b.logits).any()

    # 2. Case A: Exact N placeholder tokens matching n_vis (4 placeholders)
    input_a = torch.tensor([[5, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 6]])
    out_a = model(input_ids=input_a, pixel_values=pixels)
    assert out_a.logits.shape == (1, 6, compact_llm_config.vocab_size)
    assert not torch.isnan(out_a.logits).any()

    # 3. Prepend fallback: zero placeholder tokens provided
    input_c = torch.tensor([[5, 6, 7]])
    out_c = model(input_ids=input_c, pixel_values=pixels)
    assert out_c.logits.shape == (1, 3 + n_patches, compact_llm_config.vocab_size)
    assert not torch.isnan(out_c.logits).any()

    # 4. Explicit image_mask routing
    input_d = torch.tensor([[10, 0, 0, 0, 0, 20]])
    img_mask = torch.tensor([[False, True, True, True, True, False]])
    out_d = model(input_ids=input_d, pixel_values=pixels, image_mask=img_mask)
    assert out_d.logits.shape == (1, 6, compact_llm_config.vocab_size)
    assert not torch.isnan(out_d.logits).any()

    # 5. Text-only execution fallback (pixel_values is None)
    out_text = model(input_ids=torch.tensor([[10, 20, 30]]), pixel_values=None)
    assert out_text.logits.shape == (1, 3, compact_llm_config.vocab_size)
    assert not torch.isnan(out_text.logits).any()


def test_adversarial_loss_spike_detection_extremes() -> None:
    """Stress-test loss spike detector across numerical boundaries, anomalies, and edge baselines."""
    baseline = 2.0

    # Boundary tests around 3.0x threshold
    assert not detect_loss_spike(loss=5.999, baseline_loss=baseline, threshold=3.0)
    assert detect_loss_spike(loss=6.000, baseline_loss=baseline, threshold=3.0)
    assert detect_loss_spike(loss=6.001, baseline_loss=baseline, threshold=3.0)

    # Extreme sudden loss surges
    assert detect_loss_spike(loss=100.0, baseline_loss=baseline, threshold=3.0)
    assert detect_loss_spike(loss=1e6, baseline_loss=baseline, threshold=3.0)

    # Numerical anomalies (NaN and Inf) must ALWAYS flag
    assert detect_loss_spike(loss=float("nan"), baseline_loss=baseline)
    assert detect_loss_spike(loss=float("inf"), baseline_loss=baseline)
    assert detect_loss_spike(loss=float("-inf"), baseline_loss=baseline)

    # Edge cases: zero, negative, or NaN baselines must not falsely trigger on finite loss
    assert not detect_loss_spike(loss=3.0, baseline_loss=0.0)
    assert not detect_loss_spike(loss=3.0, baseline_loss=-5.0)
    assert not detect_loss_spike(loss=3.0, baseline_loss=float("nan"))
    assert not detect_loss_spike(loss=3.0, baseline_loss=None)  # type: ignore[arg-type]


def test_adversarial_checkpoint_corruption_defense_and_resumption(
    tmp_path: Path,
    compact_llm_config: HeliosConfig,
) -> None:
    """Stress-test checkpoint corruption detection (tampering, garbage, NaNs) and 6-state resumption."""
    model = HeliosForCausalLM(compact_llm_config)
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = LambdaLR(opt, lr_lambda=lambda s: 0.9**s)

    # Run step to generate active optimizer states
    dummy = torch.randint(0, 128, (1, 4))
    model(dummy, labels=dummy)["loss"].backward()
    opt.step()
    sched.step()

    ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path, keep_last_n=2)
    saved_dir = ckpt_mgr.save_checkpoint(
        step=10,
        model=model,
        optimizer=opt,
        scheduler=sched,
        loss=1.23,
        dataloader_cursor={"shard": 1, "offset": 500},
        metrics={"step": 10, "loss": 1.23},
    )

    # 1. Clean resumption verification
    new_model = HeliosForCausalLM(compact_llm_config)
    new_opt = AdamW(new_model.parameters(), lr=1e-3)
    new_sched = LambdaLR(new_opt, lr_lambda=lambda s: 0.9**s)

    meta = resume_training_from_checkpoint(
        checkpoint_path=saved_dir,
        model=new_model,
        optimizer=new_opt,
        scheduler=new_sched,
        restore_rng=True,
    )
    assert meta["step"] == 10
    assert meta["dataloader_cursor"]["shard"] == 1

    # 2. Byte tampering defense (SHA-256 mismatch)
    corrupted_dir = tmp_path / "corrupted_ckpt"
    import shutil

    shutil.copytree(saved_dir, corrupted_dir)
    state_file = corrupted_dir / "state.pt"
    with open(state_file, "ab") as f:
        f.write(b"TAMPERED_BYTES")

    with pytest.raises(CheckpointCorruptionError, match="SHA-256 mismatch"):
        load_multi_tier_checkpoint(corrupted_dir)

    # 3. Unreadable garbage payload defense
    garbage_dir = tmp_path / "garbage_ckpt"
    garbage_dir.mkdir()
    (garbage_dir / "state.pt").write_bytes(b"\x00\x01\x02\x03GARBAGE_PAYLOAD")
    with pytest.raises(CheckpointCorruptionError, match="Corrupted checkpoint payload"):
        load_multi_tier_checkpoint(garbage_dir)

    # 4. NaN parameter detection defense
    nan_dir = tmp_path / "nan_ckpt"
    nan_dir.mkdir()
    torch.save(
        {"model_state_dict": {"embed_tokens.weight": torch.tensor([1.0, float("nan")])}},
        nan_dir / "state.pt",
    )
    with pytest.raises(CheckpointCorruptionError, match="NaN parameters detected"):
        load_multi_tier_checkpoint(nan_dir)


def test_adversarial_all_six_tiers_specifications_and_parameters() -> None:
    """Stress-test configuration invariants and theoretical parameter calculations for all 6 tiers."""
    tier_specs = [
        # tier_name, config_fn, exp_V, exp_d, exp_L, exp_Hq, exp_Hkv, exp_dffn, exp_ctx, exp_theta, exp_tied_calc
        ("Nano (20M)", HeliosConfig.nano_20m(), 32000, 320, 7, 8, 2, 864, 16384, 100000.0, 17842880),
        ("Alpha (100M)", HeliosConfig.alpha_100m(), 32000, 768, 12, 12, 4, 2048, 32768, 500000.0, 100092672),
        ("Beta (300M)", HeliosConfig.beta_300m(), 32000, 1024, 24, 16, 4, 2816, 65536, 1000000.0, 303350784),
        ("Core (1.1B)", HeliosConfig.core_1b(), 32000, 2048, 24, 16, 4, 5504, 131072, 5000000.0, 1128892416),
        ("Pro (3.3B)", HeliosConfig.pro_3b(), 32000, 3072, 32, 24, 8, 8192, 131072, 5000000.0, 3319729152),
        ("Flagship (5.8B)", HeliosConfig.flagship_6b(), 32000, 4096, 32, 32, 8, 11008, 262144, 10000000.0, 5802037248),
    ]

    for name, cfg, exp_v, exp_d, exp_l, exp_hq, exp_hkv, exp_dff, exp_ctx, exp_theta, exp_tied in tier_specs:
        # Check architecture dimensions
        assert cfg.vocab_size == exp_v, f"{name}: vocab mismatch"
        assert cfg.d_model == exp_d, f"{name}: d_model mismatch"
        assert cfg.n_layers == exp_l, f"{name}: n_layers mismatch"
        assert cfg.n_heads == exp_hq, f"{name}: n_heads mismatch"
        assert cfg.n_kv_heads == exp_hkv, f"{name}: n_kv_heads mismatch"
        assert cfg.d_ffn == exp_dff, f"{name}: d_ffn mismatch"
        assert cfg.max_seq_len == exp_ctx, f"{name}: context window mismatch"
        assert cfg.rope_theta == exp_theta, f"{name}: rope theta mismatch"

        # Check GQA head dimension and ratio
        head_dim = cfg.d_model // cfg.n_heads
        assert cfg.head_dim == head_dim
        assert cfg.n_heads % cfg.n_kv_heads == 0, f"{name}: invalid GQA ratio"

        # Check calculated parameter counts
        cal_params = calculate_theoretical_parameters(cfg)["total_parameters"]
        assert cal_params == exp_tied, f"{name}: parameter calculation mismatch {cal_params} != {exp_tied}"


def test_adversarial_file_line_limits_compliance() -> None:
    """Verify newly created adversarial test file adheres strictly to <= 500 lines ceiling."""
    current_file = Path(__file__).resolve()
    with open(current_file, "r", encoding="utf-8") as f:
        line_count = len(f.readlines())
    assert line_count <= 500, f"Line ceiling breached in {current_file.name}: {line_count} > 500"
