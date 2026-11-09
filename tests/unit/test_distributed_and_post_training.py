"""Unit tests for Frontier Distributed Sharding, Fused Kernels, MFU Profiler, DPO, and GRPO."""

from __future__ import annotations

import torch

from helios.model.config import HeliosConfig
from helios.model.fused_kernels import (
    benchmark_fused_vs_eager,
    fused_chunked_cross_entropy,
    fused_rmsnorm,
    fused_swiglu,
    get_kernel_backend_info,
)
from helios.training.distributed import (
    DistributedShardSampler,
    run_distributed_benchmark,
)
from helios.training.dpo import (
    DPOConfig,
    DPOTrainer,
    compute_dpo_loss,
    compute_sequence_logprobs,
)
from helios.training.grpo import (
    GRPOConfig,
    GRPOTrainer,
    VerifiableRewardEngine,
    compute_group_advantages,
    compute_grpo_loss,
)
from helios.training.mfu import (
    MFUProfiler,
    compute_step_flops,
    detect_hardware_peak_specs,
)


def test_fused_chunked_cross_entropy_parity_and_memory() -> None:
    """Verify FusedChunkedCrossEntropy matches eager cross-entropy loss & gradients."""
    res = benchmark_fused_vs_eager(
        batch_size=2, seq_len=128, d_model=64, vocab_size=512, chunk_size=32, device="cpu"
    )
    assert res["loss_abs_diff"] < 1e-4
    assert res["grad_h_max_diff"] < 1e-4
    assert res["grad_w_max_diff"] < 1e-4
    assert res["logit_memory_reduction_pct"] > 50.0


def test_fused_chunked_cross_entropy_with_ignore_index_and_z_loss() -> None:
    """Verify ignore_index masking and PaLM Z-loss gradient propagation."""
    torch.manual_seed(7)
    h = torch.randn(2, 16, 32, requires_grad=True)
    w = torch.randn(100, 32, requires_grad=True)
    targets = torch.randint(0, 100, (2, 16))
    targets[0, :4] = -100

    loss = fused_chunked_cross_entropy(h, w, targets, ignore_index=-100, chunk_size=8, z_loss_weight=1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert h.grad is not None and torch.all(torch.isfinite(h.grad))
    assert torch.all(h.grad[0, :4] == 0.0)
    assert w.grad is not None and torch.all(torch.isfinite(w.grad))


def test_fused_rmsnorm_and_swiglu() -> None:
    """Verify fused RMSNorm and SwiGLU helper kernels."""
    x = torch.randn(2, 8, 16)
    weight = torch.ones(16)
    out = fused_rmsnorm(x, weight)
    assert out.shape == x.shape
    assert torch.all(torch.isfinite(out))

    gate = torch.randn(2, 8, 32)
    up = torch.randn(2, 8, 32)
    act = fused_swiglu(gate, up)
    assert act.shape == gate.shape
    info = get_kernel_backend_info("cpu")
    assert info.chunked_ce_enabled is True


def test_mfu_profiler_and_roofline_telemetry() -> None:
    """Verify exact PaLM/Chinchilla FLOPs and Roofline MFU calculation."""
    cfg = HeliosConfig.nano_20m()
    flops_tok, flops_step, non_emb = compute_step_flops(
        cfg, total_params=17_840_000, batch_size=2, seq_len=128, is_training=True
    )
    assert non_emb > 0
    assert flops_tok > 6.0 * non_emb
    assert abs(flops_step - flops_tok * 256) < 1.0

    hw = detect_hardware_peak_specs("cpu")
    assert hw.peak_tflops_fp16 > 0.0
    profiler = MFUProfiler(cfg, total_params=17_840_000, preset="nano-20m", device="cpu")
    rep = profiler.record_step(batch_size=2, seq_len=128, step_latency_sec=0.05)
    assert rep.tokens_per_step == 256
    assert rep.mfu_pct >= 0.0
    assert rep.bottleneck_regime in ("Compute-Bound", "Memory-Bandwidth-Bound")


def test_distributed_shard_sampler_zero_overlap() -> None:
    """Verify rank-strided shard sampler assigns disjoint indices across ranks."""
    world_size = 4
    rank_sets = []
    for r in range(world_size):
        sampler = DistributedShardSampler(total_records=1000, rank=r, world_size=world_size, start_offset=10)
        idxs = sampler.indices_for_steps(num_steps=10, batch_size=2)
        assert len(idxs) == 20
        rank_sets.append(set(idxs))

    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert rank_sets[i].isdisjoint(rank_sets[j])


def test_zero_sharded_optimizer_convergence_and_memory_savings() -> None:
    """Verify ZeRO-1 and ZeRO-2 optimizer state partitioning and loss reduction."""
    res_z1 = run_distributed_benchmark(
        preset="nano-15m", strategy="zero1", virtual_world_size=4, steps=3, seq_len=32, batch_size=2
    )
    assert res_z1["loss_decreased"] is True
    assert res_z1["memory_saved_pct"] > 15.0

    res_z2 = run_distributed_benchmark(
        preset="nano-15m", strategy="zero2", virtual_world_size=4, steps=3, seq_len=32, batch_size=2
    )
    assert res_z2["loss_decreased"] is True
    assert res_z2["memory_saved_pct"] > res_z1["memory_saved_pct"]


def test_dpo_and_ipo_loss_and_trainer() -> None:
    """Verify DPO/IPO loss math and DPOTrainer positive reward margin progression."""
    pol_c = torch.tensor([-10.0, -12.0], requires_grad=True)
    pol_r = torch.tensor([-15.0, -18.0], requires_grad=True)
    ref_c = torch.tensor([-11.0, -13.0])
    ref_r = torch.tensor([-14.0, -16.0])

    loss, c_rew, r_rew, margin, acc = compute_dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    assert torch.isfinite(loss)
    assert float(margin.item()) > 0.0
    assert float(acc.item()) == 1.0

    ipo_loss, _, _, _, _ = compute_dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1, ipo=True)
    assert torch.isfinite(ipo_loss)

    logits = torch.randn(2, 10, 64)
    labels = torch.randint(0, 64, (2, 10))
    seq_lp = compute_sequence_logprobs(logits, labels, prompt_lens=[3, 4])
    assert seq_lp.shape == (2,)

    trainer = DPOTrainer(
        DPOConfig(preset="nano-15m", steps=3, batch_size=2, seq_len=48, save_checkpoint=False)
    )
    rep = trainer.run()
    assert rep.final_loss <= rep.initial_loss
    assert rep.final_reward_margin >= rep.initial_reward_margin


def test_grpo_verifiers_and_trainer() -> None:
    """Verify AST math, Python syntax, <thought> verifiers, and GRPOTrainer execution."""
    verifier = VerifiableRewardEngine()

    good_math = "<thought>\n1. 3x = 15.\n2. x = 5.\n</thought>\nx = 5"
    bad_math = "The answer is 99"
    assert verifier.score_completion(good_math, "math", "5") > 1.0
    assert verifier.score_completion(bad_math, "math", "5") < 0.0

    good_code = "<thought>\n1. Define add(a, b).\n</thought>\ndef add(a, b):\n    return a + b"
    bad_code = "def add(a, b) return"
    assert verifier.score_completion(good_code, "coding", "add") > 1.0
    assert verifier.score_completion(bad_code, "coding", "add") < 0.0

    rewards = torch.tensor([1.5, 1.2, -0.2, -0.5])
    adv = compute_group_advantages(rewards)
    assert abs(float(adv.mean().item())) < 1e-5
    assert float(adv[0].item()) > 0.0 and float(adv[-1].item()) < 0.0

    pol_lp = torch.tensor([-8.0, -9.0, -14.0, -15.0], requires_grad=True)
    old_lp = pol_lp.detach().clone()
    ref_lp = torch.tensor([-8.5, -9.2, -13.5, -14.8])
    loss, pol_loss, kl = compute_grpo_loss(pol_lp, old_lp, ref_lp, adv)
    assert torch.isfinite(loss)
    assert float(kl.item()) >= 0.0

    grpo = GRPOTrainer(
        GRPOConfig(preset="nano-15m", steps=3, group_size=4, seq_len=48, save_checkpoint=False)
    )
    rep = grpo.run()
    assert rep.steps == 3
    assert rep.thought_format_rate >= 0.5
