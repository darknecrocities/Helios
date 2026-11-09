"""Model FLOPs Utilization (MFU) & Roofline Hardware Profiler for Helios.

Implements the PaLM / Chinchilla analytical FLOPs formulation:
  FLOPs_per_token (training) = 6 * N_params + 12 * L * H_q * d_head * T
  FLOPs_per_token (inference) = 2 * N_params + 4 * L * H_q * d_head * T

Computes live achieved TFLOPs/s, Model FLOPs Utilization (MFU %), arithmetic intensity
(FLOPs/Byte), and Roofline bottleneck classification across Apple Silicon MPS, CUDA, and CPU.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from helios.model.config import HeliosConfig
from helios.model.fused_kernels import fused_chunked_cross_entropy, get_kernel_backend_info
from helios.model.transformer import HeliosForCausalLM


@dataclass(frozen=True)
class HardwarePeakSpecs:
    """Theoretical hardware peak compute and memory bandwidth specifications."""

    device_name: str
    device_type: str
    peak_tflops_fp16: float
    memory_bandwidth_gbps: float

    @property
    def ridge_point_flops_per_byte(self) -> float:
        """Roofline ridge point (FLOPs/Byte) where compute-bound meets memory-bound."""
        return (self.peak_tflops_fp16 * 1e12) / max(1.0, self.memory_bandwidth_gbps * 1e9)


@dataclass(frozen=True)
class MFUProfileReport:
    """Comprehensive Roofline and MFU telemetry report for a Helios model tier."""

    preset: str
    device_name: str
    device_type: str
    kernel_tier: str
    total_params: int
    non_embedding_params: int
    batch_size: int
    seq_len: int
    tokens_per_step: int
    flops_per_token: float
    flops_per_step: float
    step_latency_ms: float
    tokens_per_sec: float
    achieved_tflops: float
    hardware_peak_tflops: float
    mfu_pct: float
    arithmetic_intensity_flops_per_byte: float
    roofline_ridge_point: float
    bottleneck_regime: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_hardware_peak_specs(device: torch.device | str | None = None) -> HardwarePeakSpecs:
    """Detect theoretical FP16/BF16 peak TFLOPs and memory bandwidth for the active accelerator."""
    if device is None:
        if torch.cuda.is_available():
            dev_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev_type = "mps"
        else:
            dev_type = "cpu"
    else:
        dev_type = torch.device(device).type

    if dev_type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        upper = name.upper()
        if "H100" in upper:
            return HardwarePeakSpecs(name, "cuda", 989.0, 3350.0)
        if "A100" in upper:
            return HardwarePeakSpecs(name, "cuda", 312.0, 2039.0)
        if "L4" in upper:
            return HardwarePeakSpecs(name, "cuda", 121.0, 300.0)
        if "T4" in upper:
            return HardwarePeakSpecs(name, "cuda", 65.0, 320.0)
        if "4090" in upper:
            return HardwarePeakSpecs(name, "cuda", 165.2, 1008.0)
        return HardwarePeakSpecs(name, "cuda", 100.0, 600.0)

    if dev_type == "mps":
        chip_brand = "Apple Silicon GPU (MPS)"
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                chip_brand = f"{res.stdout.strip()} (Metal MPS)"
        except Exception:
            pass
        upper = chip_brand.upper()
        if "M4" in upper:
            return HardwarePeakSpecs(chip_brand, "mps", 9.2, 120.0)
        if "M3" in upper:
            return HardwarePeakSpecs(chip_brand, "mps", 7.1, 100.0)
        if "M2" in upper:
            return HardwarePeakSpecs(chip_brand, "mps", 7.1, 100.0)
        return HardwarePeakSpecs(chip_brand, "mps", 5.2, 68.25)

    cpu_name = platform.processor() or "Host CPU"
    return HardwarePeakSpecs(cpu_name, "cpu", 0.5, 50.0)


def compute_step_flops(
    config: HeliosConfig,
    total_params: int,
    batch_size: int,
    seq_len: int,
    is_training: bool = True,
) -> tuple[float, float, int]:
    """Compute exact PaLM/Chinchilla FLOPs per token and per step.

    Returns:
        (flops_per_token, flops_per_step, non_embedding_params)
    """
    emb_params = config.vocab_size * config.d_model
    non_emb_params = max(1, total_params - emb_params)
    attn_context_flops = (
        12.0 * config.n_layers * config.n_heads * config.head_dim * seq_len
        if is_training
        else 4.0 * config.n_layers * config.n_heads * config.head_dim * seq_len
    )
    param_mult = 6.0 if is_training else 2.0
    flops_per_token = param_mult * float(non_emb_params) + attn_context_flops
    tokens_per_step = max(1, batch_size * seq_len)
    flops_per_step = flops_per_token * float(tokens_per_step)
    return flops_per_token, flops_per_step, non_emb_params


class MFUProfiler:
    """Live Roofline and MFU profiler for Helios training and inference steps."""

    def __init__(
        self,
        config: HeliosConfig,
        total_params: int,
        preset: str = "nano-20m",
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.total_params = total_params
        self.preset = preset
        self.hw_specs = detect_hardware_peak_specs(device)
        self.kernel_info = get_kernel_backend_info(device)

    def record_step(
        self,
        batch_size: int,
        seq_len: int,
        step_latency_sec: float,
        is_training: bool = True,
        bytes_per_param: float = 2.0,
    ) -> MFUProfileReport:
        """Compute MFU and Roofline telemetry from a measured step duration."""
        elapsed = max(1e-6, float(step_latency_sec))
        tokens = max(1, batch_size * seq_len)
        flops_per_tok, flops_per_step, non_emb = compute_step_flops(
            self.config,
            self.total_params,
            batch_size=batch_size,
            seq_len=seq_len,
            is_training=is_training,
        )
        achieved_flops_per_sec = flops_per_step / elapsed
        achieved_tflops = achieved_flops_per_sec / 1e12
        mfu_pct = min(
            100.0, (achieved_tflops / max(1e-6, self.hw_specs.peak_tflops_fp16)) * 100.0
        )

        # Memory traffic estimate: weights read/write + activations
        weight_traffic_bytes = float(self.total_params) * bytes_per_param * (3.0 if is_training else 1.0)
        act_traffic_bytes = float(tokens * self.config.d_model * self.config.n_layers) * bytes_per_param * 4.0
        total_bytes = max(1.0, weight_traffic_bytes + act_traffic_bytes)
        arith_intensity = flops_per_step / total_bytes
        ridge = self.hw_specs.ridge_point_flops_per_byte
        regime = (
            "Compute-Bound"
            if arith_intensity >= ridge
            else "Memory-Bandwidth-Bound"
        )

        return MFUProfileReport(
            preset=self.preset,
            device_name=self.hw_specs.device_name,
            device_type=self.hw_specs.device_type,
            kernel_tier=self.kernel_info.tier,
            total_params=self.total_params,
            non_embedding_params=non_emb,
            batch_size=batch_size,
            seq_len=seq_len,
            tokens_per_step=tokens,
            flops_per_token=round(flops_per_tok, 2),
            flops_per_step=round(flops_per_step, 2),
            step_latency_ms=round(elapsed * 1000.0, 2),
            tokens_per_sec=round(tokens / elapsed, 2),
            achieved_tflops=round(achieved_tflops, 4),
            hardware_peak_tflops=self.hw_specs.peak_tflops_fp16,
            mfu_pct=round(mfu_pct, 2),
            arithmetic_intensity_flops_per_byte=round(arith_intensity, 2),
            roofline_ridge_point=round(ridge, 2),
            bottleneck_regime=regime,
        )


def profile_model_mfu(
    preset: str = "nano-20m",
    batch_size: int = 2,
    seq_len: int = 256,
    warmup_steps: int = 1,
    measure_steps: int = 3,
    device: str | None = None,
) -> MFUProfileReport:
    """Execute a live training step benchmark on a Helios preset and return MFU telemetry."""
    if device is None:
        if torch.cuda.is_available():
            dev_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev_str = "mps"
        else:
            dev_str = "cpu"
    else:
        dev_str = device

    dev = torch.device(dev_str)
    cfg = HeliosConfig.from_preset(preset)
    model = HeliosForCausalLM(cfg).to(dev)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    profiler = MFUProfiler(cfg, total_params=total_params, preset=preset, device=dev)

    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)

    def _run_step() -> None:
        model.zero_grad(set_to_none=True)
        h = model.model(input_ids)["last_hidden_state"]
        loss = fused_chunked_cross_entropy(
            h, model.lm_head.weight, targets, chunk_size=512, z_loss_weight=cfg.z_loss_weight
        )
        loss.backward()
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(max(0, warmup_steps)):
        _run_step()

    t0 = time.perf_counter()
    for _ in range(max(1, measure_steps)):
        _run_step()
    avg_sec = (time.perf_counter() - t0) / max(1, measure_steps)

    return profiler.record_step(
        batch_size=batch_size,
        seq_len=seq_len,
        step_latency_sec=avg_sec,
        is_training=True,
    )
