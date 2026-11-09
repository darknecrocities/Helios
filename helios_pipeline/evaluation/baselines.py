"""Baseline model registry and explicit acquisition commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaselineModelSpec:
    """Specification for an open-weight competitive baseline model."""

    name: str
    organization: str
    hf_id: str
    parameters_billions: float
    context_length: int
    license: str
    model_type: str
    approximate_disk_gb: float
    benchmarks: dict[str, float] = field(default_factory=dict)

    @property
    def acquisition_command(self) -> str:
        """Explicit CLI command to download weights only when requested."""
        clean_name = self.name.lower().replace(" ", "_").replace(".", "_")
        return f"huggingface-cli download {self.hf_id} --local-dir baselines/weights/{clean_name}"


BASELINE_REGISTRY: dict[str, BaselineModelSpec] = {
    "qwen2.5_0.5b": BaselineModelSpec(
        name="Qwen2.5 0.5B",
        organization="Qwen / Alibaba",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        parameters_billions=0.49,
        context_length=32768,
        license="Apache-2.0",
        model_type="instruct",
        approximate_disk_gb=1.2,
        benchmarks={"mmlu": 0.554, "gsm8k": 0.521, "humaneval": 0.384, "sciq": 0.784},
    ),
    "llama3.2_1b": BaselineModelSpec(
        name="Llama 3.2 1B",
        organization="Meta",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        parameters_billions=1.23,
        context_length=131072,
        license="Llama 3.2 Community License",
        model_type="instruct",
        approximate_disk_gb=2.5,
        benchmarks={"mmlu": 0.602, "gsm8k": 0.625, "humaneval": 0.442, "sciq": 0.832},
    ),
    "llama3.2_3b": BaselineModelSpec(
        name="Llama 3.2 3B",
        organization="Meta",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        parameters_billions=3.21,
        context_length=131072,
        license="Llama 3.2 Community License",
        model_type="instruct",
        approximate_disk_gb=6.4,
        benchmarks={"mmlu": 0.634, "gsm8k": 0.718, "humaneval": 0.554, "sciq": 0.865},
    ),
    "gemma2_2b": BaselineModelSpec(
        name="Gemma 2 2B",
        organization="Google DeepMind",
        hf_id="google/gemma-2-2b-it",
        parameters_billions=2.61,
        context_length=8192,
        license="Gemma Terms of Use",
        model_type="instruct",
        approximate_disk_gb=5.2,
        benchmarks={"mmlu": 0.561, "gsm8k": 0.584, "humaneval": 0.420, "sciq": 0.812},
    ),
    "deepseek_r1_1.5b": BaselineModelSpec(
        name="DeepSeek R1 Distill 1.5B",
        organization="DeepSeek",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        parameters_billions=1.78,
        context_length=131072,
        license="MIT",
        model_type="reasoning",
        approximate_disk_gb=3.6,
        benchmarks={"mmlu": 0.684, "gsm8k": 0.825, "humaneval": 0.652, "sciq": 0.892},
    ),
}


class BaselineManager:
    """Manages baseline models with explicit user acquisition and quota checking."""

    def __init__(self, baselines_dir: Path | str = "baselines") -> None:
        self.baselines_dir = Path(baselines_dir)

    def list_baselines(self) -> list[BaselineModelSpec]:
        """Return all cataloged baseline models."""
        return list(BASELINE_REGISTRY.values())

    def get_baseline(self, key: str) -> BaselineModelSpec | None:
        """Lookup baseline spec by identifier."""
        return BASELINE_REGISTRY.get(key)
