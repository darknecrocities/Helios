"""NPU detection and compute-device resolution.

Honest scope: PyTorch cannot train directly on the Apple Neural Engine. On macOS,
"npu" resolves to MPS for training and ONNX export (CoreML EP) for ANE inference.
Ascend (torch_npu) is used natively when installed.
"""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class NPUBackendInfo:
    """Detected accelerator capabilities."""

    backend: str
    train_device: str
    onnx_providers: list[str] = field(default_factory=list)
    notes: str = ""


def _onnx_providers() -> list[str]:
    if importlib.util.find_spec("onnxruntime") is None:
        return []
    import onnxruntime as ort

    wanted = {"CoreMLExecutionProvider", "QNNExecutionProvider", "OpenVINOExecutionProvider"}
    return [p for p in ort.get_available_providers() if p in wanted]


def detect_npu_backend() -> NPUBackendInfo:
    """Probe available NPU backends in priority order."""
    if importlib.util.find_spec("torch_npu") is not None:
        return NPUBackendInfo("ascend", "npu", _onnx_providers(), "torch_npu native")
    providers = _onnx_providers()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        return NPUBackendInfo(
            "apple_ane", dev, providers, "Train on MPS; ANE inference via ONNX CoreML EP"
        )
    if providers:
        return NPUBackendInfo("onnx_npu", "cpu", providers, "Inference-only NPU via ONNX EP")
    return NPUBackendInfo("none", "cpu", [], "No NPU detected; CPU fallback")


def resolve_compute_device(requested: str) -> str:
    """Resolve 'auto' | 'npu' | 'mps' | 'cuda' | 'cpu' to a usable torch device string."""
    req = requested.lower()
    if req == "npu":
        return detect_npu_backend().train_device
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return req


class NPUAccelerator:
    """Exports static-shape ONNX graphs suitable for NPU compilers."""

    def __init__(self) -> None:
        self.info = detect_npu_backend()

    def export_npu_package(
        self, model: torch.nn.Module, vocab_size: int, out_dir: Path, seq_len: int = 128
    ) -> Path:
        """Export model to static-shape ONNX (batch=1, fixed seq_len)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "model_static.onnx"

        class _Logits(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, ids: torch.Tensor) -> torch.Tensor:
                return self.inner(ids)["logits"]

        wrapper = _Logits(model.cpu().eval())
        dummy = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        torch.onnx.export(wrapper, (dummy,), str(target), opset_version=18, dynamo=False)
        (out_dir / "npu_manifest.txt").write_text(
            f"backend={self.info.backend}\nseq_len={seq_len}\n"
            f"providers={','.join(self.info.onnx_providers)}\n",
            encoding="utf-8",
        )
        return target
