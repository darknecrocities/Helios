"""NPU acceleration helpers."""

from helios_pipeline.npu.accelerator import (
    NPUAccelerator,
    NPUBackendInfo,
    detect_npu_backend,
    resolve_compute_device,
)

__all__ = ["NPUAccelerator", "NPUBackendInfo", "detect_npu_backend", "resolve_compute_device"]
