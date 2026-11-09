"""Test helpers, synthetic generators, and fault injectors for HELIOS pipeline."""

from helios_pipeline.testing.fault_injection import (
    corrupt_checkpoint_file,
    corrupt_checksum_file,
    restore_storage_mount,
    simulate_storage_disconnect,
)
from helios_pipeline.testing.synthetic_data import (
    generate_synthetic_document,
    generate_synthetic_shard,
    synthetic_token_batches,
)
from helios_pipeline.testing.tiny_model import (
    create_tiny_helios_config,
    create_tiny_helios_model,
)

__all__ = [
    "corrupt_checkpoint_file",
    "corrupt_checksum_file",
    "create_tiny_helios_config",
    "create_tiny_helios_model",
    "generate_synthetic_document",
    "generate_synthetic_shard",
    "restore_storage_mount",
    "simulate_storage_disconnect",
    "synthetic_token_batches",
]
