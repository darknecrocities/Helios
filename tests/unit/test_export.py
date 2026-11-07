"""Unit and integration tests for Helios export and deployment modules.

Covers Ollama Modelfile generation, Hugging Face Hub package serialization,
OneDrive mirroring with SHA-256 integrity and local disk space reclamation,
and export CLI commands.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helios.cli.main import app
from helios.export.hf import (
    export_huggingface,
)
from helios.export.ollama import (
    OllamaModelfileConfig,
    export_ollama,
    generate_modelfile,
)
from helios.export.onedrive import (
    OneDriveMirrorManager,
    OneDriveSyncIntegrityError,
    compute_sha256,
)
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

runner = CliRunner()


@pytest.mark.unit
def test_ollama_modelfile_generation(tmp_path: Path) -> None:
    """Verify Ollama Modelfile rendering and file generation."""
    cfg = OllamaModelfileConfig(
        base_model="llama3.2",
        temperature=0.8,
        model_tag="parejasarronkian/Helios",
    )
    rendered = cfg.render()
    assert "FROM llama3.2" in rendered
    assert "PARAMETER temperature 0.8" in rendered
    assert "Arron Kian Parejas" in rendered
    assert "parejasarronkian/Helios" in rendered

    out_file = tmp_path / "Modelfile"
    generate_modelfile(cfg, output_path=out_file)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == rendered

    export_res = export_ollama(output_dir=tmp_path / "ollama_export", base_model="llama3.2")
    assert Path(export_res["modelfile_path"]).exists()
    assert export_res["model_tag"] == "parejasarronkian/Helios"


@pytest.mark.unit
def test_huggingface_export_package(tmp_path: Path) -> None:
    """Verify Hugging Face export package structure, safetensors, and model card."""
    cfg = HeliosConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        max_seq_len=32,
    )
    model = HeliosForCausalLM(cfg)

    export_dir = tmp_path / "hf_export"
    res = export_huggingface(
        output_dir=export_dir,
        config=cfg,
        model=model,
        repo_id="Darknecrocities/Helios",
        generate_card=True,
    )

    assert (export_dir / "config.json").exists()
    assert (export_dir / "generation_config.json").exists()
    assert (export_dir / "model.safetensors").exists()
    assert (export_dir / "README.md").exists()
    assert res["weights_bytes"] > 0

    # Verify config contents
    hf_cfg = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    assert hf_cfg["architectures"] == ["HeliosForCausalLM"]
    assert hf_cfg["developer"] == "Arron Kian Parejas"

    # Verify card contents
    card = (export_dir / "README.md").read_text(encoding="utf-8")
    assert "Darknecrocities/Helios" in card
    assert "Arron Kian Parejas" in card
    assert "FineWeb-Edu" in card


@pytest.mark.unit
def test_onedrive_mirroring_and_integrity(tmp_path: Path) -> None:
    """Verify OneDrive mirroring with SHA-256 verification and local disk freeing."""
    workspace = tmp_path / "workspace"
    onedrive_dir = tmp_path / "OneDrive_Helios"
    workspace.mkdir()
    onedrive_dir.mkdir()

    # Create dummy checkpoint directory
    ckpt_step = workspace / "checkpoints" / "checkpoint_step_10"
    ckpt_step.mkdir(parents=True)
    manifest_file = ckpt_step / "manifest.json"
    manifest_file.write_text('{"step": 10}', encoding="utf-8")
    state_file = ckpt_step / "state.pt"
    state_file.write_bytes(b"dummy_weights_data_1234567890")

    manager = OneDriveMirrorManager(onedrive_root=onedrive_dir, workspace_root=workspace)
    manager.ensure_onedrive_accessible()

    # Mirror without freeing local
    items = manager.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=False)
    assert len(items) == 2
    assert (onedrive_dir / "checkpoints" / "checkpoint_step_10" / "state.pt").exists()
    assert state_file.exists()  # local not freed

    # Mirror with free_local=True
    items_freed = manager.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=True)
    assert len(items_freed) == 2
    assert not state_file.exists()  # local state.pt removed
    pointer_file = ckpt_step / "state.pt.onedrive_pointer.json"
    assert pointer_file.exists()
    pointer_data = json.loads(pointer_file.read_text(encoding="utf-8"))
    assert pointer_data["local_freed"] is True

    # Audit sync
    audit = manager.audit_sync()
    assert audit["local_checkpoints_freed"] == 1
    assert audit["bytes_saved_locally"] > 0

    # Test integrity error simulation
    corrupt_source = workspace / "corrupt.bin"
    corrupt_source.write_bytes(b"good")
    target_dst = onedrive_dir / "corrupt.bin"

    # Monkeypatch compute_sha256 during staging to force mismatch
    orig_compute = compute_sha256
    call_count = 0

    def faulty_sha(path: Path | str, chunk_size: int = 65536) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return "mismatch_hash"
        return orig_compute(path, chunk_size)

    import helios.export.disk as disk_mod

    saved_fn = disk_mod.compute_sha256
    disk_mod.compute_sha256 = faulty_sha
    try:
        with pytest.raises(OneDriveSyncIntegrityError):
            manager.mirror_file(corrupt_source, target_dst, verify_sha=True)
    finally:
        disk_mod.compute_sha256 = saved_fn


@pytest.mark.unit
def test_cli_export_subcommands(tmp_path: Path) -> None:
    """Verify typer CLI subcommands under helios export."""
    # 1. Ollama export CLI
    res_ollama = runner.invoke(
        app,
        [
            "export",
            "ollama",
            "--output-dir",
            str(tmp_path / "cli_ollama"),
            "--base-model",
            "llama3.2",
        ],
    )
    assert res_ollama.exit_code == 0
    assert "Generated Modelfile" in res_ollama.stdout
    assert (tmp_path / "cli_ollama" / "Modelfile").exists()

    # 2. Hugging Face export CLI
    res_hf = runner.invoke(
        app,
        [
            "export",
            "hf",
            "--output-dir",
            str(tmp_path / "cli_hf"),
            "--preset",
            "nano-15m",
            "--repo-id",
            "Darknecrocities/Helios",
        ],
    )
    assert res_hf.exit_code == 0
    assert "✓ Configuration:" in res_hf.stdout
    assert "✓ Weights      :" in res_hf.stdout
    assert (tmp_path / "cli_hf" / "config.json").exists()
    assert (tmp_path / "cli_hf" / "model.safetensors").exists()

    # 3. External disk / OneDrive export CLI (audit-only)
    res_onedrive = runner.invoke(
        app,
        [
            "export",
            "onedrive",
            "--destination",
            str(tmp_path / "cli_onedrive"),
            "--audit-only",
        ],
    )
    assert res_onedrive.exit_code == 0
    assert "# External Disk Mirror Audit" in res_onedrive.stdout
    assert "Destination Accessible : True" in res_onedrive.stdout

    # 4. Disk command alias
    res_disk = runner.invoke(
        app,
        [
            "export",
            "disk",
            "--destination",
            str(tmp_path / "cli_disk"),
            "--audit-only",
        ],
    )
    assert res_disk.exit_code == 0
    assert "# External Disk Mirror Audit" in res_disk.stdout

    # 5. OneTouch command alias
    res_onetouch = runner.invoke(
        app,
        [
            "export",
            "onetouch",
            "--destination",
            str(tmp_path / "cli_onetouch"),
            "--audit-only",
        ],
    )
    assert res_onetouch.exit_code == 0
    assert "# External Disk Mirror Audit" in res_onetouch.stdout
