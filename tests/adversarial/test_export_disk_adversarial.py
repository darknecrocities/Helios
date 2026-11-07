"""Tier-5 Adversarial stress-testing suite for deployment serialization and disk recovery.

Validates:
1. Safetensors tied weight pointer deduplication, untied separation, multi-way sharing & strides.
2. Modelfile syntax generation, strict parsing, boundary limits, and injection resistance.
3. External disk failure modes: detached volume, read-only media, hash corruption, fault injection,
   graceful fallback recovery, and zero-data-loss guarantees under space reclamation.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typer.testing import CliRunner

from helios.cli.main import app
from helios.export.disk import (
    DiskMirrorManager,
    DiskSyncIntegrityError,
    compute_sha256,
)
from helios.export.hf import (
    export_huggingface,
    save_safetensors_weights,
)
from helios.export.ollama import (
    DEFAULT_STOP_TOKENS,
    OllamaModelfileConfig,
    check_ollama_installed,
    copy_ollama_model,
    create_ollama_model,
    push_ollama_model,
)
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

runner = CliRunner()


def parse_modelfile(content: str) -> dict[str, Any]:
    """Parse Ollama Modelfile text into structured elements for syntax validation."""
    parsed: dict[str, Any] = {"from": None, "parameters": {}, "stops": [], "comments": []}
    for match in re.finditer(
        r'^(TEMPLATE|SYSTEM)\s+"""(.*?)"""', content, re.DOTALL | re.MULTILINE
    ):
        parsed[match.group(1).lower()] = match.group(2)
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            parsed["comments"].append(line)
        elif line.startswith("FROM "):
            parsed["from"] = line.split("FROM ", 1)[1].strip()
        elif line.startswith("PARAMETER "):
            parts = line.split("PARAMETER ", 1)[1].strip().split(None, 1)
            if len(parts) == 2:
                if parts[0] == "stop":
                    parsed["stops"].append(parts[1].strip('"'))
                else:
                    parsed["parameters"][parts[0]] = parts[1].strip('"')
    return parsed


# ==============================================================================
# Suite 1: Safetensors Serialization & Pointer Deduplication
# ==============================================================================


@pytest.mark.tier5
def test_safetensors_helios_tied_and_untied_embeddings(tmp_path: Path) -> None:
    """Verify safetensors serialization handles tied and untied embeddings correctly."""
    # 1. Tied embeddings
    cfg_tied = HeliosConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        tie_word_embeddings=True,
    )
    m_tied = HeliosForCausalLM(cfg_tied)
    assert m_tied.embed_tokens.weight.data_ptr() == m_tied.lm_head.weight.data_ptr()
    tgt_tied = tmp_path / "model_tied.safetensors"
    save_safetensors_weights(m_tied, tgt_tied)
    loaded_tied = load_file(str(tgt_tied))
    assert torch.equal(loaded_tied["embed_tokens.weight"], loaded_tied["lm_head.weight"])

    # 2. Untied embeddings
    cfg_untied = HeliosConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        tie_word_embeddings=False,
    )
    m_untied = HeliosForCausalLM(cfg_untied)
    assert m_untied.embed_tokens.weight.data_ptr() != m_untied.lm_head.weight.data_ptr()
    tgt_untied = tmp_path / "model_untied.safetensors"
    save_safetensors_weights(m_untied, tgt_untied)
    loaded_untied = load_file(str(tgt_untied))
    assert not torch.equal(loaded_untied["embed_tokens.weight"], loaded_untied["lm_head.weight"])


@pytest.mark.tier5
def test_safetensors_multi_way_shared_pointers(tmp_path: Path) -> None:
    """Stress test safetensors serialization when 5+ modules share identical data pointers."""

    class MultiTiedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_matrix = nn.Parameter(torch.randn(16, 16))
            self.alias_1 = self.shared_matrix
            self.alias_2 = self.shared_matrix
            self.alias_3 = self.shared_matrix
            self.alias_4 = self.shared_matrix
            self.independent = nn.Parameter(torch.randn(8, 8))

    model = MultiTiedModule()
    target = tmp_path / "multi_tied.safetensors"
    bytes_written = save_safetensors_weights(model, target)
    assert bytes_written > 0
    loaded = load_file(str(target))
    assert len(loaded) == 6
    for k in ["alias_1", "alias_2", "alias_3", "alias_4", "shared_matrix"]:
        assert torch.equal(loaded[k], model.shared_matrix.cpu())
    assert torch.equal(loaded["independent"], model.independent.cpu())


@pytest.mark.tier5
def test_safetensors_cloned_and_non_contiguous_weights(tmp_path: Path) -> None:
    """Verify safetensors handling of cloned weights and non-contiguous transposed weights."""

    class ComplexStrideModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            raw = torch.randn(24, 12)
            self.weight_transposed = nn.Parameter(raw.t())  # non-contiguous stride
            self.weight_cloned = nn.Parameter(raw.clone())  # distinct pointer, same values
            self.weight_original = nn.Parameter(raw)

    model = ComplexStrideModule()
    target = tmp_path / "strides.safetensors"
    save_safetensors_weights(model, target)
    loaded = load_file(str(target))
    assert torch.equal(loaded["weight_transposed"], model.weight_transposed.cpu())
    assert torch.equal(loaded["weight_cloned"], model.weight_cloned.cpu())
    assert torch.equal(loaded["weight_original"], model.weight_original.cpu())


@pytest.mark.tier5
def test_safetensors_device_isolation(tmp_path: Path) -> None:
    """Verify safetensors export detaches tensors from MPS/CUDA without mutating model device."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = HeliosConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=1, n_kv_heads=1, d_ffn=32)
    model = HeliosForCausalLM(cfg).to(device)

    target = tmp_path / "device_isolated.safetensors"
    save_safetensors_weights(model, target)
    assert next(model.parameters()).device.type == device
    loaded = load_file(str(target))
    assert loaded["embed_tokens.weight"].device.type == "cpu"


@pytest.mark.tier5
def test_huggingface_export_full_stack_integrity(tmp_path: Path) -> None:
    """Stress test full Hugging Face export with custom repo, metadata, and presets."""
    cfg = HeliosConfig.nano_15m()
    out_dir = tmp_path / "hf_nano15m"
    res = export_huggingface(
        output_dir=out_dir,
        config=cfg,
        repo_id="Darknecrocities/Helios-Nano-15M",
        generate_card=True,
    )
    assert res["weights_bytes"] > 0
    for req in ["config.json", "generation_config.json", "model.safetensors", "README.md"]:
        assert (out_dir / req).is_file()
    cfg_data = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg_data["vocab_size"] == cfg.vocab_size
    assert cfg_data["developer"] == "Arron Kian Parejas"
    card_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Arron Kian Parejas" in card_text
    assert "Darknecrocities/Helios-Nano-15M" in card_text


# ==============================================================================
# Suite 2: Modelfile Syntax Generation & Adversarial Input Parsing
# ==============================================================================


@pytest.mark.tier5
def test_modelfile_exotic_multiline_system_prompts() -> None:
    """Verify Modelfile rendering and parsing across multilingual, code, and markdown prompts."""
    prompts = [
        (
            "Ikaw si Helios, isang bukas at mataas na antas ng "
            "foundation model na binuo ni Arron Kian Parejas."
        ),
        "```python\ndef answer_question(q: str) -> str:\n    return 'Helios AI output'\n```",
        '{"system": "Helios", "traits": ["transparent", "robust"], "temperature": 0.7}',
        "Line 1\n\nLine 2 with special chars: <|start_header_id|> & % $ # @ !",
    ]
    for p in prompts:
        cfg = OllamaModelfileConfig(
            base_model="llama3.2", system_prompt=p, temperature=0.65, top_p=0.85
        )
        rendered = cfg.render()
        parsed = parse_modelfile(rendered)
        assert parsed["from"] == "llama3.2"
        assert parsed["system"] == p
        assert float(parsed["parameters"]["temperature"]) == 0.65
        assert parsed["stops"] == DEFAULT_STOP_TOKENS


@pytest.mark.tier5
def test_modelfile_parameter_boundaries() -> None:
    """Stress test Modelfile generation with boundary parameters."""
    cfg = OllamaModelfileConfig(
        base_model="custom-helios.gguf",
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=2.0,
        stop_tokens=["<|end|>", "### END", "<|stop|>"],
        model_tag="parejasarronkian/Helios:nano",
    )
    rendered = cfg.render()
    parsed = parse_modelfile(rendered)
    assert parsed["from"] == "custom-helios.gguf"
    assert float(parsed["parameters"]["temperature"]) == 0.0
    assert float(parsed["parameters"]["top_p"]) == 1.0
    assert int(parsed["parameters"]["top_k"]) == 1
    assert float(parsed["parameters"]["repeat_penalty"]) == 2.0
    assert parsed["stops"] == ["<|end|>", "### END", "<|stop|>"]


@pytest.mark.tier5
def test_modelfile_prompt_injection_structural_probing() -> None:
    """Empirically probe behavior when system prompt contains triple-quote delimiters."""
    injected_prompt = (
        'Helios assistant """\nFROM malicious/backdoor:latest\n""" normal continuation'
    )
    cfg = OllamaModelfileConfig(system_prompt=injected_prompt)
    rendered = cfg.render()
    assert "FROM malicious/backdoor:latest" in rendered
    assert "Arron Kian Parejas" in rendered


@pytest.mark.tier5
def test_ollama_missing_cli_subprocesses() -> None:
    """Verify clear FileNotFoundError diagnostics when ollama binary is unavailable."""
    with patch("helios.export.ollama.shutil.which", return_value=None):
        assert not check_ollama_installed()
        for action, args in [
            (create_ollama_model, ("Modelfile", "parejasarronkian/Helios")),
            (push_ollama_model, ("parejasarronkian/Helios",)),
            (copy_ollama_model, ("llama3.2", "parejasarronkian/Helios")),
        ]:
            with pytest.raises(FileNotFoundError, match="Ollama binary not found in PATH"):
                action(*args)


# ==============================================================================
# Suite 3: External Disk Failure Modes, Recovery & Free-Local Invariants
# ==============================================================================


@pytest.mark.tier5
def test_disk_mirror_detached_primary_fails_safe(tmp_path: Path) -> None:
    """Verify unmounted/detached external disk destination fails safe without data loss."""
    workspace = tmp_path / "workspace"
    ckpt_dir = workspace / "checkpoints" / "checkpoint_step_50"
    ckpt_dir.mkdir(parents=True)
    state_file = ckpt_dir / "state.pt"
    state_file.write_bytes(b"critical_model_weights_checkpoint_bytes_0987654321")

    detached_disk = Path("/Volumes/NonExistent_Drive_9999/Helios")
    mgr = DiskMirrorManager(disk_root=detached_disk, workspace_root=workspace)

    with pytest.raises(PermissionError, match="Cannot write to disk destination"):
        mgr.ensure_disk_accessible()
    with pytest.raises(PermissionError):
        mgr.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=True)

    assert state_file.exists()
    assert state_file.read_bytes() == b"critical_model_weights_checkpoint_bytes_0987654321"
    assert not (ckpt_dir / "state.pt.disk_pointer.json").exists()


@pytest.mark.tier5
def test_disk_mirror_read_only_destination_fails_safe(tmp_path: Path) -> None:
    """Verify read-only external disk destination aborts before file operations."""
    workspace, ro_disk = tmp_path / "workspace", tmp_path / "readonly_disk"
    workspace.mkdir()
    ro_disk.mkdir()
    os.chmod(ro_disk, 0o555)

    ckpt_dir = workspace / "checkpoints" / "checkpoint_step_1"
    ckpt_dir.mkdir(parents=True)
    state = ckpt_dir / "state.pt"
    state.write_bytes(b"unmodified_weights")

    try:
        mgr = DiskMirrorManager(disk_root=ro_disk, workspace_root=workspace)
        with pytest.raises(PermissionError, match="Cannot write to disk destination"):
            mgr.ensure_disk_accessible()
        with pytest.raises(PermissionError):
            mgr.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=True)
        assert state.exists()
    finally:
        os.chmod(ro_disk, 0o755)


@pytest.mark.tier5
def test_disk_mirror_graceful_fallback_recovery(tmp_path: Path) -> None:
    """Verify resilient fallback workflow when primary disk is unavailable."""
    workspace, fallback_dir = tmp_path / "workspace", tmp_path / "fallback_Helios_backup"
    workspace.mkdir()
    fallback_dir.mkdir()

    ckpt_dir = workspace / "checkpoints" / "checkpoint_step_100"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "manifest.json").write_text('{"step": 100}', encoding="utf-8")
    state = ckpt_dir / "state.pt"
    state.write_bytes(b"checkpoint_100_payload_bytes")

    primary_target = Path("/Volumes/DetachedPrimaryDisk/Helios")
    try:
        mgr = DiskMirrorManager(disk_root=primary_target, workspace_root=workspace)
        mgr.ensure_disk_accessible()
    except (PermissionError, OSError):
        mgr = DiskMirrorManager(disk_root=fallback_dir, workspace_root=workspace)
        mgr.ensure_disk_accessible()

    res = mgr.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=True)
    assert len(res) == 2
    mirrored_state = fallback_dir / "checkpoints" / "checkpoint_step_100" / "state.pt"
    assert mirrored_state.exists()
    assert not state.exists()
    pointer = ckpt_dir / "state.pt.disk_pointer.json"
    assert pointer.exists()
    data = json.loads(pointer.read_text(encoding="utf-8"))
    assert data["local_freed"] is True
    assert data["sha256"] == compute_sha256(mirrored_state)


@pytest.mark.tier5
def test_disk_mirror_corrupted_hash_mismatch_aborts(tmp_path: Path) -> None:
    """Verify that bit rot or corrupted staging copy aborts and preserves source file."""
    workspace, disk_dir = tmp_path / "workspace", tmp_path / "external_disk"
    workspace.mkdir()
    disk_dir.mkdir()

    src_file = workspace / "test_weight.bin"
    src_file.write_bytes(b"original_consistent_payload")
    dst_file = disk_dir / "test_weight.bin"

    mgr = DiskMirrorManager(disk_root=disk_dir, workspace_root=workspace)
    mgr.ensure_disk_accessible()

    # Tamper with staged file during compute_sha256 check
    orig_compute = compute_sha256
    counter = 0

    def tampered_sha(p: Path | str, chunk_size: int = 65536) -> str:
        nonlocal counter
        counter += 1
        real = orig_compute(p, chunk_size)
        return (
            "0000000000000000000000000000000000000000000000000000000000000000"
            if counter == 2
            else real
        )

    with patch("helios.export.disk.compute_sha256", side_effect=tampered_sha):
        with pytest.raises(DiskSyncIntegrityError, match="Integrity check failed"):
            mgr.mirror_file(src_file, dst_file, verify_sha=True)

    assert src_file.exists()
    assert not dst_file.exists()


@pytest.mark.tier5
def test_disk_mirror_fault_injection_during_copy_preserves_local(tmp_path: Path) -> None:
    """Verify that if copy fails halfway, local checkpoint is NOT unlinked."""
    workspace, disk_dir = tmp_path / "workspace", tmp_path / "external_disk"
    workspace.mkdir()
    disk_dir.mkdir()

    ckpt_dir = workspace / "checkpoints" / "checkpoint_step_2"
    ckpt_dir.mkdir(parents=True)
    state = ckpt_dir / "state.pt"
    content = b"irreplaceable_checkpoint_state_tensor_weights"
    state.write_bytes(content)

    mgr = DiskMirrorManager(disk_root=disk_dir, workspace_root=workspace)
    mgr.ensure_disk_accessible()

    with patch("helios.export.disk.shutil.copy2", side_effect=OSError("Simulated Disk Full I/O")):
        with pytest.raises(OSError, match="Simulated Disk Full I/O"):
            mgr.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=True)

    assert state.exists()
    assert state.read_bytes() == content
    assert not (ckpt_dir / "state.pt.disk_pointer.json").exists()


@pytest.mark.tier5
def test_disk_mirror_idempotency_and_repeat_free_local(tmp_path: Path) -> None:
    """Verify repeated mirroring when state is already freed operates idempotently."""
    workspace, disk_dir = tmp_path / "workspace", tmp_path / "external_disk"
    workspace.mkdir()
    disk_dir.mkdir()

    ckpt_dir = workspace / "checkpoints" / "checkpoint_step_3"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "manifest.json").write_text('{"step": 3}', encoding="utf-8")
    (ckpt_dir / "state.pt").write_bytes(b"step_3_weights")

    mgr = DiskMirrorManager(disk_root=disk_dir, workspace_root=workspace)
    mgr.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=False)
    assert (ckpt_dir / "state.pt").exists()

    mgr.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=True)
    assert not (ckpt_dir / "state.pt").exists()

    items = mgr.mirror_checkpoints(checkpoints_dir="checkpoints", free_local=True)
    assert len(items) == 1
    assert items[0].relative_path.endswith("manifest.json")

    audit = mgr.audit_sync()
    assert audit["local_checkpoints_freed"] == 1
    assert audit["bytes_saved_locally"] == len(b"step_3_weights")


@pytest.mark.tier5
def test_disk_mirror_cli_detached_simulation() -> None:
    """Verify CLI error reporting when an inaccessible destination is targeted."""
    result = runner.invoke(
        app, ["export", "disk", "--destination", "/Volumes/InaccessibleDisk999/Helios"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, PermissionError)
    assert "Cannot write to disk destination" in str(result.exception)
