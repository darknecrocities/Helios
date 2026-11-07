"""CLI commands for Helios export, deployment, and cloud/external storage mirroring."""

from __future__ import annotations

from typing import Optional

import typer

from helios.export.disk import (
    DiskMirrorManager,
    mirror_to_disk,
)
from helios.export.hf import export_huggingface
from helios.export.ollama import export_ollama
from helios.model.config import HeliosConfig

export_app = typer.Typer(
    name="export",
    help="Helios deployment export: Ollama, Hugging Face Hub, and external disk mirroring.",
    add_completion=False,
)


@export_app.command(name="ollama")
def export_ollama_cmd(
    output_dir: str = typer.Option(
        "deploy/ollama",
        "--output-dir",
        "-o",
        help="Target output directory for the Modelfile.",
    ),
    base_model: str = typer.Option(
        "llama3.2",
        "--base-model",
        "-b",
        help="Base model reference or path to converted GGUF model.",
    ),
    model_tag: str = typer.Option(
        "parejasarronkian/Helios",
        "--model-tag",
        "-t",
        help="Ollama model namespace and tag name.",
    ),
    create_model: bool = typer.Option(
        False,
        "--create",
        "-c",
        help="Execute `ollama create -f Modelfile <tag>` automatically.",
    ),
    push_model: bool = typer.Option(
        False,
        "--push",
        "-p",
        help="Execute `ollama push <tag>` to Ollama registry.",
    ),
) -> None:
    """Generate Ollama Modelfile and optionally create/push to Ollama registry."""
    typer.echo(f"Exporting Ollama deployment definition for {model_tag}...")
    res = export_ollama(
        output_dir=output_dir,
        base_model=base_model,
        model_tag=model_tag,
        create_model=create_model,
        push_model=push_model,
    )
    typer.echo(f"✓ Generated Modelfile at {res['modelfile_path']}")
    if res.get("created"):
        typer.echo(f"✓ Created local Ollama model {model_tag}")
    if res.get("pushed"):
        typer.echo(f"✓ Pushed {model_tag} to Ollama registry")


@export_app.command(name="hf")
def export_hf_cmd(
    output_dir: str = typer.Option(
        "deploy/huggingface",
        "--output-dir",
        "-o",
        help="Target directory for Hugging Face repository package.",
    ),
    preset: str = typer.Option(
        "nano-20m",
        "--preset",
        "-m",
        help="Helios model preset (nano-15m, nano-20m, nano-25m, nano-35m).",
    ),
    checkpoint_path: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to trained checkpoint directory or state.pt file.",
    ),
    repo_id: str = typer.Option(
        "Darknecrocities/Helios",
        "--repo-id",
        "-r",
        help="Target Hugging Face Hub repository identifier.",
    ),
    generate_card: bool = typer.Option(
        True,
        "--card/--no-card",
        help="Generate comprehensive Model Card README.md.",
    ),
) -> None:
    """Export model to Hugging Face Hub format (config, safetensors, tokenizer, card)."""
    typer.echo(f"Exporting Hugging Face package for {repo_id} (preset: {preset})...")
    cfg = HeliosConfig.from_preset(preset)
    res = export_huggingface(
        output_dir=output_dir,
        config=cfg,
        checkpoint_path=checkpoint_path,
        repo_id=repo_id,
        generate_card=generate_card,
    )
    typer.echo(f"✓ Configuration: {res['config_file']}")
    typer.echo(f"✓ Weights      : {res['weights_file']} ({res['weights_bytes'] / 1e6:.2f} MB)")
    typer.echo(f"✓ Tokenizer    : {len(res['tokenizer_files'])} artifacts written")
    typer.echo(f"✓ Model Card   : {res['model_card_file']}")
    typer.echo(f"✓ Full Package : {res['output_directory']}")


@export_app.command(name="disk")
@export_app.command(name="onetouch")
@export_app.command(name="onedrive")
def export_disk_cmd(
    destination: Optional[str] = typer.Option(
        None,
        "--destination",
        "-d",
        help="Path to external disk mirror directory (default: /Volumes/One Touch/Helios).",
    ),
    checkpoints_dir: str = typer.Option(
        "checkpoints",
        "--checkpoints-dir",
        help="Local checkpoints directory to mirror.",
    ),
    dist_dir: str = typer.Option(
        "dist",
        "--dist-dir",
        help="Local distribution wheels directory to mirror.",
    ),
    free_local: bool = typer.Option(
        False,
        "--free-local",
        help="Free local disk space by removing local state.pt after SHA-256 confirmation.",
    ),
    audit_only: bool = typer.Option(
        False,
        "--audit-only",
        help="Only display synchronization and disk space audit without copying.",
    ),
) -> None:
    """Mirror checkpoints and distribution artifacts to external disk with SHA-256 integrity."""
    mgr = DiskMirrorManager(disk_root=destination)

    if audit_only:
        audit = mgr.audit_sync()
        typer.echo("# External Disk Mirror Audit")
        typer.echo(f"Destination Accessible : {audit['disk_accessible']}")
        typer.echo(f"Destination Path       : {audit['disk_root']}")
        typer.echo(f"Manifest Present       : {audit['manifest_exists']}")
        typer.echo(f"Local Checkpoints Freed: {audit['local_checkpoints_freed']}")
        typer.echo(f"Local Disk Space Saved : {audit.get('mb_saved_locally', 0.0)} MB")
        return

    typer.echo(f"Mirroring Helios assets to external disk ({mgr.disk_root})...")
    if free_local:
        typer.echo(
            "⚠️  --free-local is active: local state.pt files will be "
            "replaced with pointers after verified copy."
        )

    res = mirror_to_disk(
        destination_path=destination,
        checkpoints_dir=checkpoints_dir,
        dist_dir=dist_dir,
        free_local=free_local,
    )
    typer.echo(
        f"✓ Mirrored {res['mirrored_count']} items ({res['total_mb_mirrored']:.2f} MB) "
        f"with SHA-256 integrity."
    )
    typer.echo(f"✓ Updated sync manifest at {res['manifest_path']}")
    if free_local:
        typer.echo(f"✓ Local disk space reclaimed: {res['audit'].get('mb_saved_locally', 0.0)} MB")


@export_app.command(name="physical")
def export_physical_cmd(
    output_dir: str = typer.Option(
        "dist/physical_models",
        "--output-dir",
        "-o",
        help="Target directory for physical model exports (.onnx, .tflite, .safetensors, .pt).",
    ),
    preset: str = typer.Option(
        "nano-20m",
        "--preset",
        "-m",
        help="Helios model preset if initializing weights.",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-p",
        help="Path to trained checkpoint directory or state.pt.",
    ),
) -> None:
    """Export physical on-device formats (.onnx, .tflite, .safetensors, .pt)."""
    from helios.cli.utils import load_model_and_config
    from helios.export.physical import export_all_physical_formats

    typer.echo(f"Exporting physical models to {output_dir}...")
    model, config = load_model_and_config(preset=preset, model_path=model_path)
    res = export_all_physical_formats(model, config, output_dir=output_dir)
    for fmt, info in res["formats"].items():
        mb = info["bytes"] / 1e6
        typer.echo(f"✓ {fmt:<12}: {info['file']} ({mb:.2f} MB)")
    typer.echo(f"✓ Physical manifest written: {res['manifest_file']}")
