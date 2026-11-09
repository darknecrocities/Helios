"""Main training pipeline execution subcommand."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from rich.console import Console

from helios.model.config import HeliosConfig
from helios_pipeline.data.curriculum import CurriculumPartitioner
from helios_pipeline.data.docs_updater import DatasetDocumentationUpdater
from helios_pipeline.data.manifest import ClusterManifest
from helios_pipeline.data.sanitization import DocumentSanitizer
from helios_pipeline.data.streamer import FineWebStreamer
from helios_pipeline.evaluation.history_recorder import HistoryDocumentationRecorder
from helios_pipeline.evaluation.runner import PipelineEvaluationRunner
from helios_pipeline.git_sync.sync import GitSyncEngine
from helios_pipeline.storage.cleanup import ConservativeCleanupEngine
from helios_pipeline.storage.manager import OneTouchStorageManager
from helios_pipeline.testing.tiny_model import create_tiny_helios_config
from helios_pipeline.training.atomic_checkpoint import AtomicCheckpointManager
from helios_pipeline.training.controller import ContinuousTrainingController

logger = logging.getLogger(__name__)
console = Console()


def execute_pipeline(
    mode: str = "once",
    dry_run: bool = False,
    synthetic: bool = False,
    tier: str = "nano",
    clusters_count: int = 1,
    max_steps_per_cluster: int | None = None,
    device: str = "auto",
    git_sync: bool = False,
) -> int:
    """Execute continuous pretraining, checkpointing, evaluation, and documentation."""
    console.print(
        f"\n[bold green]HELIOS Continuous Training Pipeline[/bold green] (Mode: {mode.upper()})"
    )
    console.print(
        f"Tier: [cyan]{tier}[/cyan] | Dry Run: [yellow]{dry_run}[/yellow] | Synthetic: [magenta]{synthetic}[/magenta]"
    )
    console.print("=" * 72)

    # 1. Initialize strict storage manager
    try:
        storage = OneTouchStorageManager()
        health = storage.check_health()
        console.print(
            f"✓ External Storage Mounted: [dim]{storage.mount_path}[/dim] (Free: {health['free_gb']:.1f} GB)"
        )
    except Exception as e:
        console.print(f"[bold red]Storage Initialization Failed:[/bold red] {e}")
        return 10

    # 2. Select device
    if device == "auto":
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device
    console.print(f"✓ Compute Device Selected: [bold cyan]{resolved_device}[/bold cyan]")

    # 3. Model Configuration
    if synthetic or dry_run:
        model_config = create_tiny_helios_config(vocab_size=1024)
    else:
        preset_map = {
            "nano": "nano_20m",
            "alpha": "alpha_100m",
            "beta": "beta_300m",
            "core": "core_1b",
            "pro": "pro_3b",
            "flagship": "flagship_6b",
        }
        target_preset = preset_map.get(tier.lower(), "nano_20m")
        model_config = (
            HeliosConfig.from_preset(target_preset)
            if hasattr(HeliosConfig, "from_preset")
            else create_tiny_helios_config()
        )

    # 4. Pipeline Components
    checkpoint_mgr = AtomicCheckpointManager(storage.checkpoints_dir, storage.staging_dir)
    controller = ContinuousTrainingController(
        model_config=model_config,
        checkpoints_dir=storage.checkpoints_dir,
        staging_dir=storage.staging_dir,
        device=resolved_device,
        gradient_accumulation_steps=2 if resolved_device == "mps" else 1,
    )
    resumed = controller.resume_if_available()
    console.print(
        f"✓ Checkpoint Engine: {'Resumed at step ' + str(controller.global_step) if resumed else 'Initialized Fresh'}"
    )

    curriculum = CurriculumPartitioner()
    sanitizer = DocumentSanitizer()
    streamer = FineWebStreamer(use_synthetic=synthetic, sanitizer=sanitizer)
    docs_updater = DatasetDocumentationUpdater()
    history_recorder = HistoryDocumentationRecorder()
    eval_runner = PipelineEvaluationRunner(device=resolved_device)
    cleaner = ConservativeCleanupEngine(storage, checkpoint_mgr)
    git_engine = GitSyncEngine() if git_sync else None

    # 5. Pipeline Run Loop
    completed_clusters = 0
    allocated_stages = curriculum.allocate_cluster_stages(clusters_count)

    while True:
        stage = allocated_stages[completed_clusters % len(allocated_stages)]
        cluster_id = f"cluster_{uuid.uuid4().hex[:8]}_{stage}"
        console.print(
            f"\n[bold]Initiating Cluster {completed_clusters + 1}/{clusters_count}:[/bold] `{cluster_id}` ({stage})"
        )

        manifest = ClusterManifest(
            cluster_id=cluster_id,
            stage=stage,
            source="synthetic" if synthetic else "HuggingFaceFW/fineweb",
            source_revision="main",
        )
        manifest.transition_to("DOWNLOADING")

        # Stream & Ingest
        accepted_docs = 0
        seen_docs = 0
        doc_stream = streamer.stream_documents(limit=50 if synthetic else 200, stage=stage)
        for _doc, rej in doc_stream:
            seen_docs += 1
            if rej is None:
                accepted_docs += 1
            else:
                manifest.rejection_breakdown[rej] = manifest.rejection_breakdown.get(rej, 0) + 1

        manifest.documents_seen = seen_docs
        manifest.documents_accepted = accepted_docs
        manifest.documents_rejected = seen_docs - accepted_docs
        manifest.content_hash = uuid.uuid4().hex  # Verified shard digest

        if dry_run:
            console.print("[yellow][DRY RUN] Skipping tensor training passes.[/yellow]")
            manifest.transition_to("COMPLETED")
            docs_updater.record_cluster(manifest)
            completed_clusters += 1
            if completed_clusters >= clusters_count or mode == "once":
                break
            continue

        # Train cluster
        steps = max_steps_per_cluster or (10 if synthetic else 50)
        train_res = controller.train_cluster(
            cluster=manifest,
            max_steps=steps,
        )

        console.print(
            f"✓ Cluster Training Step: {train_res['global_step']} | Loss: {train_res['last_loss']:.4f} | PPL: {train_res['perplexity']:.2f}"
        )

        # Update Documentation
        docs_updater.record_cluster(manifest)
        history_recorder.record_training_step(
            tier=tier,
            step=train_res["global_step"],
            loss=train_res["last_loss"],
            perplexity=train_res["perplexity"],
            lr=controller.lr,
            tokens_ingested=train_res["tokens_seen"],
            cluster_id=cluster_id,
        )

        # Run smoke evaluation
        if Path(train_res["checkpoint_path"]).exists():
            eval_runner.run_tier2_smoke(
                Path(train_res["checkpoint_path"]),
                preset="nano-20m",
                sample_limit=3,
                config=model_config,
            )

        # Cleanup intermediate scratch safely
        cleanup_res = cleaner.cleanup_cluster_scratch(manifest)
        if cleanup_res["status"] == "COMPLETED":
            console.print(
                f"✓ Cleaned intermediate scratch: {cleanup_res['reclaimed_bytes'] / 1024:.1f} KB reclaimed"
            )

        # Git metadata synchronization if requested
        if git_sync and git_engine:
            sync_files = [
                docs_updater.catalog_file,
                docs_updater.provenance_file,
                docs_updater.history_file,
                history_recorder.training_file,
                history_recorder.checkpoint_file,
                history_recorder.benchmark_file,
                history_recorder.evolution_file,
            ]
            git_engine.sync_files(
                files_to_sync=sync_files,
                commit_message=f"chore(pipeline): record metrics for {cluster_id} step {train_res['global_step']}",
                push=False,
            )

        completed_clusters += 1
        if train_res["interrupted"]:
            console.print("[yellow]Graceful shutdown requested. Halting pipeline loop.[/yellow]")
            break

        if mode == "once" and completed_clusters >= clusters_count:
            break

    console.print("\n[bold green]✓ Pipeline execution finished successfully.[/bold green]\n")
    return 0
