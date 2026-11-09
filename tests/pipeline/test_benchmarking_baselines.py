"""Unit tests for benchmarking baselines registry and history documentation recorder."""

from pathlib import Path
from tempfile import TemporaryDirectory

from helios_pipeline.evaluation.baselines import BaselineManager
from helios_pipeline.evaluation.history_recorder import HistoryDocumentationRecorder


def test_baseline_registry_completeness() -> None:
    """Verify standard open-weight baselines are registered with accurate specs."""
    manager = BaselineManager()
    baselines = manager.list_baselines()
    assert len(baselines) >= 5

    keys = [b.name for b in baselines]
    assert any("Qwen2.5" in k for k in keys)
    assert any("Llama 3.2 1B" in k for k in keys)
    assert any("Llama 3.2 3B" in k for k in keys)
    assert any("Gemma 2 2B" in k for k in keys)
    assert any("DeepSeek R1" in k for k in keys)

    qwen = manager.get_baseline("qwen2.5_0.5b")
    assert qwen is not None
    assert "huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct" in qwen.acquisition_command
    assert qwen.benchmarks.get("mmlu") == 0.554


def test_history_documentation_recorder() -> None:
    """Verify append-only updates across historical audit markdown documents."""
    with TemporaryDirectory() as tmp_dir:
        docs_path = Path(tmp_dir) / "docs"
        recorder = HistoryDocumentationRecorder(docs_dir=docs_path)

        assert recorder.training_file.exists()
        assert recorder.checkpoint_file.exists()
        assert recorder.benchmark_file.exists()
        assert recorder.evolution_file.exists()

        recorder.record_training_step(
            tier="nano-20m",
            step=100,
            loss=2.4567,
            perplexity=11.66,
            lr=5e-4,
            tokens_ingested=204800,
            cluster_id="cluster_001",
        )
        content_train = recorder.training_file.read_text(encoding="utf-8")
        assert (
            "| `nano-20m` | 100 | 2.4567 | 11.66 | 5.00e-04 | 204,800 | `cluster_001` |"
            in content_train
        )

        recorder.record_checkpoint(
            checkpoint_id="ckpt_step_100",
            tier="nano-20m",
            step=100,
            loss=2.4567,
            size_mb=82.5,
            sha256_digest="abcdef1234567890abcdef1234567890",
            status="VERIFIED",
        )
        content_ckpt = recorder.checkpoint_file.read_text(encoding="utf-8")
        assert (
            "| `ckpt_step_100` | `nano-20m` | 100 | 2.4567 | 82.5 | `abcdef1234567890...` | VERIFIED |"
            in content_ckpt
        )
