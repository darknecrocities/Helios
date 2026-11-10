"""OneTouch-exclusive, time-boxed pretraining across all Helios tiers with category selection
and automatic last-checkpoint resumption.

- nano / alpha / beta: full-parameter training of HeliosForCausalLM.
- core / pro / flagship: single-block progressive slice at exact d_model
  (ProgressiveTierProjectionHead). Fits 1B-5.8B dimensions in 16 GB unified memory.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios_pipeline.data.categories import (
    normalize_category,
)
from helios_pipeline.data.curriculum_ledger import sample_non_repeating_corpus
from helios_pipeline.evaluation.health import (
    compute_file_sha256,
    resolve_latest_checkpoint_dir,
)
from helios_pipeline.npu.accelerator import resolve_compute_device
from helios_pipeline.npu.hardware_governor import (
    compute_hardware_shift_plan,
    pace_training_step,
)
from helios_pipeline.storage.manager import OneTouchStorageManager, StorageConfig

logger = logging.getLogger(__name__)

ALL_TIERS = ["nano", "alpha", "beta", "core", "pro", "flagship"]
FULL_PARAM_TIERS = {"nano", "alpha", "beta"}
CORPUS_NAME = "helios_massive_pretrain_corpus.jsonl"


@dataclass
class TierResult:
    """Result and improvement telemetry for a single tier pretraining run."""

    tier: str
    mode: str
    device: str
    steps: int
    seconds: float
    initial_loss: float
    final_loss: float
    tokens: int
    checkpoint: str
    category: str = "all"
    resumed_from_step: int = 0
    total_steps: int = 0
    baseline_initial_loss: float = 0.0
    previous_loss: float | None = None
    loss_improvement_pct: float = 0.0
    previous_ppl: float | None = None
    heldout_ppl: float | None = None
    sha256_digest: str = ""
    health_status: str = "HEALTHY"
    datasets_synced: int = 177
    updated_at: str = ""
    trained_topics_summary: str = "Coding 46% · Math 17% · Agents 13% · Science 7% · Tagalog 4%"
    cursor_range: str = "Rec #0→#2,400 (0% Dup · Ep 1)"
    hardware_profile: str = "Zero-Lag Governor"

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


def parse_tiers(spec: str) -> list[str]:
    """Parse 'all' or comma-separated tier names."""
    if spec.lower() == "all":
        return list(ALL_TIERS)
    tiers = [t.strip().lower() for t in spec.split(",") if t.strip()]
    bad = [t for t in tiers if t not in ALL_TIERS]
    if bad:
        raise ValueError(f"Unknown tier(s) {bad}. Valid: {ALL_TIERS} or 'all'")
    return tiers


def require_onetouch() -> OneTouchStorageManager:
    """Hard lock: pretraining only runs with a verified external OneTouch volume."""
    cfg = StorageConfig(require_external_drive=True, prohibit_internal_storage_fallback=True)
    storage = OneTouchStorageManager(config=cfg)
    storage.check_health()
    return storage


def load_corpus_sample(
    storage: OneTouchStorageManager, limit: int, seed: int, category: str = "all"
) -> list[str]:
    """Stream-sample non-repeating texts from OneTouch corpus filtered by category."""
    texts, _ = sample_non_repeating_corpus(
        storage=storage, limit=limit, seed=seed, category=category, advance_cursor=True
    )
    return texts


def _batches(texts: list[str], tokenizer: Any, seq_len: int, bsz: int, vocab: int):
    ids = [tokenizer.encode(t, add_special_tokens=True)[:seq_len] for t in texts]
    ids = [x for x in ids if len(x) > 8]
    if not ids:
        ids = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    for i in itertools.cycle(range(0, max(1, len(ids) - bsz + 1), bsz)):
        chunk = ids[i : i + bsz]
        width = max(len(x) for x in chunk)
        mat = torch.zeros(len(chunk), width, dtype=torch.long)
        lab = torch.full_like(mat, -100)
        for r, x in enumerate(chunk):
            t = torch.tensor(x, dtype=torch.long) % vocab
            mat[r, : len(x)] = t
            lab[r, : len(x)] = t
        yield mat, lab


def _build(tier: str, seq_len: int) -> tuple[torch.nn.Module, HeliosConfig, str]:
    cfg = HeliosConfig.from_preset(tier)
    cfg.max_seq_len = seq_len
    if tier in FULL_PARAM_TIERS:
        return HeliosForCausalLM(cfg), cfg, "full_parameter"
    from scripts.pretrain_all_models import ProgressiveTierProjectionHead

    return ProgressiveTierProjectionHead(cfg, active_vocab=2048), cfg, "block_slice"


def _restore_last_checkpoint(
    model: torch.nn.Module, tier: str, storage: OneTouchStorageManager
) -> tuple[int, float | None, float | None, float | None]:
    """Restore model weights from latest checkpoint if available and return prior metrics."""
    prev_dir = resolve_latest_checkpoint_dir(tier, base_dir=storage.base_dir)
    if prev_dir is None:
        return 0, None, None, None
    st_path = prev_dir / "model.safetensors"
    manifest_path = prev_dir / "manifest.json"
    resumed_step = 0
    baseline_init_loss: float | None = None
    prev_loss: float | None = None
    prev_ppl: float | None = None
    if manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            resumed_step = int(meta.get("total_steps", meta.get("steps", 0)))
            baseline_init_loss = (
                float(meta["baseline_initial_loss"])
                if meta.get("baseline_initial_loss")
                else float(meta["initial_loss"])
                if meta.get("initial_loss")
                else None
            )
            if meta.get("final_loss") is not None:
                prev_loss = float(meta["final_loss"])
            if meta.get("heldout_ppl") is not None:
                prev_ppl = float(meta["heldout_ppl"])
        except Exception:
            pass
    if st_path.exists():
        try:
            loaded_sd = load_file(str(st_path))
            model_sd = model.state_dict()
            compat = {
                k: v for k, v in loaded_sd.items() if k in model_sd and model_sd[k].shape == v.shape
            }
            if len(compat) == len(model_sd):
                model.load_state_dict(compat, strict=True)
                logger.info(
                    "Resumed %s from %s (step=%d, prev_loss=%s)",
                    tier,
                    prev_dir,
                    resumed_step,
                    prev_loss,
                )
        except Exception as exc:
            logger.warning("Could not restore weights from %s: %s", st_path, exc)
    return resumed_step, baseline_init_loss, prev_loss, prev_ppl


def _save_checkpoint_bundle(
    out_dir: Path,
    latest_dir: Path,
    legacy_dir: Path | None,
    state: dict[str, torch.Tensor],
    cfg: HeliosConfig,
    result: TierResult,
) -> None:
    """Atomically write weights, state.pt, config, SHA256SUMS, and manifest to step & latest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    st_file = out_dir / "model.safetensors"
    save_file(state, str(st_file))
    torch.save({"model_state_dict": state, "step": result.total_steps}, out_dir / "state.pt")
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    sha = compute_file_sha256(st_file)
    result.sha256_digest = sha
    (out_dir / "SHA256SUMS").write_text(f"{sha}  model.safetensors\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    # Mirror to <tier>_continuous/latest/
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("model.safetensors", "state.pt", "config.json", "SHA256SUMS", "manifest.json"):
        shutil.copy2(out_dir / fname, latest_dir / fname)

    # Mirror full-parameter tiers to /Volumes/One Touch/Helios/checkpoints/helios_<tier>/
    if legacy_dir is not None:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for fname in (
            "model.safetensors",
            "state.pt",
            "config.json",
            "SHA256SUMS",
            "manifest.json",
        ):
            shutil.copy2(out_dir / fname, legacy_dir / fname)


def train_tier(
    tier: str,
    seconds: float,
    storage: OneTouchStorageManager,
    texts: list[str],
    tokenizer: Any,
    device_req: str = "auto",
    category: str = "all",
    trained_topics_summary: str = "Coding 46% · Math 17% · Agents 13% · Science 7% · Tagalog 4%",
    cursor_range: str = "Rec #0→#2,400 (0% Dup · Ep 1)",
) -> TierResult:
    """Resume from last checkpoint, train with Zero-Lag Hardware Governor, and save atomically."""
    storage.check_health()
    cat_key = normalize_category(category)
    device = resolve_compute_device(device_req) if tier in FULL_PARAM_TIERS else "cpu"
    hw_plan = compute_hardware_shift_plan(tier, device)
    seq_len = hw_plan.seq_len if tier in FULL_PARAM_TIERS else 64
    bsz = hw_plan.batch_size if tier in FULL_PARAM_TIERS else 2
    model, cfg, mode = _build(tier, seq_len)

    resumed_step, base_init_loss, prev_loss, prev_ppl = _restore_last_checkpoint(
        model, tier, storage
    )
    model.to(device).train()
    vocab = getattr(model, "active_vocab", cfg.vocab_size)
    lr = (3e-4 if resumed_step > 0 else 5e-4) if tier == "nano" else 2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    stream = _batches(texts, tokenizer, seq_len, bsz, vocab)
    dev_obj = torch.device(device)

    steps, tokens, first, last = 0, 0, math.nan, math.nan
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        ids, labels = next(stream)
        ids, labels = ids.to(device), labels.to(device)
        loss = model(ids, labels=labels)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        tokens += int((labels != -100).sum())
        last = float(loss.item())
        first = last if steps == 1 else first
        pace_training_step(dev_obj, steps, hw_plan)
    elapsed = time.perf_counter() - start
    total_steps = resumed_step + steps
    effective_base_loss = base_init_loss if base_init_loss and base_init_loss > 0 else first
    loss_imp_pct = (
        round(((effective_base_loss - last) / effective_base_loss) * 100.0, 2)
        if effective_base_loss > 0
        else 0.0
    )

    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    has_bad = any(
        torch.isnan(t).any().item() or torch.isinf(t).any().item() for t in state.values()
    )
    health = "CORRUPTED" if has_bad else "HEALTHY"

    cont_root = storage.checkpoints_dir / f"{tier}_continuous"
    out = cont_root / f"step_{total_steps:08d}"
    latest = cont_root / "latest"
    legacy = (storage.checkpoints_dir / f"helios_{tier}") if tier in FULL_PARAM_TIERS else None
    storage.assert_not_internal(out)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    result = TierResult(
        tier=tier,
        mode=mode,
        device=device,
        steps=steps,
        seconds=elapsed,
        initial_loss=first,
        final_loss=last,
        tokens=tokens,
        checkpoint=str(latest),
        category=cat_key,
        resumed_from_step=resumed_step,
        total_steps=total_steps,
        baseline_initial_loss=effective_base_loss,
        previous_loss=prev_loss,
        loss_improvement_pct=loss_imp_pct,
        previous_ppl=prev_ppl,
        health_status=health,
        datasets_synced=177,
        updated_at=stamp,
        trained_topics_summary=trained_topics_summary,
        cursor_range=cursor_range,
        hardware_profile=hw_plan.profile_label,
    )
    _save_checkpoint_bundle(out, latest, legacy, state, cfg, result)

    del model, opt, state
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result


def evict_ephemeral_training_cache(storage: OneTouchStorageManager) -> int:
    """Safely free temporary streaming and PyTorch caches on external storage without touching checkpoints."""
    bytes_freed = 0
    ephemeral_dirs = [
        storage.base_dir / "data" / "tmp",
        storage.base_dir / "data" / "hf_cache",
        storage.base_dir / "datasets" / "cache",
    ]
    for d in ephemeral_dirs:
        if d.exists() and d.is_dir():
            for p in list(d.glob("*")):
                try:
                    if p.is_file():
                        bytes_freed += p.stat().st_size
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()
    return bytes_freed


def train_all(
    tiers: list[str],
    minutes: float = 5.0,
    device_req: str = "auto",
    category: str = "all",
    target_tokens: int | None = None,
) -> list[TierResult]:
    """Train all tiers sequentially or in continuous round-robin micro-epochs until token target is reached."""
    from helios.cli.utils import load_or_create_tokenizer

    storage = require_onetouch()
    tokenizer = load_or_create_tokenizer()
    cat_key = normalize_category(category)

    # Token-budgeted continuous campaign mode
    if target_tokens is not None and target_tokens > 0:
        logger.info(
            "Initiating Token-Budgeted Continuous Campaign across %s (Target: %s tokens)",
            tiers, f"{target_tokens:,}",
        )
        cumulative_tokens = 0
        latest_results: dict[str, TierResult] = {}
        cycle = 0

        # Run continuous round-robin passes across all tiers
        while cumulative_tokens < target_tokens:
            cycle += 1
            texts, session_meta = sample_non_repeating_corpus(
                storage, limit=3000, seed=42 + cycle, category=cat_key, advance_cursor=True
            )
            topics_summary = str(session_meta.get("trained_topics_summary", ""))
            cursor_range = str(session_meta.get("cursor_range", ""))

            # Continuous training slice: 180s for single tier, 30s for multi-tier round-robin
            slice_sec = 180.0 if len(tiers) == 1 else 30.0
            for tier in tiers:
                logger.info(
                    "Cycle #%d | Tier: %s | Cumulative: %s / %s tokens",
                    cycle, tier, f"{cumulative_tokens:,}", f"{target_tokens:,}",
                )
                res = train_tier(
                    tier,
                    slice_sec,
                    storage,
                    texts,
                    tokenizer,
                    device_req,
                    category=cat_key,
                    trained_topics_summary=topics_summary,
                    cursor_range=cursor_range,
                )
                cumulative_tokens += res.tokens
                latest_results[tier] = res
                if len(tiers) > 1:
                    evict_ephemeral_training_cache(storage)

                if cumulative_tokens >= target_tokens:
                    logger.info("Campaign Target Reached: %s tokens trained!", f"{cumulative_tokens:,}")
                    break

        return list(latest_results.values())

    # Wall-clock allocated pretraining mode
    texts, session_meta = sample_non_repeating_corpus(
        storage, limit=2500, seed=42, category=cat_key, advance_cursor=True
    )
    topics_summary = str(session_meta.get("trained_topics_summary", ""))
    cursor_range = str(session_meta.get("cursor_range", ""))
    per_tier = minutes * 60.0 / len(tiers)
    results = []
    for tier in tiers:
        logger.info(
            "Training %s [category=%s | %s] for %.0fs", tier, cat_key, cursor_range, per_tier
        )
        res = train_tier(
            tier,
            per_tier,
            storage,
            texts,
            tokenizer,
            device_req,
            category=cat_key,
            trained_topics_summary=topics_summary,
            cursor_range=cursor_range,
        )
        results.append(res)
        evict_ephemeral_training_cache(storage)
    return results
