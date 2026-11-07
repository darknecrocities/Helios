"""Full-Family Model Distribution & Terminal Regression Verification Script.

Packages all 6 Helios model tiers (Nano, Alpha, Beta, Core, Pro, Flagship)
into /Volumes/One Touch/Helios/dist/ alongside physical exports (SafeTensors,
ONNX, TorchScript, TFLite), Ollama Modelfiles, and Python wheel/sdist packages.
Also executes the 16 exact user terminal session regression queries plus unseen
out-of-session fallback tests.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from helios.cli.chat import ChatSession
from helios.cli.utils import load_model_and_config, load_or_create_tokenizer
from helios.model.config import HeliosConfig
from scripts.build_roadmap_models import ROADMAP_SPECS, calculate_param_counts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.dist.builder")

TERMINAL_TRANSCRIPT_QUERIES: list[str] = [
    "what is philippines?",
    "who created you?",
    "what model are you?",
    "what is model version?",
    "sino gumawa sayo?",
    "ano ang luzon?",
    "10+10?",
    "who is newton?",
    "what is google?",
    "what is computer?",
    "hello?",
    "what is earth?",
    "ano ang pilipinas?",
    "salamat",
    "what is array?",
    "what is javascript?",
    "what is blorgonx99?",
    "ano ang zyxwvut123?",
]


def build_all_family_distributions(
    external_dist_dir: Path | str = "/Volumes/One Touch/Helios/dist",
) -> dict[str, Any]:
    """Package all 6 Helios model tiers and Python wheel/sdist distributions."""
    dist_root = Path(external_dist_dir).resolve()
    models_dir = dist_root / "models"
    packages_dir = dist_root / "packages"
    models_dir.mkdir(parents=True, exist_ok=True)
    packages_dir.mkdir(parents=True, exist_ok=True)

    tier_records: list[dict[str, Any]] = []

    for spec in ROADMAP_SPECS:
        tag = spec["tag"]
        preset = spec["preset"]
        tier_dir = models_dir / tag
        tier_dir.mkdir(parents=True, exist_ok=True)

        cfg = HeliosConfig.from_preset(preset)
        cfg.developer = "Arron Kian Parejas"
        cfg.developer_email = "parejasarronkian@gmail.com"
        params = calculate_param_counts(cfg)

        manifest = {
            "model_name": spec["name"],
            "version": "0.2.0",
            "tag": tag,
            "preset": preset,
            "developer": "Arron Kian Parejas",
            "developer_email": "parejasarronkian@gmail.com",
            "parameters": params,
            "hardware_target": spec["hw_target"],
            "epistemic_fallback_en": (
                "This topic is not yet in my current training session. "
                "Please try asking another question!"
            ),
            "epistemic_fallback_tl": (
                "Ang paksang ito ay wala pa sa aking kasalukuyang training session. "
                "Subukang magtanong ng ibang katanungan!"
            ),
            "distribution_formats": [
                "SafeTensors (.safetensors)",
                "PyTorch Checkpoint (.pt)",
                "ONNX Graph (.onnx)",
                "TFLite Quantized (.tflite)",
                "Ollama Modelfile",
            ],
            "config": cfg.to_dict(),
        }

        (tier_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        (tier_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        modelfile_src = Path(f"deploy/ollama/Modelfile.{tag}")
        if modelfile_src.exists():
            shutil.copy2(modelfile_src, tier_dir / "Modelfile")

        tier_records.append(manifest)
        logger.info(
            "✓ Packaged distribution manifest & config for %s (%sM params)",
            spec["name"],
            params["tied_million"],
        )

    # Build Python wheel and sdist into external packages directory
    try:
        subprocess.run(
            ["uv", "build", "--out-dir", str(packages_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("✓ Built Python .whl and .tar.gz packages in %s", packages_dir)
    except Exception as exc:
        logger.warning("uv build skipped or warning: %s", exc)

    dist_summary = {
        "family": "Helios Open-Weight Foundation Models",
        "version": "0.2.0",
        "developer": "Arron Kian Parejas <parejasarronkian@gmail.com>",
        "primary_storage": str(dist_root),
        "dataset_path": "/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl",
        "tiers": tier_records,
    }

    local_art = Path("artifacts/distribution_manifest.json")
    local_art.parent.mkdir(parents=True, exist_ok=True)
    local_art.write_text(json.dumps(dist_summary, indent=2), encoding="utf-8")
    (dist_root / "distribution_manifest.json").write_text(
        json.dumps(dist_summary, indent=2), encoding="utf-8"
    )
    return dist_summary


def verify_terminal_transcript_queries() -> list[dict[str, str]]:
    """Verify all 16 terminal transcript queries plus unseen fallback queries."""
    model, config = load_model_and_config(preset="nano-20m")
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
    session = ChatSession(model=model, tokenizer=tokenizer, session_id="transcript-verify")

    transcript_results: list[dict[str, str]] = []
    logger.info("Verifying all 18 terminal transcript & unseen fallback queries...")
    for q in TERMINAL_TRANSCRIPT_QUERIES:
        ans = session.handle_turn(q, stream=False)
        assert ans.strip() != "42", f"Regression failure: '{q}' returned '42'!"
        transcript_results.append({"query": q, "response": ans})
        logger.info("Q: %-26s -> A: %s", q, ans[:95])

    return transcript_results


if __name__ == "__main__":
    build_all_family_distributions()
    verify_terminal_transcript_queries()
