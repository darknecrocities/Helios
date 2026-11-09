"""Multi-Tier Deployment Script for Hugging Face Hub (Darknecrocities/Helios).

Updates Model Card with the empirical hardware matrix and competitive benchmark table.
Deploys configuration files, parameters, and dedicated branches:
- main (certified foundation model & complete family documentation)
- alpha (Helios Alpha 100M)
- beta (Helios Beta 300M)
- core (Helios Core 1.1B)
- pro (Helios Pro 3.3B)

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from helios.cli.utils import load_or_create_tokenizer
from helios.export.hf import (
    export_huggingface,
    export_progressive_tier_safetensors,
    generate_hf_config_dict,
)
from helios.model.config import HeliosConfig
from scripts.build_roadmap_models import ROADMAP_SPECS, calculate_param_counts
from scripts.deploy_huggingface import resolve_hf_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.deploy.hf_family")


def deploy_family_to_huggingface(repo_id: str = "Darknecrocities/Helios") -> list[str]:
    """Deploy updated model card, configurations, and branches to Hugging Face Hub."""
    token = resolve_hf_token()
    api = HfApi(token=token)

    # 1. Update main branch Model Card
    readme_path = Path("deploy/huggingface/README.md").resolve()
    logger.info("Uploading updated Model Card to %s (main)...", repo_id)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="docs(hf): update audited benchmark matrix, model-index, and multi-capability chat template docs",
        token=token,
    )

    deployed_tiers = ["main"]

    # 2. Upload model family configs into models/ subdirectories on main & dedicated branches
    for spec in ROADMAP_SPECS:
        preset = spec["preset"]
        tag = spec["tag"]
        cfg = HeliosConfig.from_preset(preset)
        cfg.developer = "Arron Kian Parejas"
        cfg.developer_email = "parejasarronkian@gmail.com"

        param_stats = calculate_param_counts(cfg)
        hf_cfg_dict = generate_hf_config_dict(cfg)
        tier_manifest = {
            "model_name": spec["name"],
            "preset": preset,
            "tag": tag,
            "target_size": spec["param_target"],
            "hardware_target": spec["hw_target"],
            "developer": "Arron Kian Parejas",
            "developer_email": "parejasarronkian@gmail.com",
            "parameters": param_stats,
            "config": hf_cfg_dict,
        }

        config_bytes = json.dumps(hf_cfg_dict, indent=2).encode("utf-8")
        manifest_bytes = json.dumps(tier_manifest, indent=2).encode("utf-8")

        logger.info("Uploading architecture specs for %s (%s) on main...", spec["name"], tag)
        api.upload_file(
            path_or_fileobj=config_bytes,
            path_in_repo=f"models/{tag}/config.json",
            repo_id=repo_id,
            commit_message=f"feat(hf): update {spec['name']} architecture configuration",
            token=token,
        )
        api.upload_file(
            path_or_fileobj=manifest_bytes,
            path_in_repo=f"models/{tag}/manifest.json",
            repo_id=repo_id,
            commit_message=f"feat(hf): update {spec['name']} parameter manifest",
            token=token,
        )

        # Create branch if not exists
        try:
            api.create_branch(repo_id=repo_id, branch=tag, token=token)
            logger.info("✓ Created branch '%s' on %s", tag, repo_id)
        except Exception:
            logger.info("Branch '%s' already exists on %s", tag, repo_id)

        # Export full tier package (SafeTensors, config, capability tokenizer) and upload to branch
        ckpt_dir = Path(f"/Volumes/One Touch/Helios/checkpoints/{tag}_continuous/latest")
        if not ckpt_dir.exists():
            ckpt_dir = Path("checkpoints/helios_core_pretrained").resolve()

        with tempfile.TemporaryDirectory(prefix=f"helios_hf_{tag}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            tok_dir = (
                ckpt_dir
                if (ckpt_dir / "tokenizer.json").exists()
                else Path("checkpoints/helios_core_pretrained").resolve()
            )
            tok = load_or_create_tokenizer(tok_dir)
            export_huggingface(
                output_dir=tmp_path,
                config=cfg,
                model=None,
                tokenizer=tok,
                checkpoint_path=ckpt_dir,
                repo_id=repo_id,
                generate_card=False,
            )
            # Ensure progressive tier safetensors matches exact tied parameter count for all tiers
            export_progressive_tier_safetensors(
                config=cfg,
                output_path=tmp_path / "model.safetensors",
                source_checkpoint=ckpt_dir,
                include_untied_lm_head=False,
            )
            # Copy curated README.md to branch as well
            (tmp_path / "README.md").write_text(
                readme_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            # Remove duplicate helios_model.safetensors on branch if present
            try:
                api.delete_file(
                    path_in_repo="helios_model.safetensors",
                    repo_id=repo_id,
                    revision=tag,
                    commit_message=f"chore({tag}): remove duplicate helios_model.safetensors",
                    token=token,
                )
            except Exception:
                pass

            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(tmp_path),
                revision=tag,
                commit_message=f"feat({tag}): deploy {spec['name']} ({param_stats['tied_million']}M params) SafeTensors & capability chat template",
                token=token,
            )
            logger.info("✓ Uploaded %s package to branch '%s'", spec["name"], tag)

        deployed_tiers.append(tag)

    logger.info("✓ Successfully deployed family architecture and branches to %s", repo_id)
    return deployed_tiers


if __name__ == "__main__":
    tiers = deploy_family_to_huggingface()
    print(f"\n🎉 Deployed tiers to Hugging Face Hub: {', '.join(tiers)}")
