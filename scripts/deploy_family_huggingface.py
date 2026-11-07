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
from pathlib import Path

from huggingface_hub import HfApi

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
        commit_message="docs: update empirical hardware matrix and competitive benchmark table",
        token=token,
    )

    deployed_tiers = ["main"]

    # 2. Upload model family configs into models/ subdirectories on main
    from helios.model.config import HeliosConfig

    for spec in ROADMAP_SPECS:
        preset = spec["preset"]
        tag = spec["tag"]
        cfg = HeliosConfig.from_preset(preset)
        cfg.developer = "Arron Kian Parejas"
        cfg.developer_email = "parejasarronkian@gmail.com"

        param_stats = calculate_param_counts(cfg)
        tier_manifest = {
            "model_name": spec["name"],
            "preset": preset,
            "tag": tag,
            "target_size": spec["param_target"],
            "hardware_target": spec["hw_target"],
            "developer": "Arron Kian Parejas",
            "developer_email": "parejasarronkian@gmail.com",
            "parameters": param_stats,
            "config": cfg.to_dict(),
        }

        # Upload config to models/<tag>/config.json
        config_bytes = json.dumps(cfg.to_dict(), indent=2).encode("utf-8")
        manifest_bytes = json.dumps(tier_manifest, indent=2).encode("utf-8")

        logger.info("Uploading architecture specs for %s (%s)...", spec["name"], tag)
        api.upload_file(
            path_or_fileobj=config_bytes,
            path_in_repo=f"models/{tag}/config.json",
            repo_id=repo_id,
            commit_message=f"feat: add {spec['name']} architecture configuration",
            token=token,
        )
        api.upload_file(
            path_or_fileobj=manifest_bytes,
            path_in_repo=f"models/{tag}/manifest.json",
            repo_id=repo_id,
            commit_message=f"feat: add {spec['name']} parameter manifest",
            token=token,
        )

        # Create branch if not exists
        try:
            api.create_branch(repo_id=repo_id, branch=tag, token=token)
            logger.info("✓ Created branch '%s' on %s", tag, repo_id)
        except Exception:
            logger.info("Branch '%s' already exists on %s", tag, repo_id)

        deployed_tiers.append(tag)

    logger.info("✓ Successfully deployed family architecture and branches to %s", repo_id)
    return deployed_tiers


if __name__ == "__main__":
    tiers = deploy_family_to_huggingface()
    print(f"\n🎉 Deployed tiers to Hugging Face Hub: {', '.join(tiers)}")
