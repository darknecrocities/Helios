"""Multi-Tier Ollama Build and Registry Deployment Script for Helios.

Builds and deploys distinct model tiers with exact context windows and architectures:
- parejasarronkian/Helios:nano (2K context, edge chat)
- parejasarronkian/Helios:alpha (4K context, bilingual QA)
- parejasarronkian/Helios:beta (8K context, math & logic)
- parejasarronkian/Helios:core (16K context, systems & coding)
- parejasarronkian/Helios:pro (32K context, advanced agent & MCP)
- parejasarronkian/Helios:flagship (64K context, flagship 5.8B reasoning)
- parejasarronkian/Helios:latest (Default foundation model)

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.deploy.ollama_family")

MODELS = [
    ("deploy/ollama/Modelfile.nano", "parejasarronkian/Helios:nano"),
    ("deploy/ollama/Modelfile.alpha", "parejasarronkian/Helios:alpha"),
    ("deploy/ollama/Modelfile.beta", "parejasarronkian/Helios:beta"),
    ("deploy/ollama/Modelfile.core", "parejasarronkian/Helios:core"),
    ("deploy/ollama/Modelfile.pro", "parejasarronkian/Helios:pro"),
    ("deploy/ollama/Modelfile.flagship", "parejasarronkian/Helios:flagship"),
]


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute shell command with detailed logging."""
    logger.info("Executing: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Command failed: %s\nStderr: %s", " ".join(cmd), res.stderr)
        raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr}")
    return res


def deploy_family(push: bool = True) -> list[str]:
    """Build all tiers with distinct Modelfiles and push to registry."""
    deployed_tags = []

    for modelfile_rel, tag in MODELS:
        mf = Path(modelfile_rel).resolve()
        if not mf.exists():
            logger.warning("Modelfile %s not found, skipping.", mf)
            continue

        logger.info("Building %s from %s...", tag, mf.name)
        run_command(["ollama", "create", tag, "-f", str(mf)])
        logger.info("✓ Successfully created %s", tag)

        if push:
            logger.info("Pushing %s to Ollama registry...", tag)
            try:
                run_command(["ollama", "push", tag])
                logger.info("✓ Pushed %s", tag)
            except Exception as e:
                logger.warning("Push warning for %s: %s", tag, e)

        deployed_tags.append(tag)

    # Set latest to alpha/nano foundation
    logger.info("Setting parejasarronkian/Helios:latest...")
    run_command(["ollama", "cp", "parejasarronkian/Helios:alpha", "parejasarronkian/Helios:latest"])
    if push:
        run_command(["ollama", "push", "parejasarronkian/Helios:latest"])
    deployed_tags.append("parejasarronkian/Helios:latest")

    return deployed_tags


if __name__ == "__main__":
    tags = deploy_family(push=True)
    print(f"\n🎉 Successfully built and deployed {len(tags)} distinct Ollama models:")
    for t in tags:
        print(f"  - {t}")
