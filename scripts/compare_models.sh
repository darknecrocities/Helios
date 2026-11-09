#!/usr/bin/env bash
# HELIOS Checkpoint & Baseline Comparison Runner
set -euo pipefail

CANDIDATE="${1:?Error: Provide path to candidate manifest JSON as arg 1}"
BASELINE="${2:-}"

if [[ -n "${BASELINE}" ]]; then
    python -m evaluation.cli compare "${CANDIDATE}" --baseline "${BASELINE}"
else
    python -m evaluation.cli compare "${CANDIDATE}" --baselines-file baselines/model_registry.yaml
fi
