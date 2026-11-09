#!/usr/bin/env bash
# HELIOS Local Evaluation Runner (macOS MPS / CPU)
set -euo pipefail

CHECKPOINT="${1:-configs/models/nano.yaml}"
TASKS="${2:-all}"
DEVICE="cpu"

# Detect macOS Apple Silicon MPS
if [[ "$(uname)" == "Darwin" ]]; then
    DEVICE="mps"
fi

echo "=================================================="
echo " Launching HELIOS Local Evaluation"
echo " Target Checkpoint: ${CHECKPOINT}"
echo " Suite:             ${TASKS}"
echo " Device:            ${DEVICE}"
echo "=================================================="

python -m evaluation.cli run "${CHECKPOINT}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}" \
    --output-dir "evaluation_records" \
    --seed 42
