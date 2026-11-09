#!/usr/bin/env bash
# HELIOS Smoke Test Runner (1-Sample Per Task Fast Verification)
set -euo pipefail

CHECKPOINT="${1:-configs/models/nano.yaml}"
DEVICE="cpu"

if [[ "$(uname)" == "Darwin" ]]; then
    DEVICE="mps"
elif command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
fi

echo "=================================================="
echo " Launching HELIOS Smoke Test Verification"
echo " Target Checkpoint: ${CHECKPOINT}"
echo " Device:            ${DEVICE}"
echo "=================================================="

python -m evaluation.cli smoke "${CHECKPOINT}" --device "${DEVICE}"
