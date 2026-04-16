#!/bin/bash
# Run the cross-axis capping experiment on Gemma-2-27B (single GPU).
#
# Gemma-2-27B has 46 layers. Cap layers 33-39 target the upper quarter
# (72-85% depth), matching the same relative range used for Qwen3-32B.
#
# Usage:
#   chmod +x run_gemma.sh
#   ./run_gemma.sh              # full run (250 jailbreak + 100 benign)
#   ./run_gemma.sh sanity       # smoke test (5 jailbreak + 10 benign)

set -e

PRESET="${1:-full}"
MODEL="google/gemma-2-27b-it"
CAP_LAYERS="33-39"
OUTPUT_DIR="results/crosscap_gemma_${PRESET}"

echo "============================================"
echo "  Cross-Axis Capping -- Gemma-2-27B"
echo "  Preset:     ${PRESET}"
echo "  Cap layers: L${CAP_LAYERS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "============================================"
echo ""

python run_crosscap.py \
    --preset "$PRESET" \
    --model "$MODEL" \
    --cap-layers "$CAP_LAYERS" \
    --output-dir "$OUTPUT_DIR"
