#!/bin/bash
# Run the cross-axis capping experiment on Qwen3-32B (single GPU).
#
# Assistant-axis capping uses the original paper's exact vectors and
# thresholds from capping_config.pt (experiment: layers_46:54-p0.25).
#
# Usage:
#   chmod +x run_qwen.sh
#   ./run_qwen.sh                                # full run, defaults
#   ./run_qwen.sh sanity                         # smoke test (10 jailbreak + 10 benign)
#   ./run_qwen.sh full optimal                   # full run with midpoint threshold
#   ./run_qwen.sh full p25                       # full run with 25th percentile threshold
#   ./run_qwen.sh full mean+std benign-p1        # full run, tighter cross-cap detect gate
#
# Compliance threshold options:  mean+std (default), optimal, mean, p25
# Cross-detect method options:   benign-p1, benign-p5 (default), benign-p10, midpoint

set -e

PRESET="${1:-full}"
THRESHOLD="${2:-mean+std}"
CROSS_DETECT="${3:-benign-p1}"
OUTPUT_DIR="results/crosscap_qwen_${PRESET}_${THRESHOLD}_${CROSS_DETECT}"

echo "============================================"
echo "  Cross-Axis Capping -- Qwen3-32B"
echo "  Preset:               ${PRESET}"
echo "  Compliance threshold: ${THRESHOLD}"
echo "  Cross-detect method:  ${CROSS_DETECT}"
echo "  Output:               ${OUTPUT_DIR}"
echo "============================================"
echo ""

python run_crosscap.py \
    --preset "$PRESET" \
    --compliance-threshold "$THRESHOLD" \
    --cross-detect-method "$CROSS_DETECT" \
    --output-dir "$OUTPUT_DIR"
