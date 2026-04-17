#!/bin/bash
# Run the cross-axis capping experiment on Llama-3.3-70B-Instruct.
#
# Assistant-axis capping uses the original paper's exact vectors and
# thresholds from capping_config.pt (experiment: layers_56:72-p0.25).
# cap_layers are read directly from that config, so --cap-layers is
# not needed.
#
# Note: Llama-3.3-70B in bf16 needs ~140 GB of GPU memory. device_map="auto"
# will spread layers across visible GPUs; make sure CUDA_VISIBLE_DEVICES
# exposes enough of them (typically 2+ H100/A100-80GB or 4+ smaller).
#
# Usage:
#   chmod +x run_llama.sh
#   ./run_llama.sh                          # full run, default threshold (mean+std)
#   ./run_llama.sh sanity                   # smoke test (10 jailbreak + 10 benign)
#   ./run_llama.sh full optimal             # full run with midpoint threshold
#   ./run_llama.sh full p25                 # full run with 25th percentile threshold
#
# Compliance threshold options: mean+std (default), optimal, mean, p25

set -e

PRESET="${1:-full}"
THRESHOLD="${2:-mean+std}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
OUTPUT_DIR="results/crosscap_llama_${PRESET}_${THRESHOLD}"

echo "============================================"
echo "  Cross-Axis Capping -- Llama-3.3-70B"
echo "  Preset:               ${PRESET}"
echo "  Compliance threshold: ${THRESHOLD}"
echo "  Output:               ${OUTPUT_DIR}"
echo "============================================"
echo ""

python run_crosscap.py \
    --preset "$PRESET" \
    --model "$MODEL" \
    --compliance-threshold "$THRESHOLD" \
    --output-dir "$OUTPUT_DIR"
