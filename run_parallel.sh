#!/bin/bash
# Cross-axis capping experiment — data-parallel launcher
#
# Usage:
#   ./run_parallel.sh [PRESET] [N_GPUS]
#
# Examples:
#   ./run_parallel.sh full 4
#   ./run_parallel.sh sanity 2
#   ./run_parallel.sh small 1    # single GPU, no parallelism

set -e

PRESET="${1:-full}"
N_GPUS="${2:-4}"

echo "Preset: $PRESET"
echo "GPUs:   $N_GPUS"
echo ""

# Step 1: Warmup — download model, datasets, compute axes + thresholds
echo "=== Step 1: Warmup ==="
python run_crosscap.py --preset "$PRESET" --warmup
echo ""

# Step 2: Launch one chunk per GPU in parallel
echo "=== Step 2: Running $N_GPUS chunks in parallel ==="
pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
    echo "  Starting chunk $i/$N_GPUS on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python run_crosscap.py --preset "$PRESET" --chunk "$i/$N_GPUS" &
    pids+=($!)
done

# Wait for all chunks and check for failures
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "  ERROR: chunk $i failed"
        failed=1
    fi
done

if [ "$failed" -eq 1 ]; then
    echo "Some chunks failed. Aborting merge."
    exit 1
fi
echo ""

# Step 3: Merge chunk results into final 4 CSVs
echo "=== Step 3: Merging results ==="
python run_crosscap.py --preset "$PRESET" --merge

echo ""
echo "Done."
