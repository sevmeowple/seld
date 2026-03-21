#!/bin/bash
# Test all U2 chunk variants in parallel, one per GPU.
# Usage: bash scripts/test_u2_chunk_all.sh

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs

CONFIGS=(
    "configs/dynamic_test/chunk_4.toml"
    "configs/dynamic_test/chunk_9.toml"
    "configs/dynamic_test/chunk_24.toml"
    "configs/dynamic_test/chunk_49.toml"
)
GPUS=(0 1 2 0)

PIDS=()

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    gpu="${GPUS[$i]}"
    name=$(basename "$config" .toml)

    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run src/seld_v2/test/test_cache_model_streaming.py --config "$config" \
        > "logs/${name}.stdout.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#CONFIGS[@]} test jobs launched:"
for i in "${!CONFIGS[@]}"; do
    echo "  GPU ${GPUS[$i]} | PID ${PIDS[$i]} | $(basename ${CONFIGS[$i]})"
done
echo ""
echo "Monitor with: tail -f logs/chunk_*.stdout.log"

# Wait for all jobs, report exit status
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name=$(basename "${CONFIGS[$i]}" .toml)
    if wait "$pid"; then
        echo "[$(date '+%H:%M:%S')] $name (PID $pid) finished successfully"
    else
        echo "[$(date '+%H:%M:%S')] $name (PID $pid) FAILED with exit code $?"
        FAILED=1
    fi
done

exit $FAILED
