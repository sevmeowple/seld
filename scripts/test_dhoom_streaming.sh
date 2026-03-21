#!/bin/bash
# Test DHOOM streaming configs only
# Usage: bash scripts/test_dhoom_streaming.sh

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs

CONFIGS=(
    "configs/dynamic_test/resnet_conformer_dhoom_dynamic_chunk_streaming.toml"
    "configs/dynamic_test/resnet_conformer_dhoom_dynamic_chunk_streaming_chunk4.toml"
)
GPUS=(1 2)

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
echo "All ${#CONFIGS[@]} DHOOM streaming jobs launched:"
for i in "${!CONFIGS[@]}"; do
    echo "  GPU ${GPUS[$i]} | PID ${PIDS[$i]} | $(basename ${CONFIGS[$i]})"
done
echo ""
echo "Monitor with: tail -f logs/resnet_conformer_dhoom_dynamic_chunk_streaming*.stdout.log"

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
