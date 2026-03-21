#!/bin/bash
# Test all ResNetConformerDHOOM variants using test_dhoom.py
# Usage: bash scripts/test_dhoom_all.sh

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs

# NOTE: Update the checkpoint path to your best model before running
# Best model: experiments/20260303_181300_resnet_conformer_dhoom_dynamic_chunk/checkpoints/checkpoint_epoch50_step27300.pth

CONFIGS=(
    "configs/dynamic_test/resnet_conformer_dhoom_dynamic_chunk_offline.toml"
    "configs/dynamic_test/resnet_conformer_dhoom_dynamic_chunk_streaming.toml"
    "configs/dynamic_test/resnet_conformer_dhoom_dynamic_chunk_streaming_chunk9.toml"
    "configs/dynamic_test/resnet_conformer_dhoom_dynamic_chunk_streaming_chunk24.toml"
)

GPUS=(0 1 2 3)
PIDS=()

echo "============================================"
echo "DHOOM Test Script (using test_dhoom.py)"
echo "============================================"
echo ""

# Run standard configs (offline, streaming_chunk_mask)
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    gpu="${GPUS[$i]}"
    name=$(basename "$config" .toml)

    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run src/seld_v2/test/test_dhoom.py --config "$config" \
        > "logs/${name}.stdout.log" 2>&1 &
    PIDS+=($!)
done

# NOTE: To run streaming_cache test separately:
# CUDA_VISIBLE_DEVICES=0 uv run src/seld_v2/test/test_dhoom.py \
#   --config configs/dynamic_test/resnet_conformer_dhoom_dynamic_chunk_streaming.toml \
#   --mode streaming_cache

echo ""
echo "All ${#PIDS[@]} test jobs launched:"
for i in "${!CONFIGS[@]}"; do
    echo "  GPU ${GPUS[$i]} | PID ${PIDS[$i]} | $(basename ${CONFIGS[$i]})"
done
echo ""
echo "Monitor with: tail -f logs/resnet_conformer_dhoom*.stdout.log"
echo ""

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

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All tests completed successfully!"
else
    echo "Some tests failed. Check logs for details."
fi

exit $FAILED
