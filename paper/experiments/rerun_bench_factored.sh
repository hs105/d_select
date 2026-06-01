#!/bin/bash
# Re-run factored key benchmark with fair comparison (same code path for
# baseline and factored).  Fixed: old benchmark used attn_implementation=
# "sdpa" for baseline but "eager" for factored — unfair comparison.
#
# Requires one free GPU.  Run after Exp C2 finishes.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash rerun_bench_factored.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Re-running factored key benchmark (fair comparison) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python bench_factored.py \
    --device cuda:0 \
    --ranks 256,512 \
    --context_lengths 4096,16384 \
    --batch_sizes 1,4,8,16,32 \
    --n_decode_tokens 128 \
    --save_dir logs \
    2>&1 | tee logs/factored_bench_v2.log

echo ""
echo "Done: $(date)"
echo "Results saved to logs/factored_bench.json"
echo "Log saved to logs/factored_bench_v2.log"
echo ""
echo "Next: update paper Table 10 with new numbers from the JSON."
