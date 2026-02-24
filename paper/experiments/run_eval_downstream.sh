#!/bin/bash
# Experiment A: Downstream task evaluation on Mistral-7B
# Runs baseline and SVD-compressed models in parallel on separate GPUs
#
# Logs go to paper/experiments/logs/ (separate from the other project)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Experiment A: Downstream Evaluation"
echo "Logs: ${LOG_DIR}"
echo "=========================================="

# Baseline (no compression) on GPU 0
echo "[$(date)] Starting baseline (rank=1024) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_downstream.py" \
    --rank 1024 \
    --device cuda:0 \
    --tasks mmlu,hellaswag,arc_challenge,winogrande,gsm8k \
    > "${LOG_DIR}/downstream_baseline.log" 2>&1 &
PID_BASELINE=$!

# Compressed rank=256 (75% K cache saved) on GPU 4
echo "[$(date)] Starting compressed (rank=256) on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python "${SCRIPT_DIR}/eval_downstream.py" \
    --rank 256 \
    --device cuda:0 \
    --tasks mmlu,hellaswag,arc_challenge,winogrande,gsm8k \
    > "${LOG_DIR}/downstream_r256.log" 2>&1 &
PID_R256=$!

# Compressed rank=512 (50% K cache saved) on GPU 5
echo "[$(date)] Starting compressed (rank=512) on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python "${SCRIPT_DIR}/eval_downstream.py" \
    --rank 512 \
    --device cuda:0 \
    --tasks mmlu,hellaswag,arc_challenge,winogrande,gsm8k \
    > "${LOG_DIR}/downstream_r512.log" 2>&1 &
PID_R512=$!

echo ""
echo "PIDs: baseline=$PID_BASELINE  r256=$PID_R256  r512=$PID_R512"
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/downstream_baseline.log"
echo "  tail -f ${LOG_DIR}/downstream_r256.log"
echo "  tail -f ${LOG_DIR}/downstream_r512.log"
echo ""

# Wait for all
wait $PID_BASELINE
echo "[$(date)] Baseline done (exit code: $?)"
wait $PID_R256
echo "[$(date)] Rank 256 done (exit code: $?)"
wait $PID_R512
echo "[$(date)] Rank 512 done (exit code: $?)"

echo ""
echo "All evaluations complete. Results in ${LOG_DIR}/"
