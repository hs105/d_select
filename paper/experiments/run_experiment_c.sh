#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Experiment C: 7B from scratch — Full Attention vs Thin Keys
# ══════════════════════════════════════════════════════════════
#
# Runs two 7B LLaMA models in parallel:
#   GPUs 0-3: full_attn (baseline)
#   GPUs 4-7: thin_keys (d_select=1024 = d_model/4)
#
# Each run: ~2B tokens, ~30K steps, ~33 hours on 4×H100
#
# Usage:
#   bash run_experiment_c.sh          # full run
#   bash run_experiment_c.sh smoke    # 100-step smoke test on 2 GPUs each
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

SMOKE="${1:-}"

if [ "$SMOKE" = "smoke" ]; then
    echo "[$(date)] ═══ SMOKE TEST (100 steps, 2 GPUs each) ═══"

    # Baseline: GPUs 0-1
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
        "${SCRIPT_DIR}/experiment_c.py" \
        --mode full_attn \
        --total_tokens 13_107_200 \
        --eval_interval 50 \
        > "${LOG_DIR}/expC_7b_full_attn_smoke.log" 2>&1 &
    PID_FULL=$!

    # Thin keys: GPUs 2-3
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
        "${SCRIPT_DIR}/experiment_c.py" \
        --mode thin_keys \
        --total_tokens 13_107_200 \
        --eval_interval 50 \
        > "${LOG_DIR}/expC_7b_thin1024_smoke.log" 2>&1 &
    PID_THIN=$!

    echo "[$(date)] Baseline PID=$PID_FULL, Thin keys PID=$PID_THIN"
    echo "[$(date)] Waiting for smoke tests..."

    wait $PID_FULL
    echo "[$(date)] Baseline smoke done (exit code: $?)"

    wait $PID_THIN
    echo "[$(date)] Thin keys smoke done (exit code: $?)"

    echo "[$(date)] ═══ SMOKE TEST COMPLETE ═══"
    echo "Logs: ${LOG_DIR}/expC_7b_*_smoke.log"
    exit 0
fi

echo "[$(date)] ═══ EXPERIMENT C: 7B FROM SCRATCH ═══"
echo "[$(date)] Full attention (GPUs 0-3) + Thin keys (GPUs 4-7)"
echo ""

# GPUs 0-3: baseline (full_attn)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    "${SCRIPT_DIR}/experiment_c.py" \
    --mode full_attn \
    > "${LOG_DIR}/expC_7b_full_attn.log" 2>&1 &
PID_FULL=$!

# GPUs 4-7: thin keys (d_select=1024)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 \
    "${SCRIPT_DIR}/experiment_c.py" \
    --mode thin_keys \
    > "${LOG_DIR}/expC_7b_thin1024.log" 2>&1 &
PID_THIN=$!

echo "[$(date)] Baseline PID=$PID_FULL, Thin keys PID=$PID_THIN"
echo "[$(date)] Estimated time: ~33 hours each (running in parallel)"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/expC_7b_full_attn.log"
echo "  tail -f ${LOG_DIR}/expC_7b_thin1024.log"
echo ""

# Wait for both
wait $PID_FULL
RC_FULL=$?
echo "[$(date)] Baseline done (exit code: $RC_FULL)"

wait $PID_THIN
RC_THIN=$?
echo "[$(date)] Thin keys done (exit code: $RC_THIN)"

echo ""
echo "[$(date)] ═══ EXPERIMENT C COMPLETE ═══"
echo "Results:"
echo "  ${LOG_DIR}/expC_7b_full_attn.json"
echo "  ${LOG_DIR}/expC_7b_thin1024.json"
echo "Trajectories:"
echo "  ${LOG_DIR}/expC_7b_full_attn_trajectory.json"
echo "  ${LOG_DIR}/expC_7b_thin1024_trajectory.json"
