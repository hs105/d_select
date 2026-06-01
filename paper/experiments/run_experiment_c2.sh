#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Experiment C2: 7B from scratch — 20B tokens
# ══════════════════════════════════════════════════════════════
#
# Trains two 7B LLaMA models in parallel on full OWT (~8B tokens,
# ~2.5 epochs to reach 20B total tokens):
#   GPUs 0-3: full_attn (baseline)
#   GPUs 4-7: thin_keys (d_select=1024 = d_model/4)
#
# Each run: ~305K steps, ~10.6 days on 4×H100
#
# Usage:
#   bash run_experiment_c2.sh                # full run
#   bash run_experiment_c2.sh smoke          # 100-step smoke test (2 GPUs each)
#   bash run_experiment_c2.sh resume         # resume from latest checkpoints
#   bash run_experiment_c2.sh prepare_data   # data prep only (CPU, ~2-4 hours)
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

MODE="${1:-}"

# ── Phase 0: Data preparation ──
if [ "$MODE" = "prepare_data" ]; then
    echo "[$(date)] ═══ DATA PREPARATION (CPU-only) ═══"
    echo "[$(date)] Tokenizing full OpenWebText (~8B tokens)..."
    echo "[$(date)] This takes ~2-4 hours. Safe to run while GPUs are busy."
    python "${SCRIPT_DIR}/experiment_c2.py" --prepare_data \
        2>&1 | tee "${LOG_DIR}/expC2_data_prep.log"
    echo "[$(date)] ═══ DATA PREPARATION COMPLETE ═══"
    exit 0
fi

# ── Smoke test ──
if [ "$MODE" = "smoke" ]; then
    echo "[$(date)] ═══ SMOKE TEST (100 steps, 2 GPUs each) ═══"

    RESUME_FLAG=""

    # Baseline: GPUs 0-1
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
        "${SCRIPT_DIR}/experiment_c2.py" \
        --mode full_attn \
        --total_tokens 13_107_200 \
        --eval_interval 50 \
        --ckpt_interval 50 \
        $RESUME_FLAG \
        > "${LOG_DIR}/expC2_7b_full_attn_smoke.log" 2>&1 &
    PID_FULL=$!

    # Thin keys: GPUs 2-3
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
        "${SCRIPT_DIR}/experiment_c2.py" \
        --mode thin_keys \
        --total_tokens 13_107_200 \
        --eval_interval 50 \
        --ckpt_interval 50 \
        $RESUME_FLAG \
        > "${LOG_DIR}/expC2_7b_thin1024_smoke.log" 2>&1 &
    PID_THIN=$!

    echo "[$(date)] Baseline PID=$PID_FULL, Thin keys PID=$PID_THIN"
    echo "[$(date)] Waiting for smoke tests..."
    echo ""
    echo "Monitor:"
    echo "  tail -f ${LOG_DIR}/expC2_7b_full_attn_smoke.log"
    echo "  tail -f ${LOG_DIR}/expC2_7b_thin1024_smoke.log"

    wait $PID_FULL
    echo "[$(date)] Baseline smoke done (exit code: $?)"

    wait $PID_THIN
    echo "[$(date)] Thin keys smoke done (exit code: $?)"

    echo ""
    echo "[$(date)] ═══ SMOKE TEST COMPLETE ═══"
    echo "Logs: ${LOG_DIR}/expC2_7b_*_smoke.log"
    echo "Checkpoints: ls /sg-pretrain/checkpoints/expC2_7b/"
    exit 0
fi

# ── Resume flag ──
RESUME_FLAG=""
if [ "$MODE" = "resume" ]; then
    RESUME_FLAG="--resume"
    echo "[$(date)] ═══ RESUMING EXPERIMENT C2 ═══"
else
    echo "[$(date)] ═══ EXPERIMENT C2: 7B FROM SCRATCH, 20B TOKENS ═══"
fi

echo "[$(date)] Full attention (GPUs 0-3) + Thin keys (GPUs 4-7)"
echo "[$(date)] Estimated time: ~10.6 days each (running in parallel)"
echo ""

# GPUs 0-3: baseline (full_attn)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    "${SCRIPT_DIR}/experiment_c2.py" \
    --mode full_attn \
    $RESUME_FLAG \
    > "${LOG_DIR}/expC2_7b_full_attn.log" 2>&1 &
PID_FULL=$!

# GPUs 4-7: thin keys (d_select=1024)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 \
    "${SCRIPT_DIR}/experiment_c2.py" \
    --mode thin_keys \
    $RESUME_FLAG \
    > "${LOG_DIR}/expC2_7b_thin1024.log" 2>&1 &
PID_THIN=$!

echo "[$(date)] Baseline PID=$PID_FULL, Thin keys PID=$PID_THIN"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/expC2_7b_full_attn.log"
echo "  tail -f ${LOG_DIR}/expC2_7b_thin1024.log"
echo ""
echo "Check checkpoints:"
echo "  ls -lh /sg-pretrain/checkpoints/expC2_7b/"
echo ""

# Wait for both
wait $PID_FULL
RC_FULL=$?
echo "[$(date)] Baseline done (exit code: $RC_FULL)"

wait $PID_THIN
RC_THIN=$?
echo "[$(date)] Thin keys done (exit code: $RC_THIN)"

echo ""
echo "[$(date)] ═══ EXPERIMENT C2 COMPLETE ═══"
echo "Results:"
echo "  ${LOG_DIR}/expC2_7b_full_attn.json"
echo "  ${LOG_DIR}/expC2_7b_thin1024.json"
echo "Trajectories:"
echo "  ${LOG_DIR}/expC2_7b_full_attn_trajectory.json"
echo "  ${LOG_DIR}/expC2_7b_thin1024_trajectory.json"
echo "Checkpoints:"
echo "  ls -lh /sg-pretrain/checkpoints/expC2_7b/"
