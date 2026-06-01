#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Experiment C (v2): 7B from scratch + downstream evaluation
# ══════════════════════════════════════════════════════════════
#
# Retrains both 7B LLaMA models with checkpoint saving enabled,
# then runs downstream evaluation on the saved checkpoints.
#
#   Phase 1 (parallel):
#     GPUs 0-3: full_attn training (~26h)
#     GPUs 4-7: thin_keys training (~24h)
#
#   Phase 2 (sequential after training):
#     GPU 0: full_attn downstream eval (~2-3h)
#     GPU 4: thin_keys downstream eval (~2-3h)
#
# Usage:
#   bash run_experiment_c_with_eval.sh           # full run
#   bash run_experiment_c_with_eval.sh smoke     # 100-step smoke test
#   bash run_experiment_c_with_eval.sh eval_only # skip training, just eval
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

MODE="${1:-}"

# ── Smoke test ──
if [ "$MODE" = "smoke" ]; then
    echo "[$(date)] ═══ SMOKE TEST (100 steps, 2 GPUs each) ═══"

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
        "${SCRIPT_DIR}/experiment_c.py" \
        --mode full_attn \
        --total_tokens 13_107_200 \
        --eval_interval 50 \
        --save_checkpoint \
        > "${LOG_DIR}/expC_7b_full_attn_smoke.log" 2>&1 &
    PID_FULL=$!

    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
        "${SCRIPT_DIR}/experiment_c.py" \
        --mode thin_keys \
        --total_tokens 13_107_200 \
        --eval_interval 50 \
        --save_checkpoint \
        > "${LOG_DIR}/expC_7b_thin1024_smoke.log" 2>&1 &
    PID_THIN=$!

    echo "[$(date)] Baseline PID=$PID_FULL, Thin keys PID=$PID_THIN"
    wait $PID_FULL; echo "[$(date)] Baseline smoke done (exit=$?)"
    wait $PID_THIN; echo "[$(date)] Thin keys smoke done (exit=$?)"

    echo "[$(date)] Running smoke downstream eval..."
    python "${SCRIPT_DIR}/eval_downstream_7b_scratch.py" \
        --mode full_attn --device cuda:0 --tasks hellaswag --limit 100 \
        > "${LOG_DIR}/expC_downstream_full_attn_smoke.log" 2>&1 &

    python "${SCRIPT_DIR}/eval_downstream_7b_scratch.py" \
        --mode thin_keys --device cuda:4 --tasks hellaswag --limit 100 \
        > "${LOG_DIR}/expC_downstream_thin1024_smoke.log" 2>&1 &

    wait
    echo "[$(date)] ═══ SMOKE TEST COMPLETE ═══"
    exit 0
fi

# ── Eval only (skip training) ──
if [ "$MODE" = "eval_only" ]; then
    echo "[$(date)] ═══ EVAL ONLY (using existing checkpoints) ═══"

    echo "[$(date)] Starting full_attn eval on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_downstream_7b_scratch.py" \
        --mode full_attn --device cuda:0 \
        > "${LOG_DIR}/expC_downstream_full_attn.log" 2>&1 &
    PID_FULL=$!

    echo "[$(date)] Starting thin_keys eval on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 python "${SCRIPT_DIR}/eval_downstream_7b_scratch.py" \
        --mode thin_keys --device cuda:4 \
        > "${LOG_DIR}/expC_downstream_thin1024.log" 2>&1 &
    PID_THIN=$!

    echo "[$(date)] Waiting for evals..."
    echo "  Monitor: tail -f ${LOG_DIR}/expC_downstream_*.log"

    wait $PID_FULL; echo "[$(date)] full_attn eval done (exit=$?)"
    wait $PID_THIN; echo "[$(date)] thin_keys eval done (exit=$?)"

    echo "[$(date)] ═══ EVAL COMPLETE ═══"
    echo "Results:"
    echo "  ${LOG_DIR}/expC_downstream_full_attn.json"
    echo "  ${LOG_DIR}/expC_downstream_thin1024.json"
    exit 0
fi

# ── Full run: train + eval ──
echo "[$(date)] ═══ EXPERIMENT C v2: 7B FROM SCRATCH + DOWNSTREAM EVAL ═══"
echo "[$(date)] Phase 1: Training (full attention GPUs 0-3, thin keys GPUs 4-7)"
echo ""

# Phase 1: Training (parallel)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    "${SCRIPT_DIR}/experiment_c.py" \
    --mode full_attn \
    --save_checkpoint \
    > "${LOG_DIR}/expC_7b_full_attn.log" 2>&1 &
PID_FULL=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 \
    "${SCRIPT_DIR}/experiment_c.py" \
    --mode thin_keys \
    --save_checkpoint \
    > "${LOG_DIR}/expC_7b_thin1024.log" 2>&1 &
PID_THIN=$!

echo "[$(date)] Training PIDs: full_attn=$PID_FULL, thin_keys=$PID_THIN"
echo "[$(date)] Estimated training time: ~26 hours"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/expC_7b_full_attn.log"
echo "  tail -f ${LOG_DIR}/expC_7b_thin1024.log"
echo ""

wait $PID_FULL
RC_FULL=$?
echo "[$(date)] full_attn training done (exit=$RC_FULL)"

wait $PID_THIN
RC_THIN=$?
echo "[$(date)] thin_keys training done (exit=$RC_THIN)"

if [ $RC_FULL -ne 0 ] || [ $RC_THIN -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed. Aborting eval."
    exit 1
fi

# Phase 2: Downstream evaluation (parallel on separate GPUs)
echo ""
echo "[$(date)] ═══ Phase 2: Downstream Evaluation ═══"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_downstream_7b_scratch.py" \
    --mode full_attn --device cuda:0 \
    > "${LOG_DIR}/expC_downstream_full_attn.log" 2>&1 &
PID_EVAL_FULL=$!

CUDA_VISIBLE_DEVICES=4 python "${SCRIPT_DIR}/eval_downstream_7b_scratch.py" \
    --mode thin_keys --device cuda:4 \
    > "${LOG_DIR}/expC_downstream_thin1024.log" 2>&1 &
PID_EVAL_THIN=$!

echo "[$(date)] Eval PIDs: full_attn=$PID_EVAL_FULL, thin_keys=$PID_EVAL_THIN"
echo "[$(date)] Estimated eval time: ~2-3 hours"
echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/expC_downstream_full_attn.log"
echo "  tail -f ${LOG_DIR}/expC_downstream_thin1024.log"

wait $PID_EVAL_FULL; echo "[$(date)] full_attn eval done (exit=$?)"
wait $PID_EVAL_THIN; echo "[$(date)] thin_keys eval done (exit=$?)"

echo ""
echo "[$(date)] ═══ EXPERIMENT C v2 COMPLETE ═══"
echo "Training results:"
echo "  ${LOG_DIR}/expC_7b_full_attn.json"
echo "  ${LOG_DIR}/expC_7b_thin1024.json"
echo "Downstream results:"
echo "  ${LOG_DIR}/expC_downstream_full_attn.json"
echo "  ${LOG_DIR}/expC_downstream_thin1024.json"
