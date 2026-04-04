#!/bin/bash
# ============================================================
# Experiment E: Chinchilla-Optimal Scaling Laws
# ============================================================
#
# Launches thin_keys vs full_attn at 3 scales in sequence.
# Each scale runs both configs in parallel (4 GPUs each).
#
# Usage:
#   bash launch_exp_e.sh              # run all scales
#   bash launch_exp_e.sh 125M         # run only 125M (smoke test)
#   bash launch_exp_e.sh 350M         # run only 350M
#   bash launch_exp_e.sh 1.3B         # run only 1.3B
#   bash launch_exp_e.sh smoke        # quick 125M sanity check (100 steps)
#
# Expected wall-clock (8x H100):
#   125M: ~2 hours
#   350M: ~12 hours
#   1.3B: ~3 days
#   Total: ~4 days
# ============================================================

set -uo pipefail

cd "$(dirname "$0")"
LOGDIR="logs"
mkdir -p "$LOGDIR"

SCALE="${1:-all}"
SEED="${2:-42}"

run_scale() {
    local scale=$1
    local seed=$2
    echo ""
    echo "============================================================"
    echo "  Experiment E: ${scale} (seed=${seed})"
    echo "  $(date)"
    echo "============================================================"

    # Launch full_attn on GPUs 0-3
    echo "  Starting full_attn on GPUs 0-3..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        --master_port=29500 \
        experiment_e.py --scale "$scale" --mode full_attn \
        --seed "$seed" --resume \
        > "$LOGDIR/expE_${scale}_full_attn_launcher.log" 2>&1 &
    PID_FULL=$!

    # Launch thin_keys on GPUs 4-7
    echo "  Starting thin_keys on GPUs 4-7..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        --master_port=29501 \
        experiment_e.py --scale "$scale" --mode thin_keys \
        --seed "$seed" --resume \
        > "$LOGDIR/expE_${scale}_thin_keys_launcher.log" 2>&1 &
    PID_THIN=$!

    echo "  PIDs: full_attn=$PID_FULL  thin_keys=$PID_THIN"
    local ds
    ds=$(scale_to_dselect "$scale")
    echo "  Logs:"
    echo "    tail -f $LOGDIR/expE_${scale}_full_attn.log"
    echo "    tail -f $LOGDIR/expE_${scale}_thin${ds}.log"

    # Wait for both
    echo "  Waiting for both configs to finish..."
    wait $PID_FULL
    EXIT_FULL=$?
    wait $PID_THIN
    EXIT_THIN=$?

    echo ""
    echo "  ${scale} done: full_attn exit=$EXIT_FULL, thin_keys exit=$EXIT_THIN"
    echo "  $(date)"

    if [ $EXIT_FULL -ne 0 ] || [ $EXIT_THIN -ne 0 ]; then
        echo "  WARNING: One or both configs failed!"
        echo "  Check logs in $LOGDIR/"
    fi
}

scale_to_dselect() {
    case $1 in
        125M) echo 192 ;;
        350M) echo 256 ;;
        1.3B) echo 512 ;;
    esac
}

smoke_test() {
    echo "============================================================"
    echo "  Smoke test: 125M, 100 steps"
    echo "============================================================"
    # Just run a quick sanity check on 1 GPU each
    CUDA_VISIBLE_DEVICES=0 python experiment_e.py \
        --scale 125M --mode full_attn --seed 42 2>&1 | tail -20 &
    CUDA_VISIBLE_DEVICES=1 python experiment_e.py \
        --scale 125M --mode thin_keys --seed 42 2>&1 | tail -20 &
    wait
    echo "  Smoke test done."
}

case "$SCALE" in
    smoke)
        smoke_test
        ;;
    125M|350M|1.3B)
        run_scale "$SCALE" "$SEED"
        ;;
    all)
        echo "Running all scales sequentially (total ~4 days)..."
        echo ""
        run_scale "125M" "$SEED"
        run_scale "350M" "$SEED"
        run_scale "1.3B" "$SEED"

        # Run 125M and 350M with second seed while 1.3B results are fresh
        if [ "$SEED" = "42" ]; then
            echo ""
            echo "Running 125M and 350M with seed=137..."
            run_scale "125M" 137
            run_scale "350M" 137
        fi

        echo ""
        echo "============================================================"
        echo "  ALL SCALES COMPLETE"
        echo "  $(date)"
        echo "  Results: $LOGDIR/expE_*.json"
        echo "============================================================"
        ;;
    *)
        echo "Usage: $0 [125M|350M|1.3B|all|smoke] [seed]"
        exit 1
        ;;
esac
