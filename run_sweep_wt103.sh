#!/bin/bash
# ============================================================
# run_sweep_wt103.sh — Sweep d_select on WikiText-103
# ============================================================
# WikiText-103 has ~100M tokens (50x WikiText-2).
# Less overfitting → cleaner comparison of d_select values.
#
# Usage:
#   ./run_sweep_wt103.sh              # full sweep, 10 epochs
#   ./run_sweep_wt103.sh --quick      # 3 epochs sanity check
#   ./run_sweep_wt103.sh --single 64  # only d_select=64
# ============================================================

set -e

# ---- Model config (same as WikiText-2 for comparison) ----
D_MODEL=256
N_HEADS=8
N_LAYERS=6
D_FF=1024
MAX_SEQ_LEN=256
DROPOUT=0.1

# ---- Training config (adjusted for larger dataset) ----
OPTIMIZER="adamw"
LR=3e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=2000          # more warmup — more training steps
SCHEDULER="cosine"
GRAD_CLIP=1.0
BATCH_SIZE=64              # larger batch — more data available
EPOCHS=10                  # fewer epochs — each epoch is 50x more data
MIN_FREQ=3                 # slightly higher — larger vocab otherwise

DATA_PATH="/root/data"
SAVE_DIR="./checkpoints_wt103"
SEED=42

# ---- Parse arguments ----
QUICK=false
SINGLE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            EPOCHS=3
            QUICK=true
            shift
            ;;
        --single)
            SINGLE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --n_layers)
            N_LAYERS="$2"
            shift 2
            ;;
        --n_heads)
            N_HEADS="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- d_select values ----
if [ -n "$SINGLE" ]; then
    D_SELECT_VALUES=($SINGLE)
else
    D_SELECT_VALUES=(8 16 32 64 128 $D_MODEL)
    D_SELECT_VALUES=($(echo "${D_SELECT_VALUES[@]}" | tr ' ' '\n' | sort -un | tr '\n' ' '))
fi

# ---- Check data ----
if [ ! -d "$DATA_PATH/wikitext-103" ]; then
    echo "WikiText-103 not found at $DATA_PATH/wikitext-103"
    echo "Downloading..."
    python download_wikitext103.py
fi

if [ ! -d "$DATA_PATH/wikitext-103" ]; then
    echo "ERROR: Failed to get WikiText-103 data"
    exit 1
fi

echo "============================================================"
echo "ASYMMETRIC ATTENTION SWEEP — WikiText-103"
echo "============================================================"
echo "Config:"
echo "  d_model=$D_MODEL, n_heads=$N_HEADS, n_layers=$N_LAYERS, d_ff=$D_FF"
echo "  optimizer=$OPTIMIZER, lr=$LR, weight_decay=$WEIGHT_DECAY"
echo "  scheduler=$SCHEDULER, warmup=$WARMUP_STEPS"
echo "  batch_size=$BATCH_SIZE, epochs=$EPOCHS, seq_len=$MAX_SEQ_LEN"
echo "  min_freq=$MIN_FREQ"
echo "  data=$DATA_PATH (wikitext-103)"
echo ""
echo "Sweeping d_select: ${D_SELECT_VALUES[*]}"
echo "============================================================"
echo ""

mkdir -p "$SAVE_DIR"

# ---- Run each config ----
for D_SELECT in "${D_SELECT_VALUES[@]}"; do
    D_SELECT_PER_HEAD=$((D_SELECT / N_HEADS))
    RUN_NAME="wt103_dm${D_MODEL}_ds${D_SELECT}_L${N_LAYERS}_H${N_HEADS}"

    echo "============================================================"
    echo "Running: d_select=$D_SELECT (d_select/head=$D_SELECT_PER_HEAD)"
    echo "  Run name: $RUN_NAME"
    echo "============================================================"

    python train.py \
        --d_model $D_MODEL \
        --d_select $D_SELECT \
        --n_heads $N_HEADS \
        --n_layers $N_LAYERS \
        --d_ff $D_FF \
        --max_seq_len $MAX_SEQ_LEN \
        --dropout $DROPOUT \
        --optimizer $OPTIMIZER \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --warmup_steps $WARMUP_STEPS \
        --scheduler $SCHEDULER \
        --grad_clip $GRAD_CLIP \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --min_freq $MIN_FREQ \
        --data_path "$DATA_PATH" \
        --source wikitext \
        --save_dir "$SAVE_DIR" \
        --run_name "$RUN_NAME" \
        --seed $SEED \
        --run_leak_test \
        --generate_samples

    echo ""
done

# ---- Summary ----
echo ""
echo "============================================================"
echo "SWEEP COMPLETE — WikiText-103 Results"
echo "============================================================"
echo ""
printf "%-10s %-8s %-10s %-10s %-14s %-10s %-10s\n" \
    "d_select" "d/head" "Val PPL" "Test PPL" "Total Params" "QK Params" "QK Saved"
printf "%-10s %-8s %-10s %-10s %-14s %-10s %-10s\n" \
    "--------" "------" "-------" "--------" "------------" "---------" "--------"

# Get baseline QK params for savings calculation
BASELINE_RUN="wt103_dm${D_MODEL}_ds${D_MODEL}_L${N_LAYERS}_H${N_HEADS}"
BASELINE_JSON="$SAVE_DIR/${BASELINE_RUN}_results.json"
BASELINE_QK=0
if [ -f "$BASELINE_JSON" ]; then
    BASELINE_QK=$(python -c "import json; d=json.load(open('$BASELINE_JSON')); print(d['params']['qk'])")
fi

for D_SELECT in "${D_SELECT_VALUES[@]}"; do
    RUN_NAME="wt103_dm${D_MODEL}_ds${D_SELECT}_L${N_LAYERS}_H${N_HEADS}"
    RESULT_JSON="$SAVE_DIR/${RUN_NAME}_results.json"

    if [ -f "$RESULT_JSON" ]; then
        VAL_PPL=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(f\"{d['best_val_ppl']:.2f}\")")
        TEST_PPL=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(f\"{d['test_ppl']:.2f}\" if d.get('test_ppl') else 'N/A')")
        TOTAL_P=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(f\"{d['params']['total']:,}\")")
        QK_P=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(d['params']['qk'])")
        QK_DISPLAY=$(python -c "print(f'{$QK_P:,}')")
        D_PER_HEAD=$((D_SELECT / N_HEADS))

        if [ "$BASELINE_QK" -gt 0 ]; then
            SAVED=$(python -c "print(f\"{(1-$QK_P/$BASELINE_QK)*100:.0f}%\")")
        else
            SAVED="N/A"
        fi

        printf "%-10s %-8s %-10s %-10s %-14s %-10s %-10s\n" \
            "$D_SELECT" "$D_PER_HEAD" "$VAL_PPL" "$TEST_PPL" "$TOTAL_P" "$QK_DISPLAY" "$SAVED"
    fi
done

echo ""
echo "WikiText-2 results (for comparison):"
echo "  d_select=64:  Test PPL 122.24 (QK=197,376)"
echo "  d_select=256: Test PPL 122.22 (QK=789,504)"
echo ""
echo "Key question: with 50x more data and less overfitting,"
echo "does d_select still have minimal impact on perplexity?"