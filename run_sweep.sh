#!/bin/bash
# ============================================================
# run_sweep.sh — Sweep d_select on WikiText-2 with GPU selection
# ============================================================
# Usage examples:
#   ./run_sweep.sh                     # run all on default GPU(s)
#   ./run_sweep.sh --gpu 0              # use GPU 0 only
#   ./run_sweep.sh --gpu "1,2" --quick  # use GPUs 1 and 2 for quick test
#   ./run_sweep.sh --single 32 --gpu 3  # run only d_select=32 on GPU 3
# ============================================================

set -e  # exit on error

# ---- Default config ----
D_MODEL=256
N_HEADS=8
N_LAYERS=6
D_FF=1024
MAX_SEQ_LEN=256
DROPOUT=0.1

OPTIMIZER="adamw"
LR=3e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
SCHEDULER="cosine"
GRAD_CLIP=1.0
BATCH_SIZE=32
EPOCHS=30

DATA_PATH="/root/data"
SAVE_DIR="./checkpoints"
SEED=42

GPU_IDS=""          # empty = use all visible GPUs

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            EPOCHS=5
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
        --optimizer)
            OPTIMIZER="$2"
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
        --gpu)
            GPU_IDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- Set CUDA_VISIBLE_DEVICES if requested ----
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Using GPU(s): $GPU_IDS"
else
    echo "Using all available GPUs (CUDA_VISIBLE_DEVICES not set)"
fi

# ---- Determine d_select values to sweep ----
if [ -n "$SINGLE" ]; then
    D_SELECT_VALUES=($SINGLE)
else
    D_SELECT_VALUES=(8 16 32 64 128 $D_MODEL)
    # Remove duplicates and sort
    D_SELECT_VALUES=($(echo "${D_SELECT_VALUES[@]}" | tr ' ' '\n' | sort -un | tr '\n' ' '))
fi

# ---- Validate data directory ----
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data path $DATA_PATH not found!"
    exit 1
fi

# ---- Print configuration ----
echo "============================================================"
echo "ASYMMETRIC ATTENTION SWEEP"
echo "============================================================"
echo "Shared config:"
echo "  d_model=$D_MODEL, n_heads=$N_HEADS, n_layers=$N_LAYERS, d_ff=$D_FF"
echo "  optimizer=$OPTIMIZER, lr=$LR, weight_decay=$WEIGHT_DECAY"
echo "  scheduler=$SCHEDULER, warmup=$WARMUP_STEPS"
echo "  batch_size=$BATCH_SIZE, epochs=$EPOCHS, seq_len=$MAX_SEQ_LEN"
echo "  data=$DATA_PATH"
echo ""
echo "Sweeping d_select: ${D_SELECT_VALUES[*]}"
echo "============================================================"

# ---- Prepare results file ----
mkdir -p "$SAVE_DIR"
RESULTS_FILE="$SAVE_DIR/sweep_results.txt"
echo "d_select,d_select_per_head,val_ppl,test_ppl,total_params,qk_params,epochs" > "$RESULTS_FILE"

# ---- Run each configuration ----
for D_SELECT in "${D_SELECT_VALUES[@]}"; do
    D_SELECT_PER_HEAD=$((D_SELECT / N_HEADS))
    RUN_NAME="dmodel${D_MODEL}_dselect${D_SELECT}_L${N_LAYERS}_H${N_HEADS}"

    echo ""
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
        --data_path "$DATA_PATH" \
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
echo "SWEEP COMPLETE — Summary"
echo "============================================================"

printf "%-10s %-10s %-12s %-12s %-12s %-10s\n" \
    "d_select" "d_sel/hd" "Val PPL" "Test PPL" "Total Params" "QK Params"
printf "%-10s %-10s %-12s %-12s %-12s %-10s\n" \
    "--------" "--------" "-------" "--------" "------------" "---------"

for D_SELECT in "${D_SELECT_VALUES[@]}"; do
    RUN_NAME="dmodel${D_MODEL}_dselect${D_SELECT}_L${N_LAYERS}_H${N_HEADS}"
    RESULT_JSON="$SAVE_DIR/${RUN_NAME}_results.json"

    if [ -f "$RESULT_JSON" ]; then
        VAL_PPL=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(f\"{d['best_val_ppl']:.2f}\")")
        TEST_PPL=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(f\"{d['test_ppl']:.2f}\" if d.get('test_ppl') else 'N/A')")
        TOTAL_P=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(f\"{d['params']['total']:,}\")")
        QK_P=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(f\"{d['params']['qk']:,}\")")
        D_PER_HEAD=$((D_SELECT / N_HEADS))

        printf "%-10s %-10s %-12s %-12s %-12s %-10s\n" \
            "$D_SELECT" "$D_PER_HEAD" "$VAL_PPL" "$TEST_PPL" "$TOTAL_P" "$QK_P"
    fi
done

echo ""
echo "Detailed results in: $SAVE_DIR/*_results.json"
echo "Key question: at what d_select does perplexity start to degrade?"