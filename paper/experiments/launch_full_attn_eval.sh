#!/bin/bash
# Wait for full_attn to finish, then launch downstream eval

LOG=/root/d_select/paper/experiments/logs/expC2_7b_full_attn_s137.log
CKPT_DIR=/sg-pretrain/checkpoints/expC2_7b_s137/full_attn

echo "Waiting for full_attn seed=137 to finish..."
while true; do
    if grep -q "Saving final checkpoint" "$LOG" 2>/dev/null; then
        echo "full_attn finished! Waiting 60s for checkpoint write..."
        sleep 60
        break
    fi
    sleep 120
done

# Find the final checkpoint
CKPT=$(ls -t ${CKPT_DIR}/expC2_7b_full_attn_step*.pt | head -1)
echo "Found checkpoint: $CKPT"

# Launch downstream eval on GPU 4
cd /root/d_select/paper/experiments
CUDA_VISIBLE_DEVICES=4 python -u eval_downstream_7b_scratch.py \
    --mode full_attn \
    --device cuda:0 \
    --ckpt_path "$CKPT" \
    > logs/expC2_downstream_full_attn_s137.log 2>&1

echo "Downstream eval complete!"
