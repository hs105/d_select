cd /sg-pretrain/focus
mkdir -p checkpoints_7b logs

# Control: rank 1024 = no compression
CUDA_VISIBLE_DEVICES=0 python -u svd_finetune_7b.py \
    --rank 1024 --epochs 3 --lr 5e-5 \
    2>&1 | tee logs/mistral_r1024_control.log &

# Rank 512 = K_dim/2, 50% K cache saved
CUDA_VISIBLE_DEVICES=1 python -u svd_finetune_7b.py \
    --rank 512 --epochs 3 --lr 5e-5 \
    2>&1 | tee logs/mistral_r512.log &

# Rank 256 = K_dim/4, 75% K cache saved
CUDA_VISIBLE_DEVICES=2 python -u svd_finetune_7b.py \
    --rank 256 --epochs 3 --lr 5e-5 \
    2>&1 | tee logs/mistral_r256.log &

# Rank 128 = K_dim/8, 87% K cache saved
CUDA_VISIBLE_DEVICES=3 python -u svd_finetune_7b.py \
    --rank 128 --epochs 3 --lr 5e-5 \
    2>&1 | tee logs/mistral_r128.log &

wait
echo "All done"

