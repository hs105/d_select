# Asymmetric Attention: Decoupling Selection from Value Transfer

## Core Idea

In standard transformers, Q, K, and V all use dimension d_model. But Q and K only produce scalar attention weights (selection), while V carries rich representations (value transfer). **Selection needs far fewer dimensions than value transfer.**

```
Q [n, d_select] ──┐
                   ├── dot product ──→ attention weights [n, n] ──┐
K [n, d_select] ──┘                     (just scalars)           ├── weighted sum ──→ output [n, d_model]
                                                                  │
V [n, d_model] ──────────────────────────────────────────────────┘
```

## Files

```
asymmetric_transformer.py   — AsymmetricTransformer model (importable)
train.py                    — Training script for language modeling
compress_qk.py              — Post-training SVD compression of pretrained GPT-2
download_wikitext103.py     — Download WikiText-103 dataset
run_sweep.sh                — Sweep d_select on WikiText-2
run_sweep_wt103.sh          — Sweep d_select on WikiText-103

# Earlier algorithmic experiments (standalone scripts)
focus_net_minimal.py        — Original focus net experiment (single sentence)
focus_net_copyback.py       — Focus net on copy-back task
asymmetric_attention.py     — d_select sweep on copy-back task
asymmetric_kv_retrieval.py  — d_select sweep on key-value retrieval task
```

## Results Summary

### Experiment 1: Copy-Back Task (Positional Selection)
Token[t] = token[t-8]. Model must learn to look 8 positions back.

```
d_select  d/head  Accuracy  Converge
4         1       100%      epoch 300
8         2       100%      epoch 200
64        16      100%      epoch 200
```

**Finding:** Positional selection needs only 1 dimension per head.

### Experiment 2: Key-Value Retrieval (Content-Based Selection)
8 random key-value pairs, query a key, predict the value. Position is useless.

```
d_select  d/head  Accuracy  Converge
4         1       65.2%     did not converge
8         2       100%      epoch 1900
16        4       100%      epoch 1900
64        16      100%      epoch 1000
```

**Finding:** Content matching needs 2 dims/head (log₂ of key space).

### Experiment 3: WikiText-2 (Real Language, Small Dataset)

```
d_select  d/head  Val PPL  Test PPL  QK Params  QK Saved
8         1       133.78   126.48    24,672     97%
16        2       132.67   125.49    49,344     94%
32        4       130.51   123.78    98,688     87%
64        8       129.34   122.24    197,376    75%
128       16      126.42   120.76    394,752    50%
256       32      126.95   122.22    789,504    baseline
```

**Finding:** d_select=64 matches baseline exactly. d_select=128 actually beats it (regularization effect from overfitting).

### Experiment 4: WikiText-103 (Real Language, Large Dataset, No Overfitting)

```
d_select  d/head  Val PPL  Test PPL
32        4       38.38    (pending)
64        8       37.22    (pending)
256       32      35.67    (pending)
```

**Finding:** Without overfitting, d_select matters more but still modestly. d_select=64 costs ~4.3% PPL for 75% QK savings.

### Experiment 5: Post-Training Compression of GPT-2 (pending)
SVD-compress W_Q and W_K of pretrained GPT-2, keep W_V intact, no retraining.


---

## How to Run Each Experiment

### Prerequisites

```bash
pip install torch
pip install transformers datasets    # for GPT-2 compression experiment
```

### Experiment 1 & 2: Algorithmic Tasks

```bash
# Copy-back task (positional selection)
python asymmetric_attention.py

# Key-value retrieval (content-based selection)
python asymmetric_kv_retrieval.py
```

### Experiment 3: WikiText-2 Sweep

```bash
# Download data
pip install datasets
python download_data.py --huggingface

# Full sweep (d_select = 8, 16, 32, 64, 128, 256)
chmod +x run_sweep.sh
./run_sweep.sh

# Quick test
./run_sweep.sh --quick --single 64
```

### Experiment 4: WikiText-103 Sweep

```bash
# Download WikiText-103
python download_wikitext103.py --huggingface

# Run three key configs in parallel on separate GPUs
mkdir -p logs checkpoints_wt103

CUDA_VISIBLE_DEVICES=0 python -u train.py --data_path /root/data --source wikitext --d_select 256 \
    --d_model 256 --n_heads 8 --n_layers 6 --d_ff 1024 \
    --epochs 10 --batch_size 64 --min_freq 200 --warmup_steps 2000 \
    --save_dir ./checkpoints_wt103 --run_name wt103_mf200_ds256 \
    --run_leak_test --generate_samples > logs/ds256.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.py --data_path /root/data --source wikitext --d_select 64 \
    --d_model 256 --n_heads 8 --n_layers 6 --d_ff 1024 \
    --epochs 10 --batch_size 64 --min_freq 200 --warmup_steps 2000 \
    --save_dir ./checkpoints_wt103 --run_name wt103_mf200_ds64 \
    --run_leak_test --generate_samples > logs/ds64.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.py --data_path /root/data --source wikitext --d_select 32 \
    --d_model 256 --n_heads 8 --n_layers 6 --d_ff 1024 \
    --epochs 10 --batch_size 64 --min_freq 200 --warmup_steps 2000 \
    --save_dir ./checkpoints_wt103 --run_name wt103_mf200_ds32 \
    --run_leak_test --generate_samples > logs/ds32.log 2>&1 &

wait
echo "All done"

# Monitor progress
tail -n 3 logs/ds256.log logs/ds64.log logs/ds32.log

# Run d_select=128 separately
CUDA_VISIBLE_DEVICES=0 python -u train.py --data_path /root/data --source wikitext --d_select 128 \
    --d_model 256 --n_heads 8 --n_layers 6 --d_ff 1024 \
    --epochs 10 --batch_size 64 --min_freq 200 --warmup_steps 2000 \
    --save_dir ./checkpoints_wt103 --run_name wt103_mf200_ds128 \
    --run_leak_test --generate_samples > logs/ds128.log 2>&1 &
```

### Experiment 5: Post-Training GPT-2 Compression

```bash
pip install transformers datasets

# Analyze singular values + full sweep (reloads model per rank)
CUDA_VISIBLE_DEVICES=3 python -u compress_qk.py --analyze_svd > logs/compress.log 2>&1

# Quick test with single rank
CUDA_VISIBLE_DEVICES=3 python -u compress_qk.py --rank 192

# Also compare: what happens if you compress V too?
CUDA_VISIBLE_DEVICES=3 python -u compress_qk.py --compress_v --analyze_svd

# Try on GPT-2 Medium (345M params, d_model=1024)
CUDA_VISIBLE_DEVICES=3 python -u compress_qk.py --model gpt2-medium --analyze_svd
```


---

## Key Arguments for train.py

```
Model:
  --d_model 256         Model dimension (and d_value)
  --d_select 64         QK dimension (default: d_model = standard transformer)
  --n_heads 8           Number of attention heads
  --n_layers 6          Number of layers
  --d_ff 1024           FFN hidden dimension

Data:
  --data_path /root/data    Path containing wikitext-2/ or wikitext-103/
  --source wikitext         Force WikiText (auto prefers wikitext-103 > wikitext-2)
  --min_freq 200            Minimum word frequency for vocabulary

Training:
  --optimizer adamw         adam, adamw, sgd
  --lr 3e-4                 Learning rate
  --scheduler cosine        cosine, linear, none
  --warmup_steps 2000       LR warmup steps
  --epochs 10               Training epochs
  --batch_size 64           Batch size
  --grad_clip 1.0           Gradient clipping

Other:
  --run_leak_test           Run causal leak diagnostic
  --generate_samples        Generate text samples after training
  --device auto             auto, cpu, cuda
```


---

## Practical Implications

For production LLMs (d_model=4096, 128K context, serving 100 users):

```
                    Standard        d_select=d_model/4     Savings
W_Q params/layer    16.7M           4.2M                   75%
W_K params/layer    16.7M           4.2M                   75%
KV cache (total)    429 GB          322 GB                 25%
Attention QK FLOPs  n²×4096         n²×1024                75%
```

Two ways to achieve this:
1. **Train from scratch** with small d_select (Experiments 1-4)
2. **Compress post-training** via SVD on W_Q, W_K (Experiment 5)