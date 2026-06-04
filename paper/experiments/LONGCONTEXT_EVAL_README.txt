Long-Context Evaluation for Thin Keys Paper
=============================================
Addresses reviewer criticism #1: no long-context evaluation.

Files:
  paper/experiments/eval_longcontext.py    -- Main evaluation script
  paper/experiments/launch_longcontext.sh  -- Launch wrapper with logging


PREREQUISITES
-------------
- Python 3.8+
- PyTorch with CUDA
- transformers (pip install transformers)
- Mistral-7B weights at /sg-pretrain/models/mistral-7b (or set MODEL_PATH)
- WikiText-103 at /root/data/wikitext-103/ (for PPL eval; NIAH/passkey need no data)

If WikiText-103 is not at the default path, supply any plain text file:
  --data_path /path/to/some_text.txt


WHAT IT EVALUATES
-----------------
1. Long-context PPL curves (4K, 8K, 16K, 32K) using local WikiText-103
2. Needle-in-a-haystack: hides a 7-digit number at varying depths, tests retrieval
3. Passkey retrieval: standard passkey test with synthetic filler

All three compare baseline (rank=1024) vs compressed (rank=512, 256).
No internet/download required -- NIAH and passkey are fully synthetic,
PPL uses the local WikiText-103 already on the machine.


COMMANDS
--------

# 1. Quick sanity check (~30 min, smaller eval)
python paper/experiments/eval_longcontext.py \
    --ranks 1024,256 \
    --device cuda:0 \
    --quick

# 2. Full evaluation -- baseline + r512 + r256 (~few hours)
python paper/experiments/eval_longcontext.py \
    --ranks 1024,512,256 \
    --device cuda:0

# 3. PPL only (fastest, most important for paper figures)
python paper/experiments/eval_longcontext.py \
    --ranks 1024,512,256 \
    --device cuda:0 \
    --eval ppl

# 4. NIAH only (good for paper heatmap figure)
python paper/experiments/eval_longcontext.py \
    --ranks 1024,512,256 \
    --device cuda:0 \
    --eval niah

# 5. Passkey only
python paper/experiments/eval_longcontext.py \
    --ranks 1024,512,256 \
    --device cuda:0 \
    --eval passkey

# 6. With the launch wrapper (logs to file automatically)
bash paper/experiments/launch_longcontext.sh
bash paper/experiments/launch_longcontext.sh --quick
bash paper/experiments/launch_longcontext.sh --eval ppl

# 7. Custom model path / GPU / data
MODEL_PATH=/path/to/mistral-7b CUDA_DEVICE=cuda:1 bash paper/experiments/launch_longcontext.sh
python paper/experiments/eval_longcontext.py --data_path /path/to/text.txt --ranks 1024,256

# 8. If you have a fine-tuned checkpoint from svd_finetune_7b.py
python paper/experiments/eval_longcontext.py \
    --ranks 256 \
    --checkpoint_dir /sg-pretrain/focus/checkpoints_7b \
    --device cuda:0


OUTPUT
------
- JSON results saved to --save_dir (default: /sg-pretrain/focus/paper/experiments/logs/)
- Summary table printed to stdout with PPL at each context length and NIAH/passkey accuracy
- launch_longcontext.sh also saves a run.log in the output directory


RECOMMENDED ORDER
-----------------
1. Run --quick first to verify everything works
2. Run --eval ppl for the PPL curves (most important for paper)
3. Run --eval niah for the needle-in-a-haystack heatmap
4. Run full (no --eval flag) for complete results
