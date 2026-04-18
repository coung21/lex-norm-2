#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Extrinsic Evaluation: Lexical Normalization → Emotion Classification
#
# Downloads normalization checkpoint from WandB
# Evaluates impact on UIT-VSMEC emotion classification
# Classifiers: PhoBERT-base-v2, BiLSTM, GRU
# ─────────────────────────────────────────────────────────────

set -e

# ── Install dependencies ───────────────────────────────────────
echo "Installing dependencies (this may take a minute)..."
pip install transformers sentencepiece torch wandb accelerate \
    pandas numpy scikit-learn datasets

# ── Run extrinsic evaluation ──────────────────────────────────
python3 src/extrinsic_eval.py \
    --wandb_project lexnorm2-mtl \
    --wandb_artifact model-eval-mtl-uncertainty-test:latest \
    --output_dir outputs/extrinsic \
    --norm_batch_size 16 \
    --beam_size 4

echo ""
echo "========================================"
echo "  Extrinsic evaluation complete!"
echo "  Check: outputs/extrinsic/extrinsic_results.md"
echo "  WandB: lexnorm2-mtl / extrinsic-eval"
echo "========================================"
