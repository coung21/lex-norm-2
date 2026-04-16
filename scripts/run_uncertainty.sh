#!/bin/bash
# ─────────────────────────────────────────────────────────────
# MTL Vietnamese Lexical Normalization — Uncertainty Weighting
#
# Backbone: vinai/bartpho-syllable-base
# Task 1: NSW Detection (token classification)
# Task 2: Lexical Normalization (seq2seq generation)
#
# Ablation: MTL + Uncertainty Weighting (Learnable Loss Weights)
# ─────────────────────────────────────────────────────────────

set -e

# ── Install dependencies ───────────────────────────────────────
pip install -q transformers sentencepiece torch wandb accelerate pandas numpy scikit-learn sacrebleu

# ── Common settings ────────────────────────────────────────────
TRAIN_FILE="data/ViLexNorm/data/train.csv"
DEV_FILE="data/ViLexNorm/data/dev.csv"
TEST_FILE="data/ViLexNorm/data/test.csv"

EPOCHS=15
BATCH=8
ACCUM=4
LR=3e-5
HEAD_LR=1e-4

# ═══════════════════════════════════════════════════════════════
# 1. MTL + Uncertainty Weighting
# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  MTL: Uncertainty Weighting"
echo "========================================"

python3 src/train.py \
    --mode mtl \
    --use_uncertainty \
    --run_name mtl-uncertainty \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --gradient_accumulation_steps $ACCUM \
    --lr $LR \
    --head_lr $HEAD_LR \
    --fp16

echo "--- Evaluating MTL-Uncertainty on test set ---"
python3 src/evaluate.py \
    --model_path outputs/mtl-uncertainty/best \
    --test_file "$TEST_FILE" \
    --mode mtl \
    --run_name eval-mtl-uncertainty-test

echo ""
echo "========================================"
echo "  Experiment complete!"
echo "  Check WandB project: lexnorm2-mtl"
echo "========================================"
