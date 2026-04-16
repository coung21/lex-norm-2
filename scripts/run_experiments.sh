#!/bin/bash
# ─────────────────────────────────────────────────────────────
# MTL Vietnamese Lexical Normalization — 4 Ablation Studies
#
# Backbone: vinai/bartpho-syllable-base
# Task 1: NSW Detection (token classification)
# Task 2: Lexical Normalization (seq2seq generation)
#
# Ablations:
#   1. Single-Task Detection Only
#   2. Single-Task Normalization Only
#   3. MTL Equal Weighting (L_det + L_norm)
#   4. MTL + PCGrad (Gradient Surgery)
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
# 1. Single-Task: NSW Detection Only
# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  [1/4] Single-Task: Detection Only"
echo "========================================"

python3 src/train.py \
    --mode detection_only \
    --run_name st-detection \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --gradient_accumulation_steps $ACCUM \
    --lr $LR \
    --head_lr $HEAD_LR \
    --fp16

echo "--- Evaluating Detection on test set ---"
python3 src/evaluate.py \
    --model_path outputs/st-detection/best \
    --test_file "$TEST_FILE" \
    --mode detection_only \
    --run_name eval-st-detection-test

# ═══════════════════════════════════════════════════════════════
# 2. Single-Task: Normalization Only
# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  [2/4] Single-Task: Normalization Only"
echo "========================================"

python3 src/train.py \
    --mode normalization_only \
    --run_name st-normalization \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --gradient_accumulation_steps $ACCUM \
    --lr $LR \
    --head_lr $HEAD_LR \
    --fp16

echo "--- Evaluating Normalization on test set ---"
python3 src/evaluate.py \
    --model_path outputs/st-normalization/best \
    --test_file "$TEST_FILE" \
    --mode normalization_only \
    --run_name eval-st-normalization-test

# ═══════════════════════════════════════════════════════════════
# 3. MTL — Equal Loss Weighting (Baseline)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  [3/4] MTL: Equal Weighting"
echo "========================================"

python3 src/train.py \
    --mode mtl \
    --run_name mtl-equal \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --gradient_accumulation_steps $ACCUM \
    --lr $LR \
    --head_lr $HEAD_LR \
    --fp16

echo "--- Evaluating MTL-Equal on test set ---"
python3 src/evaluate.py \
    --model_path outputs/mtl-equal/best \
    --test_file "$TEST_FILE" \
    --mode mtl \
    --run_name eval-mtl-equal-test

# ═══════════════════════════════════════════════════════════════
# 4. MTL + PCGrad (Gradient Surgery)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  [4/4] MTL: PCGrad"
echo "========================================"

python3 src/train.py \
    --mode mtl \
    --use_pcgrad \
    --run_name mtl-pcgrad \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --gradient_accumulation_steps $ACCUM \
    --lr $LR \
    --head_lr $HEAD_LR \
    --fp16

echo "--- Evaluating MTL-PCGrad on test set ---"
python3 src/evaluate.py \
    --model_path outputs/mtl-pcgrad/best \
    --test_file "$TEST_FILE" \
    --mode mtl \
    --run_name eval-mtl-pcgrad-test

# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  All 4 experiments complete!"
echo "  Check WandB project: lexnorm2-mtl"
echo "========================================"
