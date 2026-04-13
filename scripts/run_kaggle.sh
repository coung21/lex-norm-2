#!/bin/bash
# ─────────────────────────────────────────────────────
# Script chạy so sánh T5 vs ByT5 trên ViLexNorm (Kaggle)
# ─────────────────────────────────────────────────────

set -e

pip install -q transformers sentencepiece datasets wandb accelerate torch

TRAIN_FILE="data/ViLexNorm/data/train.csv"
DEV_FILE="data/ViLexNorm/data/dev.csv"
TEST_FILE="data/ViLexNorm/data/test.csv"

EPOCHS=10
BATCH_SIZE=16

# ═══════════════════════════════════════════════
# 1. ViT5-base
# ═══════════════════════════════════════════════
echo "=============================="
echo "  Training ViT5-base"
echo "=============================="

python src/train.py \
    --model_name VietAI/vit5-base \
    --run_name vit5-base-vilexnorm \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 3e-4

echo "--- Evaluating ViT5-base on dev set ---"
python src/evaluate.py \
    --model_path outputs/vit5-base-vilexnorm/best \
    --dev_path "$DEV_FILE" \
    --run_name eval-vit5-base-dev

echo "--- Evaluating ViT5-base on test set ---"
python src/evaluate.py \
    --model_path outputs/vit5-base-vilexnorm/best \
    --dev_path "$TEST_FILE" \
    --run_name eval-vit5-base-test

# ═══════════════════════════════════════════════
# 2. ByT5-small
# ═══════════════════════════════════════════════
echo "=============================="
echo "  Training ByT5-small"
echo "=============================="

python src/train.py \
    --model_name google/byt5-small \
    --run_name byt5-small-vilexnorm \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-4

echo "--- Evaluating ByT5-small on dev set ---"
python src/evaluate.py \
    --model_path outputs/byt5-small-vilexnorm/best \
    --dev_path "$DEV_FILE" \
    --run_name eval-byt5-small-dev

echo "--- Evaluating ByT5-small on test set ---"
python src/evaluate.py \
    --model_path outputs/byt5-small-vilexnorm/best \
    --dev_path "$TEST_FILE" \
    --run_name eval-byt5-small-test

echo "=============================="
echo "  Done! Check WandB for results."
echo "=============================="
