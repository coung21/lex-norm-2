#!/bin/bash
# ─────────────────────────────────────────────────────
# Fair comparison: mT5-small vs ByT5-small trên ViLexNorm
#
# Cả hai model đều:
#   - Từ Google
#   - Pre-trained trên mC4 (multilingual)
#   - ~300M parameters
#   - Biến số duy nhất: Subword (SentencePiece) vs Byte-level
# ─────────────────────────────────────────────────────

set -e

pip install -q transformers sentencepiece datasets wandb accelerate torch

TRAIN_FILE="data/ViLexNorm/data/train.csv"
DEV_FILE="data/ViLexNorm/data/dev.csv"
TEST_FILE="data/ViLexNorm/data/test.csv"

EPOCHS=10
LR=3e-4

# ═══════════════════════════════════════════════
# 1. mT5-small (Subword tokenizer — SentencePiece)
#    ~300M params, max_len=128 (subword tokens ngắn)
# ═══════════════════════════════════════════════
echo "=============================="
echo "  Training mT5-small"
echo "=============================="

python src/train.py \
    --model_name google/mt5-small \
    --run_name mt5-small-vilexnorm \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --lr "$LR" \
    --max_src_len 128 \
    --max_tgt_len 128

echo "--- Evaluating mT5-small on dev set ---"
python src/evaluate.py \
    --model_path outputs/mt5-small-vilexnorm/best \
    --dev_path "$DEV_FILE" \
    --run_name eval-mt5-small-dev

echo "--- Evaluating mT5-small on test set ---"
python src/evaluate.py \
    --model_path outputs/mt5-small-vilexnorm/best \
    --dev_path "$TEST_FILE" \
    --run_name eval-mt5-small-test

# ═══════════════════════════════════════════════
# 2. ByT5-small (Byte-level — không dùng tokenizer)
#    ~300M params, max_len=512 (byte sequences dài hơn)
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
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr "$LR" \
    --max_src_len 256 \
    --max_tgt_len 256 \
    --fp16

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
