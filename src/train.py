"""
Fine-tuning T5 / ByT5 trên ViLexNorm.
Sử dụng HuggingFace Seq2SeqTrainer.

Usage:
    python src/train.py \
        --model_name VietAI/vit5-base \
        --run_name t5-vilexnorm \
        --train_file data/ViLexNorm/data/train.csv \
        --dev_file data/ViLexNorm/data/dev.csv \
        --epochs 10 \
        --batch_size 16
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

import wandb

from build_dataset import NormDataset


def edit_distance(ref, hyp):
    """Levenshtein edit distance giữa 2 list."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def make_compute_metrics(tokenizer):
    """Tạo hàm compute_metrics cho Seq2SeqTrainer."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        # Thay -100 bằng pad_token_id trước khi decode
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Exact Match
        total = len(decoded_preds)
        exact_match = sum(p == l for p, l in zip(decoded_preds, decoded_labels))

        # CER
        total_cer = 0
        total_chars = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            total_cer += edit_distance(list(label), list(pred))
            total_chars += len(list(label))

        # WER
        total_wer = 0
        total_words = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            total_wer += edit_distance(label.split(), pred.split())
            total_words += len(label.split())

        return {
            "exact_match": exact_match / max(1, total) * 100,
            "cer": total_cer / max(1, total_chars) * 100,
            "wer": total_wer / max(1, total_words) * 100,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5/ByT5 trên ViLexNorm")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model ID, vd: VietAI/vit5-base hoặc google/byt5-small")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Tên run trên WandB và thư mục output")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    # ────── Setup ──────
    save_dir = Path(args.output_dir) / args.run_name

    wandb.init(project="lexnorm2", name=args.run_name)

    # ────── Load tokenizer & model ──────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print(f"Model: {args.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ────── Datasets ──────
    train_ds = NormDataset(args.train_file, tokenizer,
                           max_src_len=args.max_src_len,
                           max_tgt_len=args.max_tgt_len)
    dev_ds = None
    if args.dev_file:
        dev_ds = NormDataset(args.dev_file, tokenizer,
                             max_src_len=args.max_src_len,
                             max_tgt_len=args.max_tgt_len)

    # ────── Data collator ──────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # ────── Training arguments ──────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=args.fp16,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if dev_ds else "no",
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        report_to="wandb",
        load_best_model_at_end=True if dev_ds else False,
        metric_for_best_model="cer" if dev_ds else None,
        greater_is_better=False if dev_ds else None,
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    # ────── Trainer ──────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    # ────── Train ──────
    trainer.train()

    # ────── Save best model ──────
    trainer.save_model(str(save_dir / "best"))
    tokenizer.save_pretrained(str(save_dir / "best"))

    print(f"\nModel saved to {save_dir / 'best'}")
    wandb.finish()


if __name__ == "__main__":
    main()