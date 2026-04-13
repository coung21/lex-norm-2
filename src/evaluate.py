"""
Evaluation cho T5 / ByT5 trên ViLexNorm.

Metrics:
- Exact Match (EM)
- Character Error Rate (CER)
- Word Error Rate (WER)
- Error Reduction Rate (ERR)

Usage:
    python src/evaluate.py \
        --model_path outputs/t5-vilexnorm/best \
        --dev_path data/ViLexNorm/data/dev.csv \
        --run_name eval-t5-vilexnorm
"""

import argparse

import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader

import wandb

from build_dataset import NormDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def edit_distance(ref, hyp):
    """Levenshtein edit distance giữa 2 chuỗi/list."""
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


def calculate_wer(ref, hyp):
    return edit_distance(ref.split(), hyp.split())


def calculate_cer(ref, hyp):
    return edit_distance(list(ref), list(hyp))


class Evaluator:
    def __init__(self, model_path, dev_path, batch_size=32):
        # Load tokenizer & model từ cùng thư mục (đã lưu cả hai khi train)
        if "vit5" in model_path.lower():
            from transformers import T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
        self.model.eval()

        self.dev_ds = NormDataset(dev_path, self.tokenizer)
        self.dev_dl = DataLoader(self.dev_ds, batch_size=batch_size, shuffle=False)

    def evaluate(self):
        total_exact_match = 0
        total_cer = 0
        total_wer_model = 0
        total_wer_baseline = 0
        total_gt_words = 0
        total_gt_chars = 0
        total_samples = 0

        # WandB table
        table = None
        if wandb.run is not None:
            table = wandb.Table(columns=["Original", "Ground Truth", "Prediction", "Correct?"])

        print("Starting Evaluation...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dev_dl):
                src_ids = batch['input_ids'].to(DEVICE)
                attn_mask = batch['attention_mask'].to(DEVICE)

                generated_ids = self.model.generate(
                    input_ids=src_ids,
                    attention_mask=attn_mask,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                )

                for idx in range(src_ids.size(0)):
                    dataset_idx = batch_idx * self.dev_dl.batch_size + idx
                    if dataset_idx >= len(self.dev_ds):
                        break

                    ex = self.dev_ds.data.iloc[dataset_idx]

                    original_text = str(ex['original']).strip()
                    ground_truth_text = str(ex['normalized']).strip()

                    predict_text = self.tokenizer.decode(
                        generated_ids[idx], skip_special_tokens=True
                    ).strip()

                    is_correct = (predict_text == ground_truth_text)
                    if is_correct:
                        total_exact_match += 1

                    # Console output cho vài mẫu đầu
                    if total_samples < 5:
                        print(f"\n--- Sample {total_samples} ---")
                        print(f"  Original : {original_text}")
                        print(f"  GT       : {ground_truth_text}")
                        print(f"  Predict  : '{predict_text}'")
                        print(f"  Correct  : {is_correct}")

                    # WandB table (giới hạn 200 dòng)
                    if table is not None and total_samples < 200:
                        table.add_data(original_text, ground_truth_text, predict_text,
                                       "✅" if is_correct else "❌")

                    total_wer_baseline += calculate_wer(ground_truth_text, original_text)
                    total_wer_model += calculate_wer(ground_truth_text, predict_text)
                    total_cer += calculate_cer(ground_truth_text, predict_text)

                    total_gt_words += len(ground_truth_text.split())
                    total_gt_chars += len(list(ground_truth_text))
                    total_samples += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed samples: {total_samples}")

        # Kết quả
        em_rate = total_exact_match / total_samples
        cer = total_cer / max(1, total_gt_chars)
        wer = total_wer_model / max(1, total_gt_words)
        err = (total_wer_baseline - total_wer_model) / total_wer_baseline if total_wer_baseline > 0 else 0.0

        print("=" * 40)
        print(f"EVALUATION RESULTS ({total_samples} samples)")
        print(f"EM: {em_rate*100:.2f}% | WER: {wer*100:.2f}% | CER: {cer*100:.2f}% | ERR: {err*100:.2f}%")

        if wandb.run is not None:
            wandb.log({
                "eval/exact_match": em_rate * 100,
                "eval/wer": wer * 100,
                "eval/cer": cer * 100,
                "eval/err": err * 100,
                "eval/predictions": table,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model directory (chứa cả tokenizer)")
    parser.add_argument("--dev_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run_name", type=str, default="eval")
    args = parser.parse_args()

    wandb.init(project="lexnorm2", name=args.run_name, job_type="evaluation")
    evaluator = Evaluator(args.model_path, args.dev_path, args.batch_size)
    evaluator.evaluate()
    wandb.finish()
