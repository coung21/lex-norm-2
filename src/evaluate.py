"""
Evaluation entry point for MTL Vietnamese Lexical Normalization.

Evaluates a saved model on test/dev set and logs results to WandB.

Metrics:
  - NSW Detection: Precision / Recall / F1
  - Normalization: ERR / Word Accuracy / BLEU-4 / Exact Match

Usage:
    python src/evaluate.py \
        --model_path outputs/mtl-pcgrad/best \
        --test_file data/ViLexNorm/data/test.csv \
        --mode mtl \
        --run_name eval-mtl-pcgrad
"""

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb

from model import BARTphoMTL
from dataset import MTLDataset
from metrics import compute_detection_metrics, compute_normalization_metrics
from utils import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTL Model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved checkpoint directory")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test/dev CSV file")
    parser.add_argument("--mode", type=str, default="mtl",
                        choices=["detection_only", "normalization_only", "mtl"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--project", type=str, default="lexnorm2-mtl")
    args = parser.parse_args()

    # ── WandB ──────────────────────────────────────────────────────
    wandb.init(
        project=args.project,
        name=args.run_name,
        job_type="evaluation",
        config=vars(args),
    )

    # ── Load tokenizer & model ─────────────────────────────────────
    bartpho_path = f"{args.model_path}/bartpho"
    print(f"Loading tokenizer from: {bartpho_path}")
    tokenizer = AutoTokenizer.from_pretrained(bartpho_path)

    print(f"Loading model (mode={args.mode})...")
    model = BARTphoMTL(bartpho_path, mode=args.mode)

    # Load full checkpoint (includes detection head weights)
    ckpt = load_checkpoint(args.model_path, model)
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"  Previous metrics: {ckpt.get('metrics', {})}")

    model = model.to(DEVICE)
    model.eval()

    # ── Dataset ────────────────────────────────────────────────────
    test_ds = MTLDataset(args.test_file, tokenizer, args.max_seq_len)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    print(f"Test set: {len(test_ds)} samples")

    # ── Evaluation ─────────────────────────────────────────────────
    all_det_preds = []
    all_det_labels = []
    all_pred_texts = []
    all_ref_texts = []
    all_orig_texts = []

    # WandB prediction table
    table = wandb.Table(
        columns=["Original", "Ground Truth", "Prediction", "NSW Detected", "Correct?"]
    )

    print("\nStarting Evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_device = {k: v.to(DEVICE) for k, v in batch.items()}

            # ── Detection ──────────────────────────────────────────
            if args.mode in ("detection_only", "mtl"):
                det_preds = model.predict_detection(
                    batch_device["input_ids"],
                    batch_device["attention_mask"],
                )
                det_labels = batch["detection_labels"]
                for i in range(det_preds.size(0)):
                    mask = det_labels[i] != -100
                    all_det_preds.extend(det_preds[i][mask].cpu().tolist())
                    all_det_labels.extend(det_labels[i][mask].cpu().tolist())

            # ── Normalization ──────────────────────────────────────
            if args.mode in ("normalization_only", "mtl"):
                generated_ids = model.generate(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                    max_length=args.max_seq_len,
                    num_beams=args.beam_size,
                    early_stopping=True,
                )

                pred_texts = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                pred_texts = [t.strip() for t in pred_texts]
                all_pred_texts.extend(pred_texts)

                ref_ids = batch["labels"].clone()
                ref_ids[ref_ids == -100] = tokenizer.pad_token_id
                ref_texts = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                ref_texts = [t.strip() for t in ref_texts]
                all_ref_texts.extend(ref_texts)

                orig_texts = tokenizer.batch_decode(
                    batch["input_ids"], skip_special_tokens=True
                )
                orig_texts = [t.strip() for t in orig_texts]
                all_orig_texts.extend(orig_texts)

                # Table rows (limit to 300)
                for i, (o, r, p) in enumerate(zip(orig_texts, ref_texts, pred_texts)):
                    if len(table.data) < 300:
                        is_correct = p == r
                        nsw_count = "N/A"
                        if args.mode == "mtl" and batch_idx * args.batch_size + i < len(all_det_preds):
                            # Count NSW detections for this sample
                            nsw_count = str(sum(
                                1 for l in batch["detection_labels"][i].tolist() if l == 1
                            ))
                        table.add_data(o, r, p, nsw_count, "✅" if is_correct else "❌")

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} samples...")

    # ── Compute & log metrics ──────────────────────────────────────
    metrics = {}

    if args.mode in ("detection_only", "mtl") and all_det_labels:
        det_metrics = compute_detection_metrics(all_det_preds, all_det_labels)
        metrics.update(det_metrics)

    if args.mode in ("normalization_only", "mtl") and all_ref_texts:
        norm_metrics = compute_normalization_metrics(
            all_pred_texts, all_ref_texts, all_orig_texts
        )
        metrics.update(norm_metrics)

    # Print results
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ({len(test_ds)} samples)")
    print(f"{'='*50}")

    if args.mode in ("detection_only", "mtl"):
        print(f"\nNSW Detection:")
        print(f"  Precision: {metrics.get('det_precision', 0):.2f}%")
        print(f"  Recall:    {metrics.get('det_recall', 0):.2f}%")
        print(f"  F1:        {metrics.get('det_f1', 0):.2f}%")

    if args.mode in ("normalization_only", "mtl"):
        print(f"\nNormalization:")
        print(f"  ERR:          {metrics.get('norm_err', 0):.2f}%")
        print(f"  Word Accuracy: {metrics.get('norm_word_acc', 0):.2f}%")
        print(f"  BLEU-4:       {metrics.get('norm_bleu4', 0):.2f}")
        print(f"  Exact Match:  {metrics.get('norm_exact_match', 0):.2f}%")

    # Print some sample predictions
    if all_pred_texts:
        print(f"\n── Sample Predictions ──")
        for i in range(min(5, len(all_pred_texts))):
            correct = "✅" if all_pred_texts[i] == all_ref_texts[i] else "❌"
            print(f"  [{correct}] Original:   {all_orig_texts[i]}")
            print(f"       GT:         {all_ref_texts[i]}")
            print(f"       Prediction: {all_pred_texts[i]}")
            print()

    # Log to WandB
    wandb.log({f"test/{k}": v for k, v in metrics.items()})
    if len(table.data) > 0:
        wandb.log({"test/predictions": table})

    wandb.finish()
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
