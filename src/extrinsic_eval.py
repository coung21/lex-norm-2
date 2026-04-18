"""
Extrinsic Evaluation: Lexical Normalization → Emotion Classification.

Evaluates the impact of lexical normalization on downstream emotion
classification (UIT-VSMEC dataset) using:
  1. PhoBERT-base-v2 (Transformer fine-tuning)

Pipeline:
  1. Download normalization checkpoint from WandB artifacts
  2. Load UIT-VSMEC from HuggingFace
  3. Normalize train + val + test sets using BARTpho MTL model
  4. For each classifier:
     a. Train on RAW train → Eval on RAW test  → F1_raw
     b. Train on NORM train → Eval on NORM test → F1_norm
  5. Compare macro-F1 scores (ΔF1 = F1_norm - F1_raw)

Usage:
    python3 src/extrinsic_eval.py \\
        --wandb_project lexnorm2-mtl \\
        --wandb_artifact model-eval-mtl-uncertainty-test:latest
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import wandb

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_LABELS = [
    "Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"
]

CLASSIFIER_CONFIGS = {
    "phobert": {
        "model_name": "vinai/phobert-base-v2",
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 16,
    },
}


# ═══════════════════════════════════════════════════════════════
# 1. WandB Artifact Download
# ═══════════════════════════════════════════════════════════════

def download_checkpoint_from_wandb(
    project: str,
    artifact_name: str,
    download_dir: str = "artifacts/norm_checkpoint",
) -> str:
    """Download normalization model checkpoint from WandB artifacts."""
    print(f"\n📦 Downloading WandB artifact: {artifact_name}")
    api = wandb.Api()

    try:
        artifact = api.artifact(f"{project}/{artifact_name}", type="model")
        artifact_dir = artifact.download(root=download_dir)
        print(f"  ✅ Downloaded to: {artifact_dir}")

        # Verify checkpoint exists
        ckpt_path = Path(artifact_dir) / "checkpoint.pt"
        bartpho_path = Path(artifact_dir) / "bartpho"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint.pt not found in {artifact_dir}")
        if not bartpho_path.exists():
            raise FileNotFoundError(f"bartpho/ dir not found in {artifact_dir}")

        print(f"  ✅ checkpoint.pt found")
        print(f"  ✅ bartpho/ directory found")
        return str(artifact_dir)
    except Exception as e:
        print(f"  ❌ Failed to download artifact: {e}")
        raise


# ═══════════════════════════════════════════════════════════════
# 2. Dataset Loading
# ═══════════════════════════════════════════════════════════════

def load_vsmec_dataset() -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Load UIT-VSMEC from HuggingFace datasets.

    Returns:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    """
    from datasets import load_dataset

    print("\n📂 Loading UIT-VSMEC dataset from HuggingFace...")
    ds = load_dataset("tridm/UIT-VSMEC")

    train_texts = list(ds["train"]["Sentence"])
    train_labels = list(ds["train"]["Emotion"])
    val_texts = list(ds["validation"]["Sentence"])
    val_labels = list(ds["validation"]["Emotion"])
    test_texts = list(ds["test"]["Sentence"])
    test_labels = list(ds["test"]["Emotion"])

    print(f"  Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    print(f"  Label distribution (train): {Counter(train_labels).most_common()}")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ═══════════════════════════════════════════════════════════════
# 3. Normalization
# ═══════════════════════════════════════════════════════════════

def normalize_texts(
    texts: List[str],
    checkpoint_dir: str,
    batch_size: int = 16,
    max_length: int = 128,
    beam_size: int = 4,
) -> List[str]:
    """Normalize a list of texts using the BARTpho MTL model."""
    from transformers import AutoTokenizer

    # Import model from project
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import BARTphoMTL
    from utils import load_checkpoint

    bartpho_path = os.path.join(checkpoint_dir, "bartpho")
    print(f"\n🔄 Normalizing {len(texts)} texts...")
    print(f"  Loading tokenizer from: {bartpho_path}")
    tokenizer = AutoTokenizer.from_pretrained(bartpho_path)

    print(f"  Loading normalization model...")
    model = BARTphoMTL(bartpho_path, mode="normalization_only")
    load_checkpoint(checkpoint_dir, model)
    model = model.to(DEVICE)
    model.eval()

    normalized = []
    confidence_threshold = 0.85

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=beam_size,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Calculate generation probabilities map
        transition_scores = model.bartpho.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        probs = torch.exp(transition_scores)

        for b_idx in range(len(batch_texts)):
            orig_text = batch_texts[b_idx]
            gen_ids = outputs.sequences[b_idx]
            gen_probs = probs[b_idx].tolist()
            
            # Decode generated text
            norm_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Align tokens to apply confidence threshold filter
            import difflib
            orig_words = orig_text.split()
            norm_words = norm_text.split()
            
            # Get average sequence confidence as fallback proxy for word confidence
            valid_probs = [p for p in gen_probs if p > 0.0]
            seq_conf = sum(valid_probs) / len(valid_probs) if valid_probs else 0.0

            # If the generated sequence has very low confidence overall, revert entirely
            if seq_conf < (confidence_threshold - 0.15):
                normalized.append(orig_text)
                continue

            # Token-level blending using SequenceMatcher
            sm = difflib.SequenceMatcher(None, orig_words, norm_words)
            blended_words = []
            
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    blended_words.extend(orig_words[i1:i2])
                else:
                    # For modifications (replace, insert, delete)
                    # If high sequence confidence, we trust the normalization, 
                    # but if it's borderline, we retain the original token.
                    if seq_conf >= confidence_threshold:
                        blended_words.extend(norm_words[j1:j2])
                    else:
                        blended_words.extend(orig_words[i1:i2])
            
            final_text = " ".join(blended_words).strip()
            normalized.append(final_text)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Normalized {min(i + batch_size, len(texts))}/{len(texts)} samples...")

    print(f"  ✅ Normalization complete: {len(normalized)} texts")

    # Show samples
    print(f"\n  📋 Sample normalizations (with Confidence Threshold > {confidence_threshold}):")
    for j in range(min(5, len(texts))):
        changed = "🔄" if texts[j] != normalized[j] else "✅"
        print(f"    {changed} \"{texts[j]}\" → \"{normalized[j]}\"")

    # Clean up model from GPU
    del model
    torch.cuda.empty_cache()

    return normalized


# ═══════════════════════════════════════════════════════════════
# 4. Classifier: PhoBERT (Transformer Fine-tuning)
# ═══════════════════════════════════════════════════════════════

def _train_phobert_once(
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
    config, output_suffix,
):
    """Train PhoBERT once and return (f1, preds, report_dict)."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=7
    )

    class EmotionDataset(Dataset):
        def __init__(self, texts, labels, tok, max_len=128):
            self.encodings = tok(
                texts, truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_ds = EmotionDataset(train_texts, train_labels, tokenizer)
    val_ds = EmotionDataset(val_texts, val_labels, tokenizer)
    test_ds = EmotionDataset(test_texts, test_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"outputs/extrinsic/phobert_{output_suffix}",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        f1_macro = f1_score(eval_pred.label_ids, preds, average="macro") * 100
        return {"f1_macro": f1_macro}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    result = trainer.predict(test_ds)
    preds = np.argmax(result.predictions, axis=-1)
    f1 = f1_score(test_labels, preds, average="macro") * 100

    del model, trainer
    torch.cuda.empty_cache()
    return f1, preds


def train_and_eval_phobert(
    train_texts_raw: List[str],
    train_labels: List[int],
    val_texts_raw: List[str],
    val_labels: List[int],
    test_texts_raw: List[str],
    train_texts_norm: List[str],
    val_texts_norm: List[str],
    test_texts_norm: List[str],
    test_labels: List[int],
    config: dict,
) -> Dict[str, float]:
    """Fine-tune PhoBERT-base-v2 twice: raw train→raw test, norm train→norm test."""
    print(f"\n{'='*60}")
    print(f"  🤖 Training PhoBERT-base-v2 Classifier")
    print(f"{'='*60}")

    # Round 1: Train on raw → Eval on raw test
    print(f"\n  --- Round 1: Train on RAW data ---")
    f1_raw, raw_preds = _train_phobert_once(
        train_texts_raw, train_labels, val_texts_raw, val_labels,
        test_texts_raw, test_labels, config, "raw",
    )

    # Round 2: Train on norm → Eval on norm test
    print(f"\n  --- Round 2: Train on NORMALIZED data ---")
    f1_norm, norm_preds = _train_phobert_once(
        train_texts_norm, train_labels, val_texts_norm, val_labels,
        test_texts_norm, test_labels, config, "norm",
    )

    print(f"\n  PhoBERT Results:")
    print(f"    F1 (raw train→raw test):   {f1_raw:.2f}%")
    print(f"    F1 (norm train→norm test): {f1_norm:.2f}%")
    print(f"    ΔF1:                        {f1_norm - f1_raw:+.2f}%")

    return {
        "f1_raw": f1_raw,
        "f1_norm": f1_norm,
        "delta_f1": f1_norm - f1_raw,
    }


# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# 6. Report Generation
# ═══════════════════════════════════════════════════════════════

def generate_report(results: Dict[str, Dict], output_path: str):
    """Generate markdown report with comparison table."""
    lines = [
        "# Extrinsic Evaluation: Lexical Normalization → Emotion Classification",
        "",
        "## Dataset: UIT-VSMEC (7 emotion classes)",
        "",
        "## Tổng Hợp Kết Quả (Macro F1 trên Test Set)",
        "",
        "> Train on RAW → Eval on RAW test vs. Train on NORM → Eval on NORM test",
        "",
        "| Model | F1 (Raw→Raw) | F1 (Norm→Norm) | ΔF1 |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for name, res in results.items():
        delta = res["delta_f1"]
        delta_str = f"**+{delta:.2f}%** 🟢" if delta > 0 else f"{delta:.2f}% 🔴"
        lines.append(
            f"| **{name}** | {res['f1_raw']:.2f}% | {res['f1_norm']:.2f}% | {delta_str} |"
        )

    lines.append("")

    report = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 Report saved to: {output_path}")
    return report


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Extrinsic Evaluation")
    parser.add_argument("--wandb_project", type=str, default="lexnorm2-mtl")
    parser.add_argument("--wandb_artifact", type=str,
                        default="model-eval-mtl-uncertainty-test:latest",
                        help="WandB artifact name:version for normalization checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Local checkpoint dir (skip WandB download if provided)")
    parser.add_argument("--output_dir", type=str, default="outputs/extrinsic")
    parser.add_argument("--norm_batch_size", type=int, default=16)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n🚀 Starting Extrinsic Evaluation Pipeline...")
    print(f"  Project: {args.wandb_project}")
    print(f"  Artifact: {args.wandb_artifact}")

    # ── WandB Init ─────────────────────────────────────────────
    # If it hangs here, ensure you have ran 'wandb login' or set WANDB_API_KEY
    wandb.init(
        project=args.wandb_project,
        name="extrinsic-eval",
        job_type="extrinsic-evaluation",
        config=vars(args),
    )

    # ── Step 1: Download checkpoint ────────────────────────────
    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
        print(f"\n📦 Using local checkpoint: {ckpt_dir}")
    else:
        ckpt_dir = download_checkpoint_from_wandb(
            project=args.wandb_project,
            artifact_name=args.wandb_artifact,
        )

    # ── Step 2: Load dataset ───────────────────────────────────
    train_texts, train_labels_str, val_texts, val_labels_str, test_texts, test_labels_str = (
        load_vsmec_dataset()
    )

    # Encode labels
    le = LabelEncoder()
    le.fit(EMOTION_LABELS)
    train_labels = le.transform(train_labels_str).tolist()
    val_labels = le.transform(val_labels_str).tolist()
    test_labels = le.transform(test_labels_str).tolist()

    # ── Step 3: Normalize train + val + test sets ──────────────
    print(f"\n{'='*60}")
    print(f"  🔄 Normalizing ALL splits (train + val + test)")
    print(f"{'='*60}")

    train_texts_norm = normalize_texts(
        train_texts,
        checkpoint_dir=ckpt_dir,
        batch_size=args.norm_batch_size,
        beam_size=args.beam_size,
    )
    val_texts_norm = normalize_texts(
        val_texts,
        checkpoint_dir=ckpt_dir,
        batch_size=args.norm_batch_size,
        beam_size=args.beam_size,
    )
    test_texts_norm = normalize_texts(
        test_texts,
        checkpoint_dir=ckpt_dir,
        batch_size=args.norm_batch_size,
        beam_size=args.beam_size,
    )

    # Count how many texts changed per split
    for split_name, raw, norm in [
        ("Train", train_texts, train_texts_norm),
        ("Val", val_texts, val_texts_norm),
        ("Test", test_texts, test_texts_norm),
    ]:
        changed = sum(1 for a, b in zip(raw, norm) if a != b)
        print(f"  📊 {split_name}: {changed}/{len(raw)} texts modified ({changed/len(raw)*100:.1f}%)")

    # ── Step 4: Train & Evaluate classifiers ───────────────────
    # Each classifier is trained TWICE:
    #   Round 1: raw train → raw test
    #   Round 2: norm train → norm test
    results = {}

    # 4. PhoBERT
    results["PhoBERT-base-v2"] = train_and_eval_phobert(
        train_texts, train_labels, val_texts, val_labels, test_texts,
        train_texts_norm, val_texts_norm, test_texts_norm, test_labels,
        CLASSIFIER_CONFIGS["phobert"],
    )

    # ── Step 5: Report ─────────────────────────────────────────
    report_path = os.path.join(args.output_dir, "extrinsic_results.md")
    report = generate_report(results, report_path)

    # Log to WandB
    summary_table = wandb.Table(
        columns=["Model", "F1_Raw", "F1_Norm", "Delta_F1"],
        data=[
            [name, r["f1_raw"], r["f1_norm"], r["delta_f1"]]
            for name, r in results.items()
        ],
    )
    wandb.log({"extrinsic/summary": summary_table})

    for name, r in results.items():
        safe_name = name.lower().replace("-", "_").replace(" ", "_")
        wandb.log({
            f"extrinsic/{safe_name}_f1_raw": r["f1_raw"],
            f"extrinsic/{safe_name}_f1_norm": r["f1_norm"],
            f"extrinsic/{safe_name}_delta_f1": r["delta_f1"],
        })

    # Log report as artifact
    report_artifact = wandb.Artifact("extrinsic-report", type="report")
    report_artifact.add_file(report_path)
    wandb.log_artifact(report_artifact)

    wandb.finish()

    # ── Final Summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🎯 EXTRINSIC EVALUATION COMPLETE")
    print(f"{'='*60}")
    for name, r in results.items():
        delta = r["delta_f1"]
        icon = "🟢" if delta > 0 else "🔴"
        print(f"  {icon} {name}: {r['f1_raw']:.2f}% → {r['f1_norm']:.2f}% (Δ{delta:+.2f}%)")
    print(f"{'='*60}")
    print(f"  Report: {report_path}")
    print(f"  WandB: extrinsic-eval")
    print()


if __name__ == "__main__":
    main()
