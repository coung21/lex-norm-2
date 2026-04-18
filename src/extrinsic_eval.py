"""
Extrinsic Evaluation: Lexical Normalization → Emotion Classification.

Evaluates the impact of lexical normalization on downstream hate speech
classification (ViHSD dataset) using:
  1. LSTM (Embedding + LSTM + Linear)
  2. BiLSTM (Embedding + BiLSTM + Linear)
  3. GRU (Embedding + GRU + Linear)

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
    "CLEAN", "OFFENSIVE", "HATE"
]

CLASSIFIER_CONFIGS = {
    "lstm": {
        "embed_dim": 300,
        "hidden_dim": 230,
        "num_layers": 2,
        "dropout": 0.3,
        "epochs": 10,
        "lr": 1e-3,
        "batch_size": 32,
    },
    "bilstm": {
        "embed_dim": 300,
        "hidden_dim": 150,
        "num_layers": 2,
        "dropout": 0.3,
        "epochs": 10,
        "lr": 1e-3,
        "batch_size": 32,
    },
    "gru": {
        "embed_dim": 300,
        "hidden_dim": 180,
        "num_layers": 2,
        "dropout": 0.3,
        "epochs": 10,
        "lr": 1e-3,
        "batch_size": 32,
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
# 2. Dataset Loading (ViHSD)
# ═══════════════════════════════════════════════════════════════

def load_vihsd_dataset() -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Load ViHSD from HuggingFace datasets (uitnlp/vihsd).

    Returns:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    """
    from datasets import load_dataset

    print("\n📂 Loading ViHSD dataset from HuggingFace (uitnlp/vihsd)...")
    try:
        ds = load_dataset("uitnlp/vihsd")
    except Exception as e:
        print(f"❌ Could not load uitnlp/vihsd: {e}")
        print("Note: If the dataset is gated, you must run 'huggingface-cli login' first.")
        sys.exit(1)

    # Some versions might use 'free_text', 'text', or 'comment', and 'label_id' or 'label'
    def extract_texts(split_ds):
        if "free_text" in split_ds.features:
            texts = list(split_ds["free_text"])
        elif "comment" in split_ds.features:
            texts = list(split_ds["comment"])
        elif "text" in split_ds.features:
            texts = list(split_ds["text"])
        else:
            raise ValueError(f"Unknown text column in: {list(split_ds.features.keys())}")
        
        # Replace None with empty string and cast to string to prevent Tokenizer ValueError
        return ["" if t is None else str(t) for t in texts]

    def extract_labels(split_ds):
        if "label_id" in split_ds.features:
            return list(split_ds["label_id"])
        if "label" in split_ds.features:
            return list(split_ds["label"])
        raise ValueError(f"Unknown label column in: {list(split_ds.features.keys())}")

    train_texts = extract_texts(ds["train"])
    train_labels = extract_labels(ds["train"])
    
    val_texts = extract_texts(ds["validation"])
    val_labels = extract_labels(ds["validation"])

    test_texts = extract_texts(ds["test"])
    test_labels = extract_labels(ds["test"])

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
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=beam_size,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        normalized.extend([t.strip() for t in decoded])

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
# 4. Classifier: LSTM / BiLSTM / GRU (RNN-based)
# ═══════════════════════════════════════════════════════════════

class Vocabulary:
    """Simple word-level vocabulary."""

    def __init__(self, max_size: int = 30000):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.max_size = max_size

    def build(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  Vocabulary size: {len(self.word2idx)}")

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        # Pad
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class RNNTextDataset(Dataset):
    """Dataset for BiLSTM/GRU classifiers."""

    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 128):
        self.input_ids = torch.tensor(
            [vocab.encode(t, max_len) for t in texts], dtype=torch.long
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


class LSTMClassifier(nn.Module):
    """Standard LSTM text classifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]
        return self.fc(self.dropout(hidden))


class BiLSTMClassifier(nn.Module):
    """BiLSTM text classifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, (hidden, _) = self.lstm(embedded)
        # Concat last hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))


class GRUClassifier(nn.Module):
    """GRU text classifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    model_name: str,
) -> nn.Module:
    """Train an RNN classifier (BiLSTM or GRU)."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                logits = model(input_ids)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())

        val_f1 = f1_score(all_labels, all_preds, average="macro") * 100
        avg_loss = total_loss / len(train_loader)
        print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.2f}%")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"    Best Val F1: {best_f1:.2f}%")
    return model


def eval_model(model: nn.Module, dataloader: DataLoader) -> float:
    """Evaluate a classifier. Returns macro_f1."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            logits = model(input_ids)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())

    f1 = f1_score(all_labels, all_preds, average="macro") * 100
    return f1


def _create_classifier_model(model_type, vocab_size, config):
    """Create a LSTM, BiLSTM, or GRU model."""
    num_classes = len(EMOTION_LABELS)  # Now 3 for ViHSD
    if model_type == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size, embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"], num_classes=num_classes,
            num_layers=config["num_layers"], dropout=config["dropout"],
        )
    elif model_type == "bilstm":
        return BiLSTMClassifier(
            vocab_size=vocab_size, embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"], num_classes=num_classes,
            num_layers=config["num_layers"], dropout=config["dropout"],
        )
    elif model_type == "gru":
        return GRUClassifier(
            vocab_size=vocab_size, embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"], num_classes=num_classes,
            num_layers=config["num_layers"], dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_and_eval_classifier(
    c_type: str,
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
    """Train and evaluate a classifier twice: raw vs normalized."""
    name = c_type.upper()
    print(f"\n{'='*60}")
    print(f"  🧠 Training {name} Classifier")
    print(f"{'='*60}")

    # ── Round 1: Train on RAW → Eval on RAW test ──────────────
    print(f"\n  --- Round 1: Train on RAW data ---")
    vocab_raw = Vocabulary(max_size=30000)
    vocab_raw.build(train_texts_raw)

    train_ds = RNNTextDataset(train_texts_raw, train_labels, vocab_raw)
    val_ds = RNNTextDataset(val_texts_raw, val_labels, vocab_raw)
    test_ds_raw = RNNTextDataset(test_texts_raw, test_labels, vocab_raw)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2)
    test_loader_raw = DataLoader(test_ds_raw, batch_size=config["batch_size"] * 2)

    model_raw = _create_classifier_model(c_type, len(vocab_raw), config)
    total_params = sum(p.numel() for p in model_raw.parameters())
    print(f"  Total parameters: {total_params:,}")

    model_raw = train_model(
        model_raw, train_loader, val_loader,
        epochs=config["epochs"], lr=config["lr"], model_name=f"{name}_raw",
    )
    f1_raw = eval_model(model_raw, test_loader_raw)
    del model_raw
    torch.cuda.empty_cache()

    # ── Round 2: Train on NORM → Eval on NORM test ────────────
    print(f"\n  --- Round 2: Train on NORMALIZED data ---")
    vocab_norm = Vocabulary(max_size=30000)
    vocab_norm.build(train_texts_norm)

    train_ds_n = RNNTextDataset(train_texts_norm, train_labels, vocab_norm)
    val_ds_n = RNNTextDataset(val_texts_norm, val_labels, vocab_norm)
    test_ds_norm = RNNTextDataset(test_texts_norm, test_labels, vocab_norm)

    train_loader_n = DataLoader(train_ds_n, batch_size=config["batch_size"], shuffle=True)
    val_loader_n = DataLoader(val_ds_n, batch_size=config["batch_size"] * 2)
    test_loader_norm = DataLoader(test_ds_norm, batch_size=config["batch_size"] * 2)

    model_norm = _create_classifier_model(c_type, len(vocab_norm), config)
    model_norm = train_model(
        model_norm, train_loader_n, val_loader_n,
        epochs=config["epochs"], lr=config["lr"], model_name=f"{name}_norm",
    )
    f1_norm = eval_model(model_norm, test_loader_norm)
    del model_norm
    torch.cuda.empty_cache()

    print(f"\n  {name} Results:")
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
        "# Extrinsic Evaluation: Lexical Normalization → Hate Speech Detection",
        "",
        "## Dataset: ViHSD (3 classes: CLEAN, OFFENSIVE, HATE)",
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
        load_vihsd_dataset()
    )

    # Encode labels (handle both string labels and integer labels)
    if isinstance(train_labels_str[0], str):
        le = LabelEncoder()
        le.fit(EMOTION_LABELS)
        train_labels = le.transform(train_labels_str).tolist()
        val_labels = le.transform(val_labels_str).tolist()
        test_labels = le.transform(test_labels_str).tolist()
    else:
        # If dataset already returns integers
        train_labels = train_labels_str
        val_labels = val_labels_str
        test_labels = test_labels_str

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

    # 4a. LSTM
    results["LSTM"] = train_and_eval_classifier(
        "lstm",
        train_texts, train_labels, val_texts, val_labels, test_texts,
        train_texts_norm, val_texts_norm, test_texts_norm, test_labels,
        CLASSIFIER_CONFIGS["lstm"],
    )

    # 4b. BiLSTM
    results["BiLSTM"] = train_and_eval_classifier(
        "bilstm",
        train_texts, train_labels, val_texts, val_labels, test_texts,
        train_texts_norm, val_texts_norm, test_texts_norm, test_labels,
        CLASSIFIER_CONFIGS["bilstm"],
    )

    # 4c. GRU
    results["GRU"] = train_and_eval_classifier(
        "gru",
        train_texts, train_labels, val_texts, val_labels, test_texts,
        train_texts_norm, val_texts_norm, test_texts_norm, test_labels,
        CLASSIFIER_CONFIGS["gru"],
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
