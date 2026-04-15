"""
Evaluation metrics for MTL Lexical Normalization.

NSW Detection:
  - Precision / Recall / F1 (token-level, class=1=NSW)

Normalization:
  - ERR   (Error Reduction Rate)
  - Word Accuracy
  - BLEU-4
  - Exact Match
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import sacrebleu


# ─── Levenshtein distance ──────────────────────────────────────────────


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein edit distance between two lists."""
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


# ─── NSW Detection Metrics ─────────────────────────────────────────────


def compute_detection_metrics(
    all_preds: List[int],
    all_labels: List[int],
) -> Dict[str, float]:
    """
    Compute token-level Precision, Recall, F1 for NSW detection.

    Args:
        all_preds:  Flat list of predicted labels (0 or 1).
        all_labels: Flat list of ground-truth labels (0 or 1).
                    Labels == -100 are filtered out before computation.

    Returns:
        Dict with det_precision, det_recall, det_f1.
    """
    # Filter out ignored tokens (-100)
    filtered_preds = []
    filtered_labels = []
    for p, l in zip(all_preds, all_labels):
        if l != -100:
            filtered_preds.append(p)
            filtered_labels.append(l)

    if not filtered_labels:
        return {"det_precision": 0.0, "det_recall": 0.0, "det_f1": 0.0}

    # Binary classification: pos_label=1 (NSW class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_labels, filtered_preds, pos_label=1, average="binary", zero_division=0
    )

    return {
        "det_precision": float(precision) * 100,
        "det_recall": float(recall) * 100,
        "det_f1": float(f1) * 100,
    }


# ─── Normalization Metrics ──────────────────────────────────────────────


def compute_normalization_metrics(
    pred_texts: List[str],
    ref_texts: List[str],
    orig_texts: List[str],
) -> Dict[str, float]:
    """
    Compute normalization metrics.

    Args:
        pred_texts: Model predictions (decoded strings).
        ref_texts:  Ground-truth normalized texts.
        orig_texts: Original (noisy) input texts.

    Returns:
        Dict with norm_err, norm_word_acc, norm_bleu4, norm_exact_match.
    """
    assert len(pred_texts) == len(ref_texts) == len(orig_texts)

    total = len(pred_texts)
    if total == 0:
        return {
            "norm_err": 0.0,
            "norm_word_acc": 0.0,
            "norm_bleu4": 0.0,
            "norm_exact_match": 0.0,
        }

    # ── Exact Match ────────────────────────────────────────────────
    exact_matches = sum(p.strip() == r.strip() for p, r in zip(pred_texts, ref_texts))
    exact_match_rate = exact_matches / total * 100

    # ── Word-level metrics ─────────────────────────────────────────
    total_wer_baseline = 0  # WER(original vs normalized)
    total_wer_model = 0     # WER(prediction vs normalized)
    total_correct_words = 0
    total_ref_words = 0

    for pred, ref, orig in zip(pred_texts, ref_texts, orig_texts):
        ref_words = ref.strip().split()
        pred_words = pred.strip().split()
        orig_words = orig.strip().split()

        # WER for ERR calculation
        total_wer_baseline += edit_distance(ref_words, orig_words)
        total_wer_model += edit_distance(ref_words, pred_words)

        # Word accuracy
        total_ref_words += len(ref_words)
        correct = sum(
            1 for r, p in zip(ref_words, pred_words) if r == p
        )
        total_correct_words += correct

    # ERR = (WER_baseline - WER_model) / WER_baseline
    err = 0.0
    if total_wer_baseline > 0:
        err = (total_wer_baseline - total_wer_model) / total_wer_baseline * 100

    # Word Accuracy
    word_acc = total_correct_words / max(1, total_ref_words) * 100

    # ── BLEU-4 ─────────────────────────────────────────────────────
    # sacrebleu corpus-level BLEU
    bleu = sacrebleu.corpus_bleu(
        [p.strip() for p in pred_texts],
        [[r.strip() for r in ref_texts]],
    )

    return {
        "norm_err": float(err),
        "norm_word_acc": float(word_acc),
        "norm_bleu4": float(bleu.score),
        "norm_exact_match": float(exact_match_rate),
    }
