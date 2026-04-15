"""
MTL Dataset for ViLexNorm.

Produces both:
  - NSW detection labels (token-level binary classification)
  - Normalization labels  (seq2seq generation target)

NSW labels are constructed via word alignment between original and normalized text.
"""

from difflib import SequenceMatcher
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


# ─── NSW label construction ────────────────────────────────────────────


def create_nsw_labels(original: str, normalized: str) -> List[int]:
    """
    Create word-level NSW labels by aligning original ↔ normalized tokens.

    Returns a list of labels (one per original token):
      0 = standard word (unchanged)
      1 = non-standard word (changed/deleted)

    Uses SequenceMatcher to handle insertions, deletions, and replacements.
    """
    orig_tokens = original.strip().split()
    norm_tokens = normalized.strip().split()

    if not orig_tokens:
        return []

    labels = [0] * len(orig_tokens)

    matcher = SequenceMatcher(None, orig_tokens, norm_tokens)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "replace":
            # Tokens differ → mark as NSW
            for i in range(i1, i2):
                labels[i] = 1
        elif op == "delete":
            # Tokens only in original → NSW
            for i in range(i1, i2):
                labels[i] = 1
        elif op == "insert":
            # Tokens only in normalized → mark adjacent original token
            if i1 < len(labels):
                labels[min(i1, len(labels) - 1)] = 1
        # op == "equal" → keep 0

    return labels


def align_labels_to_subwords(
    word_labels: List[int],
    word_ids: List,
) -> List[int]:
    """
    Broadcast word-level NSW labels to subword-level.

    Args:
        word_labels: One label per original word.
        word_ids:    Output of tokenizer's word_ids() — maps each subword
                     to its originating word index (None for special tokens).

    Returns:
        Subword-level labels. Special tokens and continuation subwords get -100
        (ignored in CrossEntropy loss). Only the first subword of each word
        gets the real label.
    """
    subword_labels = []
    prev_word_id = None

    for wid in word_ids:
        if wid is None:
            # Special token (BOS, EOS, PAD)
            subword_labels.append(-100)
        elif wid != prev_word_id:
            # First subword of a new word → use real label
            if wid < len(word_labels):
                subword_labels.append(word_labels[wid])
            else:
                subword_labels.append(-100)
        else:
            # Continuation subword → ignore
            subword_labels.append(-100)
        prev_word_id = wid

    return subword_labels


# ─── Dataset ───────────────────────────────────────────────────────────


class MTLDataset(Dataset):
    """
    Multi-Task Learning dataset.

    Each sample provides:
      - input_ids, attention_mask  (encoder input — original text)
      - detection_labels           (NSW labels aligned to subwords)
      - decoder_input_ids          (shifted-right normalized tokens)
      - labels                     (normalized token IDs, -100 for padding)
    """

    def __init__(self, path: str, tokenizer, max_seq_len: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = pd.read_csv(path)

        # Ensure columns exist
        assert "original" in self.data.columns, f"Missing 'original' column in {path}"
        assert "normalized" in self.data.columns, f"Missing 'normalized' column in {path}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.data.iloc[idx]
        original = str(ex["original"]).strip()
        normalized = str(ex["normalized"]).strip()

        # ── Encoder input (original text) ──────────────────────────
        src_enc = self.tokenizer(
            original,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
            return_offsets_mapping=False,
        )

        # ── NSW detection labels ───────────────────────────────────
        word_labels = create_nsw_labels(original, normalized)

        # Get word_ids for subword alignment
        # BARTpho (SentencePiece) — use tokenizer to get word mapping
        src_enc_with_words = self.tokenizer(
            original,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        # Build word_ids manually for SentencePiece tokenizers
        word_ids = self._get_word_ids(original)
        detection_labels = align_labels_to_subwords(word_labels, word_ids)

        # Pad/truncate detection labels to match input_ids length
        seq_len = len(src_enc["input_ids"])
        if len(detection_labels) < seq_len:
            detection_labels += [-100] * (seq_len - len(detection_labels))
        else:
            detection_labels = detection_labels[:seq_len]

        # ── Decoder input (normalized text → target) ───────────────
        tgt_enc = self.tokenizer(
            normalized,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        # Labels: replace pad_token_id with -100 (ignored in loss)
        labels = [
            tok if tok != self.tokenizer.pad_token_id else -100
            for tok in tgt_enc["input_ids"]
        ]

        # Decoder input ids: shifted right (prepend decoder_start_token_id)
        decoder_start_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.special_tokens_map.get("cls_token", "<s>")
        )
        if decoder_start_id is None or decoder_start_id == self.tokenizer.unk_token_id:
            decoder_start_id = self.tokenizer.bos_token_id or 0

        decoder_input_ids = [decoder_start_id] + tgt_enc["input_ids"][:-1]

        return {
            "input_ids": torch.tensor(src_enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(src_enc["attention_mask"], dtype=torch.long),
            "detection_labels": torch.tensor(detection_labels, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _get_word_ids(self, text: str) -> List:
        """
        Build word_ids mapping for SentencePiece-based tokenizers.

        Maps each subword token back to its originating word index.
        Returns None for special tokens (BOS, EOS, PAD).
        """
        words = text.strip().split()
        if not words:
            return [None] * self.max_seq_len

        # Tokenize each word individually to determine subword count
        word_ids = []

        # First: add None for BOS token
        word_ids.append(None)

        for word_idx, word in enumerate(words):
            word_tokens = self.tokenizer.tokenize(word)
            word_ids.extend([word_idx] * len(word_tokens))

        # Add None for EOS token
        word_ids.append(None)

        # Truncate if needed (account for max_seq_len)
        if len(word_ids) > self.max_seq_len:
            word_ids = word_ids[: self.max_seq_len]
            # Ensure last token maps to None (EOS)
            word_ids[-1] = None

        # Pad with None
        while len(word_ids) < self.max_seq_len:
            word_ids.append(None)

        return word_ids
