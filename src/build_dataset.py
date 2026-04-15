"""
Dataset cho ViLexNorm: chuẩn bị input/label cho T5 và ByT5.
- Input: "normalize: {original_text}"
- Label: "{normalized_text}"
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class NormDataset(Dataset):
    def __init__(self, path, tokenizer, max_src_len=128, max_tgt_len=128, prefix="normalize: "):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.prefix = prefix
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data.iloc[idx]

        src_text = self.prefix + str(ex['original'])
        tgt_text = str(ex['normalized'])

        src_enc = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
        )

        # Tokenize target
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
        )

        # Thay pad_token_id bằng -100 trong labels (ignore trong CrossEntropy)
        labels = [
            tok if tok != self.tokenizer.pad_token_id else -100
            for tok in tgt_enc['input_ids']
        ]

        return {
            'input_ids': src_enc['input_ids'],
            'attention_mask': src_enc['attention_mask'],
            'labels': labels,
        }