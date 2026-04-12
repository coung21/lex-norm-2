import pandas as pd 
import torch
from torch.utils.data import Dataset

class NormDataset(Dataset):
    def __init__(self, path, tokenizer, max_src_len=128, max_tgt_len=128):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data.iloc[idx]
        
        src_enc = self.tokenizer(
            ex['original'],
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        tgt_enc = self.tokenizer(
            ex['normalized'],
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        labels = tgt_enc['input_ids'][:]
        labels = [x if x != self.tokenizer.pad_token_id else -100 for x in labels]
        
        return {
            'input_ids': torch.tensor(src_enc['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(src_enc['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        