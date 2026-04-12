import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layer=4, dim_ff=1024, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layer,
            num_decoder_layers=num_layer,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        src_key_padding_mask = src_ids.eq(self.pad_token_id)
        tgt_key_padding_mask = tgt_ids.eq(self.pad_token_id)
        
        tgt_len = tgt_ids.size(1)
        causal_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(tgt_ids.device)

        src = self.pos_encoder(self.embed(src_ids) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.embed(tgt_ids) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src, 
            tgt=tgt, 
            tgt_mask=causal_mask, 
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Mask padding của encoder khi decoder nhìn vào
        )
        
        out = self.fc_out(out)
        return out

        
if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(vocab_size=160000).to(device)
    # Model size
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size}")

    # Giả lập batch-size=2, seq_len=128
    src_tensor = torch.randint(0, 10000, (2, 128)).to(device)
    tgt_tensor = torch.randint(0, 10000, (2, 128)).to(device)
    
    out = model(src_tensor, tgt_tensor)
    print(f"Output shape: {out.shape}") # Expect: [2, 128, 10000]
