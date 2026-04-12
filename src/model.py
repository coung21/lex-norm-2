from transformers import BartConfig, BartForConditionalGeneration


def create_model(vocab_size, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                 d_model=512, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1, max_len=512):
    """Tạo BART model từ scratch với config tùy chỉnh."""
    config = BartConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=num_layers,
        decoder_layers=num_layers,
        encoder_attention_heads=nhead,
        decoder_attention_heads=nhead,
        encoder_ffn_dim=dim_ff,
        decoder_ffn_dim=dim_ff,
        max_position_embeddings=max_len,
        dropout=dropout,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=bos_token_id,
        forced_eos_token_id=eos_token_id,
    )
    model = BartForConditionalGeneration(config)
    return model


if __name__ == "__main__":
    import torch

    model = create_model(vocab_size=16000)

    # Model size
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size:,} parameters")

    # Sanity check forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    src = torch.randint(4, 16000, (2, 32)).to(device)
    tgt = torch.randint(4, 16000, (2, 32)).to(device)

    outputs = model(input_ids=src, labels=tgt)
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")  # [2, 32, 16000]

    # Sanity check generate
    generated = model.generate(input_ids=src, max_length=32)
    print(f"Generated shape: {generated.shape}")
    print("All checks passed!")
