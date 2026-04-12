import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from build_dataset import NormDataset
from model import create_model

import wandb
from wandb import Artifact

import argparse
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(tokenizer_dir, train_file, save_path, dev_path=None, epochs=5, batch_size=32):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{tokenizer_dir}/tokenizer.json",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    train_ds = NormDataset(train_file, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if dev_path:
        dev_ds = NormDataset(dev_path, tokenizer)
        dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        num_batches = 0
        for batch in train_dl:
            src_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # BART tự động xử lý shift_right và tạo decoder_input_ids từ labels
            outputs = model(input_ids=src_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss})

        if dev_path:
            model.eval()
            with torch.no_grad():
                dev_loss = 0
                for batch in dev_dl:
                    src_ids = batch['input_ids'].to(DEVICE)
                    attn_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)

                    outputs = model(input_ids=src_ids, attention_mask=attn_mask, labels=labels)
                    dev_loss += outputs.loss.item()
                dev_loss /= len(dev_dl)
                print(f"Epoch {epoch+1}/{epochs}, Dev Loss: {dev_loss:.4f}")
                wandb.log({"dev_loss": dev_loss})
            model.train()

    # Lưu model theo chuẩn HuggingFace (config.json + model.safetensors)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    artifact = Artifact("model", type="model")
    artifact.add_dir(str(save_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    wandb.init(project="lexnorm2", name=args.run_name)
    save_path = Path("models") / args.run_name
    save_path.mkdir(exist_ok=True, parents=True)
    run_train(args.tokenizer_dir, args.train_file, save_path, args.dev_path, args.epochs, args.batch_size)