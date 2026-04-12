import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from build_dataset import NormDataset
from model import Seq2SeqTransformer

import wandb
from wandb import Artifact

import argparse
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def shift_right(labels, bos_id, pad_id):
    labels = labels.clone()
    labels[labels == -100] = pad_id
    decoder_input_ids = torch.full_like(labels, pad_id)
    decoder_input_ids[:, 0] = bos_id
    decoder_input_ids[:, 1:] = labels[:, :-1]
    return decoder_input_ids

def run_train(tokenizer_dir, train_file, save_path, dev_path=None, epochs=5, batch_size=32):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_dir}/tokenizer.json",
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
    
    model = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id
    ).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            src_ids = torch.tensor(batch['input_ids'], device=DEVICE)
            labels = torch.tensor(batch['labels'], device=DEVICE)

            dec_ids = shift_right(labels, tokenizer.bos_token_id, tokenizer.pad_token_id)
            
            logits = model(src_ids, dec_ids)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1), ignore_index=-100)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        wandb.log({"train_loss": loss.item()})
        
        if dev_path:
            model.eval()
            with torch.no_grad():
                dev_loss = 0
                for batch in dev_dl:
                    src_ids = torch.tensor(batch['input_ids'], device=DEVICE)
                    labels = torch.tensor(batch['labels'], device=DEVICE)

                    dec_ids = shift_right(labels, tokenizer.bos_token_id, tokenizer.pad_token_id)
                    logits = model(src_ids, dec_ids)
                    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1), ignore_index=-100)
                    dev_loss += loss.item()
                dev_loss /= len(dev_dl)
                print(f"Epoch {epoch+1}/{epochs}, Dev Loss: {dev_loss}")
                wandb.log({"dev_loss": dev_loss})

    
    torch.save(model.state_dict(), f"{save_path}/model.pt")
    artifact = Artifact("model.pt", type="model")
    artifact.add_file(f"{save_path}/model.pt")
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
    save_path.mkdir(exist_ok=True)
    run_train(args.tokenizer_dir, args.train_file, save_path, args.dev_path, args.epochs, args.batch_size)    