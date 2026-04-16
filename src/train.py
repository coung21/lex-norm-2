"""
Training entry point for MTL Vietnamese Lexical Normalization.

Usage:
    # Single-task detection
    python src/train.py --mode detection_only --run_name st-detection

    # Single-task normalization
    python src/train.py --mode normalization_only --run_name st-normalization

    # MTL equal weighting
    python src/train.py --mode mtl --run_name mtl-equal

    # MTL + PCGrad
    python src/train.py --mode mtl --use_pcgrad --run_name mtl-pcgrad
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import wandb
from transformers import AutoTokenizer

from config import MTLConfig
from dataset import MTLDataset
from model import BARTphoMTL
from trainer import MTLTrainer
from utils import set_seed


def parse_args() -> MTLConfig:
    """Parse CLI arguments into MTLConfig."""
    parser = argparse.ArgumentParser(
        description="MTL BARTpho-syllable Lexical Normalization Training"
    )

    # Model
    parser.add_argument("--model_name", type=str, default="vinai/bartpho-syllable-base")

    # Task mode
    parser.add_argument(
        "--mode", type=str, default="mtl",
        choices=["detection_only", "normalization_only", "mtl"],
        help="Task mode: detection_only, normalization_only, or mtl"
    )
    parser.add_argument("--use_pcgrad", action="store_true", default=False,
                        help="Enable PCGrad for MTL training")
    parser.add_argument("--use_uncertainty", action="store_true", default=False,
                        help="Enable Uncertainty Weighting for MTL training")

    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    # Paths
    parser.add_argument("--train_file", type=str, default="data/ViLexNorm/data/train.csv")
    parser.add_argument("--dev_file", type=str, default="data/ViLexNorm/data/dev.csv")
    parser.add_argument("--test_file", type=str, default="data/ViLexNorm/data/test.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Evaluation
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--generation_max_length", type=int, default=128)

    # Logging
    parser.add_argument("--log_interval", type=int, default=50)

    # WandB
    parser.add_argument("--project", type=str, default="lexnorm2-mtl")
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()

    # Handle --no_fp16 flag
    if args.no_fp16:
        args.fp16 = False

    # Build config
    config = MTLConfig(
        model_name=args.model_name,
        mode=args.mode,
        use_pcgrad=args.use_pcgrad,
        use_uncertainty=args.use_uncertainty,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        head_lr=args.head_lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        beam_size=args.beam_size,
        generation_max_length=args.generation_max_length,
        log_interval=args.log_interval,
        project=args.project,
        run_name=args.run_name,
    )

    return config


def main():
    config = parse_args()

    # ── Reproducibility ────────────────────────────────────────────
    set_seed(config.seed)

    # ── WandB ──────────────────────────────────────────────────────
    wandb.init(
        project=config.project,
        name=config.run_name,
        config=asdict(config),
        tags=[config.mode, "pcgrad" if config.use_pcgrad else "standard"],
    )

    # ── Tokenizer ──────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # ── Model ──────────────────────────────────────────────────────
    print(f"Loading model: {config.model_name} (mode={config.mode})")
    model = BARTphoMTL(config.model_name, mode=config.mode)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Datasets ───────────────────────────────────────────────────
    print(f"\nLoading datasets...")
    train_ds = MTLDataset(config.train_file, tokenizer, config.max_seq_len)
    dev_ds = MTLDataset(config.dev_file, tokenizer, config.max_seq_len)
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Dev:   {len(dev_ds)} samples")

    # ── Print sample NSW labels ────────────────────────────────────
    print("\n── Sample NSW labels ──")
    for i in range(min(5, len(train_ds))):
        ex = train_ds.data.iloc[i]
        from dataset import create_nsw_labels
        labels = create_nsw_labels(str(ex["original"]), str(ex["normalized"]))
        orig_words = str(ex["original"]).split()
        print(f"  Original:   {ex['original']}")
        print(f"  Normalized: {ex['normalized']}")
        print(f"  NSW labels: {list(zip(orig_words, labels))}")
        print()

    # ── Trainer ────────────────────────────────────────────────────
    trainer = MTLTrainer(model, config, train_ds, dev_ds, tokenizer)

    # ── Train ──────────────────────────────────────────────────────
    trainer.train()

    # ── Finish ─────────────────────────────────────────────────────
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()