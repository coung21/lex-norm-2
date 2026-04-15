"""
Configuration for MTL Vietnamese Lexical Normalization.
Dataclass-based config for all hyperparameters.
"""

from dataclasses import dataclass, field


@dataclass
class MTLConfig:
    """Configuration for Multi-Task Learning training."""

    # ── Model ──────────────────────────────────────────
    model_name: str = "vinai/bartpho-syllable-base"

    # ── Task mode ──────────────────────────────────────
    # "detection_only", "normalization_only", "mtl"
    mode: str = "mtl"
    use_pcgrad: bool = False

    # ── Training ───────────────────────────────────────
    epochs: int = 15
    batch_size: int = 8
    lr: float = 3e-5            # Encoder learning rate
    head_lr: float = 1e-4       # Task head learning rate
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    seed: int = 42

    # ── Data ───────────────────────────────────────────
    max_seq_len: int = 128
    num_workers: int = 2

    # ── Paths ──────────────────────────────────────────
    train_file: str = "data/ViLexNorm/data/train.csv"
    dev_file: str = "data/ViLexNorm/data/dev.csv"
    test_file: str = "data/ViLexNorm/data/test.csv"
    output_dir: str = "outputs"

    # ── Evaluation ─────────────────────────────────────
    eval_strategy: str = "epoch"  # "epoch" or "steps"
    eval_steps: int = 500
    beam_size: int = 4
    generation_max_length: int = 128

    # ── Logging ────────────────────────────────────────
    log_interval: int = 50
    save_total_limit: int = 2

    # ── WandB ──────────────────────────────────────────
    project: str = "lexnorm2-mtl"
    run_name: str = ""
