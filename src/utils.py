"""
Utility functions for MTL training.

Includes:
  - Seed setting for reproducibility
  - Optimizer construction (discriminative LR)
  - Scheduler construction
  - Checkpoint save/load
  - Gradient statistics computation
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LinearLR, SequentialLR


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(
    model,
    encoder_lr: float = 3e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> AdamW:
    """
    Create AdamW optimizer with discriminative learning rates.

    - Shared encoder params: encoder_lr
    - Detection head params: head_lr
    - Decoder + LM head params: encoder_lr (same as encoder for seq2seq fine-tuning)

    No weight decay on bias and LayerNorm parameters.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    param_groups = [
        # Shared encoder — low LR
        {
            "params": [
                p for n, p in model.encoder.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "group_name": "encoder",
        },
        {
            "params": [
                p for n, p in model.encoder.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
            "group_name": "encoder_no_decay",
        },
        # Detection head — high LR
        {
            "params": [
                p for p in model.detection_head.parameters() if p.requires_grad
            ],
            "lr": head_lr,
            "weight_decay": weight_decay,
            "group_name": "detection_head",
        },
        # Decoder + LM head — same as encoder LR
        {
            "params": [
                p for n, p in model.decoder.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "group_name": "decoder",
        },
        {
            "params": [
                p for n, p in model.decoder.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
            "group_name": "decoder_no_decay",
        },
        {
            "params": [
                p for p in model.lm_head.parameters() if p.requires_grad
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "group_name": "lm_head",
        },
    ]

    # Filter out empty param groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    return AdamW(param_groups)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """
    Create linear warmup → linear decay scheduler.
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )

    decay_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=num_training_steps - num_warmup_steps,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_steps],
    )

    return scheduler


def compute_gradient_stats(
    model,
    det_loss: torch.Tensor,
    norm_loss: torch.Tensor,
    shared_params: List[nn.Parameter],
) -> Dict[str, float]:
    """
    Compute gradient statistics for training dynamics logging.

    Returns:
      - grad/det_grad_norm
      - grad/norm_grad_norm
      - grad/cosine_similarity
      - grad/conflict_detected
    """
    # Compute per-task gradients without modifying model grads
    det_grads = torch.autograd.grad(
        det_loss, shared_params, retain_graph=True, allow_unused=True
    )
    norm_grads = torch.autograd.grad(
        norm_loss, shared_params, retain_graph=True, allow_unused=True
    )

    # Flatten
    flat_det = torch.cat([
        g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(det_grads, shared_params)
    ])
    flat_norm = torch.cat([
        g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(norm_grads, shared_params)
    ])

    det_norm = flat_det.norm().item()
    norm_norm = flat_norm.norm().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_det.unsqueeze(0), flat_norm.unsqueeze(0)
    ).item()
    conflict = float(torch.dot(flat_det, flat_norm).item() < 0)

    return {
        "grad/det_grad_norm": det_norm,
        "grad/norm_grad_norm": norm_norm,
        "grad/cosine_similarity": cos_sim,
        "grad/conflict_detected": conflict,
    }


class EarlyStopping:
    """Early stopping based on a monitored metric."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs with no improvement to wait.
            mode:     "max" (higher is better) or "min" (lower is better).
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        improved = False
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    path: str,
):
    """Save training checkpoint."""
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        },
        save_dir / "checkpoint.pt",
    )

    # Also save the BARTpho model in HuggingFace format for easy loading
    model.bartpho.save_pretrained(str(save_dir / "bartpho"))

    print(f"  Checkpoint saved to {save_dir}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """Load training checkpoint."""
    ckpt = torch.load(Path(path) / "checkpoint.pt", map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt
