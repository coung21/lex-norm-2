"""
Custom MTL Trainer for BARTpho-based Lexical Normalization.

Handles:
  - Standard training (single-task and equal-weight MTL)
  - PCGrad training (gradient surgery for MTL)
  - Gradient accumulation with FP16
  - Comprehensive WandB logging (losses, gradients, LR, metrics)
  - Evaluation with detection + normalization metrics
"""

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.amp

import wandb

from config import MTLConfig
from model import BARTphoMTL
from pcgrad import PCGrad
from metrics import compute_detection_metrics, compute_normalization_metrics
from utils import (
    get_optimizer,
    get_scheduler,
    compute_gradient_stats,
    save_checkpoint,
    EarlyStopping,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MTLTrainer:
    """Custom training loop for MTL lexical normalization."""

    def __init__(
        self,
        model: BARTphoMTL,
        config: MTLConfig,
        train_dataset,
        dev_dataset,
        tokenizer,
    ):
        self.model = model.to(DEVICE)
        self.config = config
        self.tokenizer = tokenizer

        # ── Data loaders ───────────────────────────────────────────
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.dev_loader = DataLoader(
            dev_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # ── Optimizer ──────────────────────────────────────────────
        base_optimizer = get_optimizer(
            model,
            encoder_lr=config.lr,
            head_lr=config.head_lr,
            weight_decay=config.weight_decay,
        )

        if config.use_pcgrad and config.mode == "mtl":
            self.pcgrad = PCGrad(base_optimizer)
            self.optimizer = base_optimizer  # Keep reference for scheduler
        else:
            self.pcgrad = None
            self.optimizer = base_optimizer

        # ── Scheduler ──────────────────────────────────────────────
        num_update_steps = (
            len(self.train_loader) // config.gradient_accumulation_steps * config.epochs
        )
        num_warmup_steps = int(num_update_steps * config.warmup_ratio)

        self.scheduler = get_scheduler(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_update_steps,
        )

        # ── FP16 ───────────────────────────────────────────────────
        self.fp16 = config.fp16 and torch.cuda.is_available()
        self.scaler = GradScaler() if self.fp16 else None

        # ── Early stopping ─────────────────────────────────────────
        if config.mode in ("detection_only",):
            self.early_stopping = EarlyStopping(patience=5, mode="max")  # F1
        elif config.mode in ("normalization_only",):
            self.early_stopping = EarlyStopping(patience=5, mode="max")  # BLEU
        else:
            self.early_stopping = EarlyStopping(patience=5, mode="max")  # F1 or BLEU

        # ── Tracking ───────────────────────────────────────────────
        self.global_step = 0
        self.best_metric = 0.0
        self.conflict_count = 0
        self.total_mtl_steps = 0

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  Training: {self.config.run_name}")
        print(f"  Mode: {self.config.mode} | PCGrad: {self.config.use_pcgrad}")
        print(f"  Epochs: {self.config.epochs} | Batch: {self.config.batch_size}")
        print(f"  Encoder LR: {self.config.lr} | Head LR: {self.config.head_lr}")
        print(f"  FP16: {self.fp16}")
        print(f"  Device: {DEVICE}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
            print(f"\n── Epoch {epoch + 1}/{self.config.epochs} ──")

            train_metrics = self._train_one_epoch(epoch)

            # Evaluation
            eval_metrics = self.evaluate()

            # Log epoch summary
            self._log_epoch(epoch, train_metrics, eval_metrics)

            # Early stopping & checkpoint
            primary_metric = self._get_primary_metric(eval_metrics)
            if primary_metric > self.best_metric:
                self.best_metric = primary_metric
                save_dir = str(Path(self.config.output_dir) / self.config.run_name / "best")
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, eval_metrics, save_dir,
                )
                # Save tokenizer alongside model
                self.tokenizer.save_pretrained(
                    str(Path(self.config.output_dir) / self.config.run_name / "best" / "bartpho")
                )

            if self.early_stopping(primary_metric):
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        # Final summary
        print(f"\n{'='*60}")
        print(f"  Training complete! Best metric: {self.best_metric:.2f}")
        if self.total_mtl_steps > 0:
            print(f"  Gradient conflict rate: {self.conflict_count / self.total_mtl_steps * 100:.1f}%")
        print(f"{'='*60}\n")

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch. Returns aggregated metrics."""
        self.model.train()
        accum = self.config.gradient_accumulation_steps

        epoch_det_loss = 0.0
        epoch_norm_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if self.pcgrad is not None:
                # ── PCGrad path ────────────────────────────────────
                pcgrad_stats = self._pcgrad_step(batch)

                det_loss_val = pcgrad_stats.get("det_loss", 0.0)
                norm_loss_val = pcgrad_stats.get("norm_loss", 0.0)
                total_loss_val = det_loss_val + norm_loss_val

                # Track conflicts
                self.total_mtl_steps += 1
                if pcgrad_stats.get("conflict_detected", 0) > 0.5:
                    self.conflict_count += 1

                # Log PCGrad stats
                if self.global_step % self.config.log_interval == 0:
                    wandb.log({
                        "pcgrad/cosine_similarity": pcgrad_stats["cosine_similarity"],
                        "pcgrad/conflict_detected": pcgrad_stats["conflict_detected"],
                        "pcgrad/projection_magnitude": pcgrad_stats["projection_magnitude"],
                        "pcgrad/det_grad_norm": pcgrad_stats["det_grad_norm"],
                        "pcgrad/norm_grad_norm": pcgrad_stats["norm_grad_norm"],
                        "train/step": self.global_step,
                    })

                self.scheduler.step()

            else:
                # ── Standard path (single-task or equal-weight MTL) ─
                loss_info = self._standard_step(batch, step, accum)

                det_loss_val = loss_info.get("det_loss", 0.0)
                norm_loss_val = loss_info.get("norm_loss", 0.0)
                total_loss_val = loss_info["total_loss"]

                # Gradient accumulation step
                if (step + 1) % accum == 0:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    if self.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Gradient tracking for MTL (non-PCGrad)
                if (
                    self.config.mode == "mtl"
                    and self.global_step % self.config.log_interval == 0
                    and "det_loss_tensor" in loss_info
                    and "norm_loss_tensor" in loss_info
                ):
                    try:
                        grad_stats = compute_gradient_stats(
                            self.model,
                            loss_info["det_loss_tensor"],
                            loss_info["norm_loss_tensor"],
                            self.model.get_shared_params(),
                        )
                        wandb.log({**grad_stats, "train/step": self.global_step})
                        self.total_mtl_steps += 1
                        if grad_stats["grad/conflict_detected"] > 0.5:
                            self.conflict_count += 1
                    except RuntimeError:
                        pass  # Graph may have been freed

            # Accumulate epoch losses
            epoch_det_loss += det_loss_val
            epoch_norm_loss += norm_loss_val
            epoch_total_loss += total_loss_val
            num_batches += 1

            # Log step metrics
            if self.global_step % self.config.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                log_dict = {
                    "train/loss_det": det_loss_val,
                    "train/loss_norm": norm_loss_val,
                    "train/loss_total": total_loss_val,
                    "train/lr": lr,
                    "train/step": self.global_step,
                    "train/epoch": epoch,
                }
                
                if self.config.use_uncertainty:
                    log_dict["train/uncertainty_det"] = self.model.log_var_det.item()
                    log_dict["train/uncertainty_norm"] = self.model.log_var_norm.item()
                    
                wandb.log(log_dict)

                if (step + 1) % (self.config.log_interval * 5) == 0:
                    print(
                        f"  Step {step+1}/{len(self.train_loader)} | "
                        f"Loss: {total_loss_val:.4f} "
                        f"(det={det_loss_val:.4f}, norm={norm_loss_val:.4f}) | "
                        f"LR: {lr:.2e}"
                    )

            self.global_step += 1

        # Return epoch averages
        n = max(1, num_batches)
        return {
            "train/epoch_loss_det": epoch_det_loss / n,
            "train/epoch_loss_norm": epoch_norm_loss / n,
            "train/epoch_loss_total": epoch_total_loss / n,
        }

    def _standard_step(self, batch: Dict, step: int, accum: int) -> Dict:
        """Standard forward + backward for single-task or equal-weight MTL."""
        amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if self.fp16 else torch.amp.autocast('cuda', enabled=False)

        with amp_ctx:
            outputs = self.model(**batch)

        # Compute total loss
        det_loss = outputs.get("det_loss", None)
        norm_loss = outputs.get("norm_loss", None)

        total_loss = torch.tensor(0.0, device=DEVICE)
        
        if getattr(self.config, "use_uncertainty", False) and self.config.mode == "mtl":
            if det_loss is not None:
                total_loss = total_loss + 0.5 * torch.exp(-self.model.log_var_det) * det_loss + 0.5 * self.model.log_var_det
            if norm_loss is not None:
                total_loss = total_loss + 0.5 * torch.exp(-self.model.log_var_norm) * norm_loss + 0.5 * self.model.log_var_norm
        else:
            if det_loss is not None:
                total_loss = total_loss + det_loss
            if norm_loss is not None:
                total_loss = total_loss + norm_loss

        # Scale for gradient accumulation
        scaled_loss = total_loss / accum

        if self.fp16:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return {
            "det_loss": det_loss.item() if det_loss is not None else 0.0,
            "norm_loss": norm_loss.item() if norm_loss is not None else 0.0,
            "total_loss": total_loss.item(),
            "det_loss_tensor": det_loss,
            "norm_loss_tensor": norm_loss,
        }

    def _pcgrad_step(self, batch: Dict) -> Dict:
        """PCGrad step: separate gradients, project conflicts, update."""
        # Forward (no AMP for PCGrad — need separate backward passes)
        outputs = self.model(**batch)

        det_loss = outputs["det_loss"]
        norm_loss = outputs["norm_loss"]

        # PCGrad step handles backward + optimizer.step internally
        pcgrad_stats = self.pcgrad.step(
            task_losses=[det_loss, norm_loss],
            shared_params=self.model.get_shared_params(),
            retain_graph=False,
        )

        # Also update task-specific heads with their own gradients
        # (PCGrad only modifies shared encoder gradients)
        self.optimizer.zero_grad()
        # Re-forward for head gradients
        outputs2 = self.model(**batch)
        head_loss = outputs2["det_loss"] + outputs2["norm_loss"]
        head_loss.backward()

        # Zero out encoder grads (already updated by PCGrad)
        for p in self.model.get_shared_params():
            if p.grad is not None:
                p.grad.zero_()

        # Step optimizer for head params only
        nn.utils.clip_grad_norm_(
            list(self.model.get_detection_params()) + list(self.model.get_normalization_params()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        pcgrad_stats["det_loss"] = det_loss.item()
        pcgrad_stats["norm_loss"] = norm_loss.item()

        return pcgrad_stats

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on dev set."""
        self.model.eval()

        # ── Detection predictions ──────────────────────────────────
        all_det_preds = []
        all_det_labels = []

        # ── Normalization predictions ──────────────────────────────
        all_pred_texts = []
        all_ref_texts = []
        all_orig_texts = []

        for batch in self.dev_loader:
            batch_device = {k: v.to(DEVICE) for k, v in batch.items()}

            # Detection
            if self.config.mode in ("detection_only", "mtl"):
                det_preds = self.model.predict_detection(
                    batch_device["input_ids"],
                    batch_device["attention_mask"],
                )
                # Collect preds and labels (flatten)
                det_labels = batch["detection_labels"]
                for i in range(det_preds.size(0)):
                    mask = det_labels[i] != -100
                    all_det_preds.extend(det_preds[i][mask].cpu().tolist())
                    all_det_labels.extend(det_labels[i][mask].cpu().tolist())

            # Normalization (generate)
            if self.config.mode in ("normalization_only", "mtl"):
                generated_ids = self.model.generate(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                    max_length=self.config.generation_max_length,
                    num_beams=self.config.beam_size,
                    early_stopping=True,
                )

                pred_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                all_pred_texts.extend([t.strip() for t in pred_texts])

                # Decode reference and original from batch
                ref_ids = batch["labels"].clone()
                ref_ids[ref_ids == -100] = self.tokenizer.pad_token_id
                ref_texts = self.tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                all_ref_texts.extend([t.strip() for t in ref_texts])

                orig_texts = self.tokenizer.batch_decode(
                    batch["input_ids"], skip_special_tokens=True
                )
                all_orig_texts.extend([t.strip() for t in orig_texts])

        # ── Compute metrics ────────────────────────────────────────
        metrics = {}

        if self.config.mode in ("detection_only", "mtl") and all_det_labels:
            det_metrics = compute_detection_metrics(all_det_preds, all_det_labels)
            metrics.update({f"eval/{k}": v for k, v in det_metrics.items()})

        if self.config.mode in ("normalization_only", "mtl") and all_ref_texts:
            norm_metrics = compute_normalization_metrics(
                all_pred_texts, all_ref_texts, all_orig_texts
            )
            metrics.update({f"eval/{k}": v for k, v in norm_metrics.items()})

        # Add conflict rate for MTL
        if self.total_mtl_steps > 0:
            metrics["eval/conflict_rate"] = self.conflict_count / self.total_mtl_steps * 100

        self.model.train()
        return metrics

    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get primary metric for early stopping / best model selection."""
        if self.config.mode == "detection_only":
            return metrics.get("eval/det_f1", 0.0)
        elif self.config.mode == "normalization_only":
            return metrics.get("eval/norm_bleu4", 0.0)
        else:
            # MTL: use average of F1 and BLEU
            f1 = metrics.get("eval/det_f1", 0.0)
            bleu = metrics.get("eval/norm_bleu4", 0.0)
            return (f1 + bleu) / 2

    def _log_epoch(self, epoch: int, train_metrics: Dict, eval_metrics: Dict):
        """Log epoch summary to console and WandB."""
        all_metrics = {**train_metrics, **eval_metrics, "epoch": epoch + 1}
        wandb.log(all_metrics)

        # Console output
        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    Train Loss: {train_metrics['train/epoch_loss_total']:.4f}")

        if self.config.mode in ("detection_only", "mtl"):
            p = eval_metrics.get("eval/det_precision", 0)
            r = eval_metrics.get("eval/det_recall", 0)
            f1 = eval_metrics.get("eval/det_f1", 0)
            print(f"    Detection — P: {p:.1f} | R: {r:.1f} | F1: {f1:.1f}")

        if self.config.mode in ("normalization_only", "mtl"):
            err = eval_metrics.get("eval/norm_err", 0)
            wa = eval_metrics.get("eval/norm_word_acc", 0)
            bleu = eval_metrics.get("eval/norm_bleu4", 0)
            em = eval_metrics.get("eval/norm_exact_match", 0)
            print(f"    Normalization — ERR: {err:.1f} | WordAcc: {wa:.1f} | BLEU: {bleu:.1f} | EM: {em:.1f}")

        if "eval/conflict_rate" in eval_metrics:
            print(f"    Gradient Conflict Rate: {eval_metrics['eval/conflict_rate']:.1f}%")

        primary = self._get_primary_metric(eval_metrics)
        print(f"    Primary Metric: {primary:.2f} (best: {self.best_metric:.2f})")
