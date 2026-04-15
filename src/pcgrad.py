"""
PCGrad — Projecting Conflicting Gradients for Multi-Task Learning.

Reference:
    Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
    https://proceedings.neurips.cc/paper/2020/hash/3fe78a89279a0d33f269a896f600f681-Abstract.html

Algorithm:
    1.  Compute per-task gradients independently.
    2.  For each pair (g_i, g_j), if g_i · g_j < 0 (conflict):
            g_i' = g_i − (g_i · g_j / ||g_j||²) · g_j
        Otherwise keep g_i unchanged.
    3.  Final gradient = mean of all (possibly projected) task gradients.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class PCGrad:
    """
    PCGrad optimizer wrapper.

    Wraps an existing optimizer and modifies gradients to resolve conflicts
    between task-specific gradient directions on shared parameters.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def step(
        self,
        task_losses: List[torch.Tensor],
        shared_params: List[nn.Parameter],
        retain_graph: bool = False,
    ) -> Dict[str, float]:
        """
        Perform one PCGrad optimization step.

        Args:
            task_losses:   List of scalar losses, one per task.
            shared_params: Shared encoder parameters to apply PCGrad on.
            retain_graph:  Whether to retain the computation graph.

        Returns:
            Dictionary of gradient statistics for logging:
              - cosine_similarity
              - conflict_detected (bool as float)
              - det_grad_norm, norm_grad_norm
              - projection_magnitude
        """
        num_tasks = len(task_losses)
        assert num_tasks >= 2, "PCGrad requires at least 2 tasks"

        # ── 1. Compute per-task gradients ──────────────────────────
        task_grads: List[torch.Tensor] = []

        for i, loss in enumerate(task_losses):
            self.optimizer.zero_grad()
            # retain_graph=True for all except possibly the last
            loss.backward(retain_graph=(i < num_tasks - 1) or retain_graph)

            # Collect and flatten gradients for shared params
            grads = []
            for p in shared_params:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().flatten())
                else:
                    grads.append(torch.zeros(p.numel(), device=p.device))
            task_grads.append(torch.cat(grads))

        # ── 2. Compute statistics (before projection) ─────────────
        cos_sim = torch.nn.functional.cosine_similarity(
            task_grads[0].unsqueeze(0), task_grads[1].unsqueeze(0)
        ).item()

        det_grad_norm = task_grads[0].norm().item()
        norm_grad_norm = task_grads[1].norm().item()

        dot_product = torch.dot(task_grads[0], task_grads[1]).item()
        conflict = dot_product < 0

        # ── 3. Project conflicting gradients ──────────────────────
        projected = [g.clone() for g in task_grads]
        projection_magnitude = 0.0

        if conflict:
            for i in range(num_tasks):
                for j in range(num_tasks):
                    if i == j:
                        continue
                    dot = torch.dot(projected[i], task_grads[j])
                    if dot < 0:
                        # Project g_i onto the normal plane of g_j
                        proj = (dot / (task_grads[j].norm() ** 2 + 1e-8)) * task_grads[j]
                        projected[i] = projected[i] - proj
                        projection_magnitude += proj.norm().item()

        # ── 4. Average projected gradients ────────────────────────
        final_grad = torch.stack(projected).mean(dim=0)

        # ── 5. Unflatten and assign gradients ─────────────────────
        self.optimizer.zero_grad()
        offset = 0
        for p in shared_params:
            numel = p.numel()
            p.grad = final_grad[offset: offset + numel].view_as(p).clone()
            offset += numel

        # ── 6. Optimizer step ─────────────────────────────────────
        self.optimizer.step()

        return {
            "cosine_similarity": cos_sim,
            "conflict_detected": float(conflict),
            "det_grad_norm": det_grad_norm,
            "norm_grad_norm": norm_grad_norm,
            "projection_magnitude": projection_magnitude,
        }


def pcgrad_backward(
    task_losses: List[torch.Tensor],
    shared_params: List[nn.Parameter],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Functional version: compute PCGrad-modified gradients without stepping.

    Returns:
        final_grad: Flattened projected gradient tensor.
        stats:      Gradient statistics dict.
    """
    num_tasks = len(task_losses)
    task_grads = []

    for i, loss in enumerate(task_losses):
        grads = torch.autograd.grad(
            loss, shared_params,
            retain_graph=True,
            allow_unused=True,
        )
        flat = torch.cat([
            g.detach().flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
            for g, p in zip(grads, shared_params)
        ])
        task_grads.append(flat)

    # Stats
    cos_sim = torch.nn.functional.cosine_similarity(
        task_grads[0].unsqueeze(0), task_grads[1].unsqueeze(0)
    ).item()
    conflict = torch.dot(task_grads[0], task_grads[1]).item() < 0

    # Project
    projected = [g.clone() for g in task_grads]
    proj_mag = 0.0
    if conflict:
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue
                dot = torch.dot(projected[i], task_grads[j])
                if dot < 0:
                    proj = (dot / (task_grads[j].norm() ** 2 + 1e-8)) * task_grads[j]
                    projected[i] = projected[i] - proj
                    proj_mag += proj.norm().item()

    final_grad = torch.stack(projected).mean(dim=0)

    stats = {
        "cosine_similarity": cos_sim,
        "conflict_detected": float(conflict),
        "det_grad_norm": task_grads[0].norm().item(),
        "norm_grad_norm": task_grads[1].norm().item(),
        "projection_magnitude": proj_mag,
    }

    return final_grad, stats
