"""Elastic Weight Consolidation utilities."""

from __future__ import annotations

from typing import List, Tuple

import torch

from core import MLP, SplitMNISTContext, make_loader, move_batch, supervised_context_loss


def ewc_penalty(model: MLP, ewc_tasks: List[Tuple[List[torch.Tensor], List[torch.Tensor]]]) -> torch.Tensor:
    """Quadratic penalty that keeps important parameters near old values."""

    if not ewc_tasks:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    params = [p for p in model.parameters() if p.requires_grad]
    for means, fishers in ewc_tasks:
        for param, mean, fisher in zip(params, means, fishers):
            penalty = penalty + (fisher * (param - mean).pow(2)).sum()
    return penalty


def estimate_fisher(model: MLP, dataset: SplitMNISTContext, args, device: torch.device, context_index: int):
    """Estimate diagonal Fisher information after finishing one context."""

    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    fishers = [torch.zeros_like(p) for p in params]
    seen = 0
    batches = 0
    for batch in make_loader(dataset, args.batch, shuffle=True, drop_last=False):
        x, y, _original = move_batch(batch, device)
        model.zero_grad(set_to_none=True)
        loss = supervised_context_loss(model(x), y, args, context_index)
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        for fisher, grad in zip(fishers, grads):
            fisher += grad.detach().pow(2)
        seen += y.numel()
        batches += 1
        if seen >= args.fisher_samples:
            break
    if batches > 0:
        fishers = [fisher / batches for fisher in fishers]
    means = [p.detach().clone() for p in params]
    model.train()
    return means, fishers
