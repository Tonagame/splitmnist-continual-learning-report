"""Learning without Forgetting utilities."""

from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from core import MLP


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled KL loss used for knowledge distillation."""

    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    target_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean") * (temperature ** 2)


def frozen_teacher(model: MLP, device: torch.device) -> MLP:
    """Snapshot the current model so later contexts can distill from it."""

    teacher = copy.deepcopy(model).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher
