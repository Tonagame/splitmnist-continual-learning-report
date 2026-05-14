"""Average Gradient Episodic Memory update step."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from core import ReplayBuffer, supervised_context_loss


def grad_vector(parameters: Sequence[nn.Parameter]) -> torch.Tensor:
    pieces = []
    for param in parameters:
        if param.grad is None:
            pieces.append(torch.zeros_like(param).flatten())
        else:
            pieces.append(param.grad.detach().flatten().clone())
    return torch.cat(pieces)


def set_grad_vector(parameters: Sequence[nn.Parameter], vector: torch.Tensor) -> None:
    pointer = 0
    for param in parameters:
        numel = param.numel()
        view = vector[pointer:pointer + numel].view_as(param)
        if param.grad is None:
            param.grad = view.clone()
        else:
            param.grad.copy_(view)
        pointer += numel


def agem_step(model, optimizer, params, replay: ReplayBuffer, x, y, args, context_index: int) -> None:
    """Project the current gradient if it conflicts with replay memory."""

    optimizer.zero_grad(set_to_none=True)
    loss_current = supervised_context_loss(model(x), y, args, context_index)
    loss_current.backward()
    grad_current = grad_vector(params)

    replay_batch = replay.sample(args.batch, x.device, with_signals=False)
    assert replay_batch is not None
    x_ref, y_ref = replay_batch
    optimizer.zero_grad(set_to_none=True)
    loss_ref = F.cross_entropy(model(x_ref), y_ref)
    loss_ref.backward()
    grad_ref = grad_vector(params)

    dot = torch.dot(grad_current, grad_ref)
    if dot < 0:
        grad_current = grad_current - dot / (torch.dot(grad_ref, grad_ref) + 1e-12) * grad_ref
    optimizer.zero_grad(set_to_none=True)
    set_grad_vector(params, grad_current)
    optimizer.step()
