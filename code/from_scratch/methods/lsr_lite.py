"""LSR-lite replay, distillation, feature anchoring, Fourier, and ASW losses."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from core import ReplayBuffer
from methods.lwf import kd_loss


def fourier_loss(current_features: torch.Tensor, stored_features: torch.Tensor) -> torch.Tensor:
    current_spec = torch.log1p(torch.abs(torch.fft.rfft(current_features.float(), dim=1)))
    stored_spec = torch.log1p(torch.abs(torch.fft.rfft(stored_features.float(), dim=1)))
    return F.mse_loss(current_spec, stored_spec)


def lsr_options(method: str) -> Tuple[bool, bool, bool]:
    is_lsr = method.startswith("lsr-lite")
    use_fourier = method in ("lsr-lite-fourier", "lsr-lite-fourier-asw")
    use_asw = method in ("lsr-lite-asw", "lsr-lite-fourier-asw")
    return is_lsr, use_fourier, use_asw


def add_lsr_replay_loss(model, loss, replay: ReplayBuffer, args, use_fourier: bool, use_asw: bool, factors: List[float]):
    """Add the LSR-lite stability terms to the current supervised loss."""

    if len(replay) == 0:
        return loss

    replay_batch = replay.sample(args.batch, next(model.parameters()).device, with_signals=True)
    assert replay_batch is not None
    x_rep, y_rep, old_logits, old_features = replay_batch
    rep_features = model.features(x_rep)
    rep_logits = model.classifier(rep_features)
    replay_ce = F.cross_entropy(rep_logits, y_rep)
    kd = kd_loss(rep_logits, old_logits, args.temperature)
    feature_anchor = F.mse_loss(rep_features, old_features)

    factor = 1.0
    if use_asw:
        old_signal = (replay_ce.detach() + kd.detach() + feature_anchor.detach()).item()
        new_signal = max(loss.detach().item(), args.asw_epsilon)
        factor = old_signal / (new_signal + args.asw_epsilon)
        factor = max(args.asw_min, min(args.asw_max, factor))
        factors.append(float(factor))

    loss = loss + args.lsr_replay_ce_lambda * replay_ce
    loss = loss + args.lsr_kd_lambda * factor * kd
    loss = loss + args.lsr_feature_lambda * factor * feature_anchor
    if use_fourier:
        loss = loss + args.lsr_fourier_lambda * fourier_loss(rep_features, old_features)
    return loss


def asw_summary(factors: List[float]):
    if not factors:
        return {}
    return {
        "asw_mean": float(np.mean(factors)),
        "asw_min": float(np.min(factors)),
        "asw_max": float(np.max(factors)),
    }
