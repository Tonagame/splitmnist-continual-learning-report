"""Joint Training upper-bound method."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence, Tuple

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from core import MLP, SplitMNISTContext, display_method, evaluate_neural, make_loader, move_batch, next_batch, record_eval


def train_joint(args, train_contexts: Sequence[SplitMNISTContext], test_contexts: Sequence[SplitMNISTContext], output_dim: int, device: torch.device) -> Tuple[float, List[Dict[str, object]], float]:
    start = time.time()
    method_name = display_method(args.method)
    model = MLP(output_dim=output_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = ConcatDataset(train_contexts)
    loader = make_loader(train_dataset, args.batch, shuffle=True, drop_last=True)
    iterator = None
    history: List[Dict[str, object]] = []
    total_steps = args.iters * args.contexts

    for step in range(1, total_steps + 1):
        batch, iterator = next_batch(loader, iterator)
        x, y, _original = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if args.eval_every and step % args.eval_every == 0:
            context = min(args.contexts, math.ceil(step / args.iters))
            acc = evaluate_neural(model, test_contexts, args, device, acc_n=args.acc_n)
            record_eval(history, args, method_name, step, context, acc)

    final_acc_n = None if args.final_acc_n == 0 else args.final_acc_n
    final_accuracy = evaluate_neural(model, test_contexts, args, device, acc_n=final_acc_n)
    runtime = time.time() - start
    return final_accuracy, history, runtime
