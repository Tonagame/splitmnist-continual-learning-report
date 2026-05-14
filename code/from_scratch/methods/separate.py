"""Separate Networks for Task-CL."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import torch
from torch import optim
from torch.nn import functional as F

from core import DIGIT_CONTEXTS, MLP, SplitMNISTContext, display_method, evaluate_separate, make_loader, move_batch, next_batch, record_eval


def train_separate(args, train_contexts: Sequence[SplitMNISTContext], test_contexts: Sequence[SplitMNISTContext], device: torch.device) -> Tuple[float, List[Dict[str, object]], float]:
    start = time.time()
    method_name = display_method(args.method)
    models = [MLP(output_dim=2, hidden=args.hidden, dropout=args.dropout).to(device) for _ in train_contexts]
    history: List[Dict[str, object]] = []
    global_step = 0

    for context_index, train_dataset in enumerate(train_contexts):
        model = models[context_index]
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = make_loader(train_dataset, args.batch, shuffle=True, drop_last=True)
        iterator = None
        for _step in range(1, args.iters + 1):
            global_step += 1
            batch, iterator = next_batch(loader, iterator)
            x, _y, original = move_batch(batch, device)
            target = original - DIGIT_CONTEXTS[context_index][0]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), target)
            loss.backward()
            optimizer.step()
            if args.eval_every and global_step % args.eval_every == 0:
                acc = evaluate_separate(models, test_contexts, args, device, acc_n=args.acc_n)
                record_eval(history, args, method_name, global_step, context_index + 1, acc)

    final_acc_n = None if args.final_acc_n == 0 else args.final_acc_n
    final_accuracy = evaluate_separate(models, test_contexts, args, device, acc_n=final_acc_n)
    runtime = time.time() - start
    return final_accuracy, history, runtime
