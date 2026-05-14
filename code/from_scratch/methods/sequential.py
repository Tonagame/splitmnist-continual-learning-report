"""Sequential continual-learning methods: None, EWC, LwF, A-GEM, and LSR-lite."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import torch
from torch import optim

from core import MLP, ReplayBuffer, SplitMNISTContext, display_method, evaluate_neural, make_loader, move_batch, next_batch, record_eval, supervised_context_loss
from methods.agem import agem_step
from methods.ewc import estimate_fisher, ewc_penalty
from methods.lsr_lite import add_lsr_replay_loss, asw_summary, lsr_options
from methods.lwf import frozen_teacher, kd_loss


def train_sequential(args, train_contexts: Sequence[SplitMNISTContext], test_contexts: Sequence[SplitMNISTContext], output_dim: int, device: torch.device) -> Tuple[float, List[Dict[str, object]], float, Dict[str, float]]:
    """Shared loop for methods that see contexts one after another."""

    start = time.time()
    method_name = display_method(args.method)
    model = MLP(output_dim=output_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]
    ewc_tasks = []
    teacher = None
    replay = ReplayBuffer(args.memory_per_class)
    lsr_factors: List[float] = []
    history: List[Dict[str, object]] = []
    global_step = 0

    is_lsr, use_fourier, use_asw = lsr_options(args.method)

    for context_index, train_dataset in enumerate(train_contexts):
        loader = make_loader(train_dataset, args.batch, shuffle=True, drop_last=True)
        iterator = None
        for _step in range(1, args.iters + 1):
            global_step += 1
            batch, iterator = next_batch(loader, iterator)
            x, y, _original = move_batch(batch, device)

            if args.method == "agem" and len(replay) > 0:
                agem_step(model, optimizer, params, replay, x, y, args, context_index)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = supervised_context_loss(logits, y, args, context_index)

                if args.method == "ewc" and ewc_tasks:
                    loss = loss + 0.5 * args.ewc_lambda * ewc_penalty(model, ewc_tasks)

                if args.method == "lwf" and teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(x)
                    loss = loss + args.lwf_lambda * kd_loss(logits, teacher_logits, args.temperature)

                if is_lsr:
                    loss = add_lsr_replay_loss(model, loss, replay, args, use_fourier, use_asw, lsr_factors)

                loss.backward()
                optimizer.step()

            if args.eval_every and global_step % args.eval_every == 0:
                acc = evaluate_neural(model, test_contexts, args, device, acc_n=args.acc_n)
                record_eval(history, args, method_name, global_step, context_index + 1, acc)

        if args.method == "ewc":
            ewc_tasks.append(estimate_fisher(model, train_dataset, args, device, context_index))
        elif args.method == "lwf":
            teacher = frozen_teacher(model, device)
        elif args.method == "agem":
            replay.add_from_dataset(model=None, dataset=train_dataset, device=device, batch_size=args.batch, store_signals=False)
        elif is_lsr:
            replay.add_from_dataset(model=model, dataset=train_dataset, device=device, batch_size=args.batch, store_signals=True)

    final_acc_n = None if args.final_acc_n == 0 else args.final_acc_n
    final_accuracy = evaluate_neural(model, test_contexts, args, device, acc_n=final_acc_n)
    runtime = time.time() - start
    return final_accuracy, history, runtime, asw_summary(lsr_factors)
