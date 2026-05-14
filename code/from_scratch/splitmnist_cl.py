#!/usr/bin/env python3
"""CLI entry point for the clean-room Split MNIST continual-learning runner.

The implementation is split by responsibility:

- ``core.py``: data, model, replay buffer, evaluation, and output helpers.
- ``methods/``: one file per method family.
- this file: argument parsing, method dispatch, and result writing.

This keeps the project defensible for the reproduction assignment: the code is
ours, readable, and no longer a single monolithic experiment script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch

from core import (
    RunMetrics,
    append_learning_curve,
    append_summary,
    build_split_mnist,
    display_method,
    normalize_method,
    safe_name,
    set_seed,
    write_json,
)
from methods.generative import train_generative
from methods.joint import train_joint
from methods.separate import train_separate
from methods.sequential import train_sequential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean-room Split MNIST continual-learning runner")
    parser.add_argument("--method", required=True, help="none, joint, ewc, lwf, agem, separate, gen-classifier, lsr-lite variants")
    parser.add_argument("--scenario", default="class", choices=["class", "domain", "task"])
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--iters", type=int, default=2000, help="iterations per context")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--acc-n", type=int, default=1024, help="examples per context for intermediate eval; final uses full test unless disabled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="E:/Codex/continual-learning-setup/data")
    parser.add_argument("--results-dir", default="E:/Codex/continual-learning-setup/splitmnist-continual-learning-report/results_from_scratch")
    parser.add_argument("--hidden", type=int, nargs="*", default=[400, 400])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--memory-per-class", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--final-acc-n", type=int, default=0, help="0 means full test set")
    parser.add_argument("--ewc-lambda", type=float, default=5000.0)
    parser.add_argument("--fisher-samples", type=int, default=1024)
    parser.add_argument("--lwf-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lsr-kd-lambda", type=float, default=1.0)
    parser.add_argument("--lsr-feature-lambda", type=float, default=0.5)
    parser.add_argument("--lsr-replay-ce-lambda", type=float, default=1.0)
    parser.add_argument("--lsr-fourier-lambda", type=float, default=0.05)
    parser.add_argument("--asw-epsilon", type=float, default=1e-8)
    parser.add_argument("--asw-min", type=float, default=0.5)
    parser.add_argument("--asw-max", type=float, default=2.0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    args.method = normalize_method(args.method)
    if args.contexts != 5:
        raise ValueError("This script currently implements the standard 5-context Split MNIST setup")
    if args.method == "separate" and args.scenario != "task":
        raise ValueError("Separate Networks is implemented only for Task-CL")
    return args


def run_method(args: argparse.Namespace, train_contexts, test_contexts, output_dim: int, device: torch.device):
    """Dispatch to the method-specific implementation."""

    extras: Dict[str, float] = {}
    if args.method == "joint":
        final_accuracy, history, runtime = train_joint(args, train_contexts, test_contexts, output_dim, device)
    elif args.method == "separate":
        final_accuracy, history, runtime = train_separate(args, train_contexts, test_contexts, device)
    elif args.method == "gen-classifier":
        final_accuracy, history, runtime = train_generative(args, train_contexts, test_contexts, output_dim, device)
    else:
        final_accuracy, history, runtime, extras = train_sequential(args, train_contexts, test_contexts, output_dim, device)
    return final_accuracy, history, runtime, extras


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{args.scenario}_{safe_name(display_method(args.method))}_seed{args.seed}"

    print(f"[setup] method={display_method(args.method)} scenario={args.scenario} device={device}", flush=True)
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_contexts, test_contexts, output_dim = build_split_mnist(args)
    final_accuracy, history, runtime, extras = run_method(args, train_contexts, test_contexts, output_dim, device)

    method_name = display_method(args.method)
    metrics = RunMetrics(
        method=method_name,
        scenario=args.scenario,
        seed=args.seed,
        contexts=args.contexts,
        iters=args.iters,
        batch=args.batch,
        acc_n=args.acc_n,
        memory_per_class=args.memory_per_class,
        final_accuracy=final_accuracy,
        runtime_seconds=runtime,
        output_dim=output_dim,
        command=" ".join(sys.argv),
    )

    per_run_curve = results_dir / f"learning_curve_{run_id}.csv"
    append_learning_curve(per_run_curve, history)
    append_learning_curve(results_dir / "learning_curve.csv", history)
    append_summary(results_dir / "summary.csv", metrics)

    payload = asdict(metrics)
    payload.update(extras)
    payload["device"] = str(device)
    payload["torch_version"] = torch.__version__
    payload["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
    write_json(results_dir / f"metrics_{run_id}.json", payload)

    print(f"[done] method={method_name} scenario={args.scenario} final_accuracy={final_accuracy:.6f} runtime_sec={runtime:.1f}", flush=True)
    if extras:
        print(f"[extra] {extras}", flush=True)
    print(f"[saved] {results_dir}", flush=True)


if __name__ == "__main__":
    main()
