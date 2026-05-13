#!/usr/bin/env python3
"""Independent Split MNIST continual-learning experiments.

This file is a clean-room implementation for the project report. It does not
import code from GMvandeVen/continual-learning. The goal is to reproduce the
Split MNIST protocols with our own PyTorch code and comparable methods.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DIGIT_CONTEXTS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def normalize_method(name: str) -> str:
    aliases = {
        "none": "none",
        "joint": "joint",
        "ewc": "ewc",
        "lwf": "lwf",
        "a-gem": "agem",
        "agem": "agem",
        "separate": "separate",
        "separate-networks": "separate",
        "generative-classifier": "gen-classifier",
        "gen-classifier": "gen-classifier",
        "lsr": "lsr-lite",
        "lsr-lite": "lsr-lite",
        "lsr-lite-fourier": "lsr-lite-fourier",
        "lsr-lite-asw": "lsr-lite-asw",
        "lsr-lite-fourier-asw": "lsr-lite-fourier-asw",
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError(f"Unknown method: {name}")
    return aliases[key]


def display_method(method: str) -> str:
    names = {
        "none": "None",
        "joint": "Joint",
        "ewc": "EWC",
        "lwf": "LwF",
        "agem": "A-GEM",
        "separate": "Separate Networks",
        "gen-classifier": "Generative Classifier",
        "lsr-lite": "LSR-lite",
        "lsr-lite-fourier": "LSR-lite + Fourier",
        "lsr-lite-asw": "LSR-lite + ASW",
        "lsr-lite-fourier-asw": "LSR-lite + Fourier + ASW",
    }
    return names[method]


def safe_name(text: str) -> str:
    return (
        text.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


@dataclass
class RunMetrics:
    method: str
    scenario: str
    seed: int
    contexts: int
    iters: int
    batch: int
    acc_n: int
    memory_per_class: int
    final_accuracy: float
    runtime_seconds: float
    output_dim: int
    command: str


class SplitMNISTContext(Dataset):
    """Subset of MNIST for one digit pair.

    Returns (image, training_label, original_digit).
    For Domain-CL the training label is local to the pair: 0 or 1.
    For Class-CL and Task-CL the training label is the original digit.
    """

    def __init__(self, base: Dataset, indices: Sequence[int], digits: Tuple[int, int], scenario: str):
        self.base = base
        self.indices = list(indices)
        self.digits = tuple(digits)
        self.scenario = scenario
        self.original_classes = list(digits)
        self._local = {digits[0]: 0, digits[1]: 1}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.base[self.indices[idx]]
        original = int(y)
        if self.scenario == "domain":
            mapped = self._local[original]
        else:
            mapped = original
        return x, torch.tensor(mapped, dtype=torch.long), torch.tensor(original, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, output_dim: int, hidden: Sequence[int] = (400, 400), dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 28 * 28
        for hidden_dim in hidden:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, output_dim)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ReplayBuffer:
    """Class-balanced CPU replay buffer.

    Keys are original digit classes, so Domain-CL can keep examples from all
    digit domains even though the training labels are only 0 or 1.
    """

    def __init__(self, samples_per_class: int):
        self.samples_per_class = samples_per_class
        self.data: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return sum(entry["x"].size(0) for entry in self.data.values())

    def add_from_dataset(
        self,
        model: Optional[MLP],
        dataset: SplitMNISTContext,
        device: torch.device,
        batch_size: int,
        store_signals: bool = False,
    ) -> None:
        per_key: Dict[int, Dict[str, List[torch.Tensor]]] = {}
        for key in dataset.original_classes:
            per_key[int(key)] = {"x": [], "y": [], "logits": [], "features": []}

        if model is not None:
            model.eval()

        with torch.no_grad():
            for x, y, original in make_loader(dataset, batch_size, shuffle=False, drop_last=False):
                x_dev = x.to(device)
                logits = None
                features = None
                if store_signals:
                    if model is None:
                        raise ValueError("store_signals=True requires a model")
                    features = model.features(x_dev).cpu()
                    logits = model(x_dev).cpu()

                for i in range(x.size(0)):
                    key = int(original[i])
                    bucket = per_key[key]
                    if len(bucket["x"]) >= self.samples_per_class:
                        continue
                    bucket["x"].append(x[i].cpu())
                    bucket["y"].append(y[i].cpu())
                    if store_signals:
                        assert logits is not None and features is not None
                        bucket["logits"].append(logits[i].cpu())
                        bucket["features"].append(features[i].cpu())

                if all(len(bucket["x"]) >= self.samples_per_class for bucket in per_key.values()):
                    break

        for key, bucket in per_key.items():
            if not bucket["x"]:
                continue
            entry = {
                "x": torch.stack(bucket["x"]),
                "y": torch.stack(bucket["y"]).long(),
            }
            if store_signals:
                entry["logits"] = torch.stack(bucket["logits"])
                entry["features"] = torch.stack(bucket["features"])
            self.data[key] = entry

        if model is not None:
            model.train()

    def sample(self, batch_size: int, device: torch.device, with_signals: bool = False):
        if len(self) == 0:
            return None
        xs = torch.cat([entry["x"] for entry in self.data.values()], dim=0)
        ys = torch.cat([entry["y"] for entry in self.data.values()], dim=0)
        idx = torch.randint(0, xs.size(0), (min(batch_size, xs.size(0)),))
        result = [xs[idx].to(device), ys[idx].to(device)]
        if with_signals:
            logits = torch.cat([entry["logits"] for entry in self.data.values()], dim=0)
            features = torch.cat([entry["features"] for entry in self.data.values()], dim=0)
            result.extend([logits[idx].to(device), features[idx].to(device)])
        return tuple(result)


class GaussianGenerativeClassifier:
    """Diagonal Gaussian class-conditional classifier.

    This is a simple independent implementation of a generative classifier. It
    stores class statistics, not raw old training data.
    """

    def __init__(self, output_dim: int, scenario: str, eps: float = 1e-4):
        self.output_dim = output_dim
        self.scenario = scenario
        self.eps = eps
        self.count: Dict[int, int] = {}
        self.sum: Dict[int, torch.Tensor] = {}
        self.sumsq: Dict[int, torch.Tensor] = {}

    def update(self, dataset: SplitMNISTContext, batch_size: int) -> None:
        for x, y, original in make_loader(dataset, batch_size, shuffle=False, drop_last=False):
            flat = x.flatten(1).cpu().double()
            labels = y if self.scenario == "domain" else original
            for i in range(flat.size(0)):
                label = int(labels[i])
                vector = flat[i]
                if label not in self.count:
                    self.count[label] = 0
                    self.sum[label] = torch.zeros_like(vector)
                    self.sumsq[label] = torch.zeros_like(vector)
                self.count[label] += 1
                self.sum[label] += vector
                self.sumsq[label] += vector * vector

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1).cpu().double()
        total_count = max(1, sum(self.count.values()))
        scores = []
        for label in range(self.output_dim):
            if label not in self.count:
                scores.append(torch.full((flat.size(0),), -1e9, dtype=torch.double))
                continue
            n = self.count[label]
            mean = self.sum[label] / n
            var = (self.sumsq[label] / n - mean * mean).clamp_min(self.eps)
            log_prior = math.log(n / total_count)
            log_prob = -0.5 * (((flat - mean) ** 2 / var) + torch.log(var)).sum(dim=1)
            scores.append(log_prob + log_prior)
        return torch.stack(scores, dim=1).float().to(x.device)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def next_batch(loader: DataLoader, iterator: Optional[Iterable]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Iterable]:
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def build_split_mnist(args: argparse.Namespace) -> Tuple[List[SplitMNISTContext], List[SplitMNISTContext], int]:
    transform = transforms.ToTensor()
    root = Path(args.data_dir)
    train_base = datasets.MNIST(root=str(root), train=True, download=not args.no_download, transform=transform)
    test_base = datasets.MNIST(root=str(root), train=False, download=not args.no_download, transform=transform)
    train_targets = torch.as_tensor(train_base.targets)
    test_targets = torch.as_tensor(test_base.targets)

    train_contexts = []
    test_contexts = []
    for digits in DIGIT_CONTEXTS:
        train_mask = (train_targets == digits[0]) | (train_targets == digits[1])
        test_mask = (test_targets == digits[0]) | (test_targets == digits[1])
        train_indices = train_mask.nonzero(as_tuple=False).flatten().tolist()
        test_indices = test_mask.nonzero(as_tuple=False).flatten().tolist()
        train_contexts.append(SplitMNISTContext(train_base, train_indices, digits, args.scenario))
        test_contexts.append(SplitMNISTContext(test_base, test_indices, digits, args.scenario))

    output_dim = 2 if args.scenario == "domain" else 10
    return train_contexts, test_contexts, output_dim


def move_batch(batch, device: torch.device):
    x, y, original = batch
    return x.to(device), y.to(device), original.to(device)


def mask_task_logits(logits: torch.Tensor, context_index: int) -> torch.Tensor:
    allowed = DIGIT_CONTEXTS[context_index]
    masked = torch.full_like(logits, -1e9)
    masked[:, allowed[0]] = logits[:, allowed[0]]
    masked[:, allowed[1]] = logits[:, allowed[1]]
    return masked


def supervised_context_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    context_index: int,
) -> torch.Tensor:
    """Cross-entropy for the active scenario.

    In Task-CL, the task identity is part of the protocol. Training therefore
    uses only the two active classes, matching the allowed-classes evaluation.
    """

    if args.scenario != "task":
        return F.cross_entropy(logits, labels)
    allowed = DIGIT_CONTEXTS[context_index]
    active_logits = logits[:, [allowed[0], allowed[1]]]
    active_labels = labels - allowed[0]
    return F.cross_entropy(active_logits, active_labels)


def evaluate_neural(
    model: MLP,
    test_contexts: Sequence[SplitMNISTContext],
    args: argparse.Namespace,
    device: torch.device,
    acc_n: Optional[int],
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for context_index, dataset in enumerate(test_contexts):
            eval_dataset: Dataset = dataset
            if acc_n is not None and acc_n > 0 and acc_n < len(dataset):
                eval_dataset = Subset(dataset, list(range(acc_n)))
            for batch in make_loader(eval_dataset, args.batch, shuffle=False, drop_last=False):
                x, y, _original = move_batch(batch, device)
                logits = model(x)
                if args.scenario == "task":
                    logits = mask_task_logits(logits, context_index)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
    model.train()
    return correct / max(1, total)


def evaluate_separate(
    models: Sequence[MLP],
    test_contexts: Sequence[SplitMNISTContext],
    args: argparse.Namespace,
    device: torch.device,
    acc_n: Optional[int],
) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for context_index, dataset in enumerate(test_contexts):
            model = models[context_index]
            model.eval()
            eval_dataset: Dataset = dataset
            if acc_n is not None and acc_n > 0 and acc_n < len(dataset):
                eval_dataset = Subset(dataset, list(range(acc_n)))
            for batch in make_loader(eval_dataset, args.batch, shuffle=False, drop_last=False):
                x, _y, original = move_batch(batch, device)
                target = original - DIGIT_CONTEXTS[context_index][0]
                pred = model(x).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.numel()
            model.train()
    return correct / max(1, total)


def evaluate_generative(
    classifier: GaussianGenerativeClassifier,
    test_contexts: Sequence[SplitMNISTContext],
    args: argparse.Namespace,
    device: torch.device,
    acc_n: Optional[int],
) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for context_index, dataset in enumerate(test_contexts):
            eval_dataset: Dataset = dataset
            if acc_n is not None and acc_n > 0 and acc_n < len(dataset):
                eval_dataset = Subset(dataset, list(range(acc_n)))
            for batch in make_loader(eval_dataset, args.batch, shuffle=False, drop_last=False):
                x, y, _original = move_batch(batch, device)
                logits = classifier.logits(x)
                if args.scenario == "task":
                    logits = mask_task_logits(logits, context_index)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
    return correct / max(1, total)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    target_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean") * (temperature ** 2)


def fourier_loss(current_features: torch.Tensor, stored_features: torch.Tensor) -> torch.Tensor:
    current_spec = torch.log1p(torch.abs(torch.fft.rfft(current_features.float(), dim=1)))
    stored_spec = torch.log1p(torch.abs(torch.fft.rfft(stored_features.float(), dim=1)))
    return F.mse_loss(current_spec, stored_spec)


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


def ewc_penalty(model: MLP, ewc_tasks: List[Tuple[List[torch.Tensor], List[torch.Tensor]]]) -> torch.Tensor:
    if not ewc_tasks:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    params = [p for p in model.parameters() if p.requires_grad]
    for means, fishers in ewc_tasks:
        for param, mean, fisher in zip(params, means, fishers):
            penalty = penalty + (fisher * (param - mean).pow(2)).sum()
    return penalty


def estimate_fisher(
    model: MLP,
    dataset: SplitMNISTContext,
    args: argparse.Namespace,
    device: torch.device,
    context_index: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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


def append_learning_curve(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "scenario", "iteration", "context", "accuracy"])
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_summary(path: Path, metrics: RunMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(metrics).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(metrics))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def record_eval(
    history: List[Dict[str, object]],
    args: argparse.Namespace,
    method_name: str,
    iteration: int,
    context: int,
    accuracy: float,
) -> None:
    row = {
        "method": method_name,
        "scenario": args.scenario,
        "iteration": iteration,
        "context": context,
        "accuracy": accuracy,
    }
    history.append(row)
    print(f"[eval] method={method_name} scenario={args.scenario} iter={iteration} context={context} acc={accuracy:.4f}", flush=True)


def train_joint(
    args: argparse.Namespace,
    train_contexts: Sequence[SplitMNISTContext],
    test_contexts: Sequence[SplitMNISTContext],
    output_dim: int,
    device: torch.device,
) -> Tuple[float, List[Dict[str, object]], float]:
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


def train_separate(
    args: argparse.Namespace,
    train_contexts: Sequence[SplitMNISTContext],
    test_contexts: Sequence[SplitMNISTContext],
    device: torch.device,
) -> Tuple[float, List[Dict[str, object]], float]:
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


def train_generative(
    args: argparse.Namespace,
    train_contexts: Sequence[SplitMNISTContext],
    test_contexts: Sequence[SplitMNISTContext],
    output_dim: int,
    device: torch.device,
) -> Tuple[float, List[Dict[str, object]], float]:
    start = time.time()
    method_name = display_method(args.method)
    classifier = GaussianGenerativeClassifier(output_dim=output_dim, scenario=args.scenario)
    history: List[Dict[str, object]] = []

    for context_index, train_dataset in enumerate(train_contexts):
        classifier.update(train_dataset, args.batch)
        acc = evaluate_generative(classifier, test_contexts, args, device, acc_n=args.acc_n)
        record_eval(history, args, method_name, (context_index + 1) * args.iters, context_index + 1, acc)

    final_acc_n = None if args.final_acc_n == 0 else args.final_acc_n
    final_accuracy = evaluate_generative(classifier, test_contexts, args, device, acc_n=final_acc_n)
    runtime = time.time() - start
    return final_accuracy, history, runtime


def train_sequential(
    args: argparse.Namespace,
    train_contexts: Sequence[SplitMNISTContext],
    test_contexts: Sequence[SplitMNISTContext],
    output_dim: int,
    device: torch.device,
) -> Tuple[float, List[Dict[str, object]], float, Dict[str, float]]:
    start = time.time()
    method_name = display_method(args.method)
    model = MLP(output_dim=output_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]
    ewc_tasks: List[Tuple[List[torch.Tensor], List[torch.Tensor]]] = []
    teacher: Optional[MLP] = None
    replay = ReplayBuffer(args.memory_per_class)
    lsr_factors: List[float] = []
    history: List[Dict[str, object]] = []
    global_step = 0

    is_lsr = args.method.startswith("lsr-lite")
    use_fourier = args.method in ("lsr-lite-fourier", "lsr-lite-fourier-asw")
    use_asw = args.method in ("lsr-lite-asw", "lsr-lite-fourier-asw")

    for context_index, train_dataset in enumerate(train_contexts):
        loader = make_loader(train_dataset, args.batch, shuffle=True, drop_last=True)
        iterator = None
        for _step in range(1, args.iters + 1):
            global_step += 1
            batch, iterator = next_batch(loader, iterator)
            x, y, _original = move_batch(batch, device)

            if args.method == "agem" and len(replay) > 0:
                optimizer.zero_grad(set_to_none=True)
                loss_current = supervised_context_loss(model(x), y, args, context_index)
                loss_current.backward()
                grad_current = grad_vector(params)

                replay_batch = replay.sample(args.batch, device, with_signals=False)
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

                if is_lsr and len(replay) > 0:
                    replay_batch = replay.sample(args.batch, device, with_signals=True)
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
                        lsr_factors.append(float(factor))

                    loss = loss + args.lsr_replay_ce_lambda * replay_ce
                    loss = loss + args.lsr_kd_lambda * factor * kd
                    loss = loss + args.lsr_feature_lambda * factor * feature_anchor

                    if use_fourier:
                        loss = loss + args.lsr_fourier_lambda * fourier_loss(rep_features, old_features)

                loss.backward()
                optimizer.step()

            if args.eval_every and global_step % args.eval_every == 0:
                acc = evaluate_neural(model, test_contexts, args, device, acc_n=args.acc_n)
                record_eval(history, args, method_name, global_step, context_index + 1, acc)

        if args.method == "ewc":
            ewc_tasks.append(estimate_fisher(model, train_dataset, args, device, context_index))
        elif args.method == "lwf":
            teacher = copy.deepcopy(model).to(device).eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
        elif args.method == "agem":
            replay.add_from_dataset(model=None, dataset=train_dataset, device=device, batch_size=args.batch, store_signals=False)
        elif is_lsr:
            replay.add_from_dataset(model=model, dataset=train_dataset, device=device, batch_size=args.batch, store_signals=True)

    final_acc_n = None if args.final_acc_n == 0 else args.final_acc_n
    final_accuracy = evaluate_neural(model, test_contexts, args, device, acc_n=final_acc_n)
    runtime = time.time() - start
    extras: Dict[str, float] = {}
    if lsr_factors:
        extras = {
            "asw_mean": float(np.mean(lsr_factors)),
            "asw_min": float(np.min(lsr_factors)),
            "asw_max": float(np.max(lsr_factors)),
        }
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

    extras: Dict[str, float] = {}
    if args.method == "joint":
        final_accuracy, history, runtime = train_joint(args, train_contexts, test_contexts, output_dim, device)
    elif args.method == "separate":
        final_accuracy, history, runtime = train_separate(args, train_contexts, test_contexts, device)
    elif args.method == "gen-classifier":
        final_accuracy, history, runtime = train_generative(args, train_contexts, test_contexts, output_dim, device)
    else:
        final_accuracy, history, runtime, extras = train_sequential(args, train_contexts, test_contexts, output_dim, device)

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
