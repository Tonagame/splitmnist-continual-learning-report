"""Shared building blocks for the clean-room Split MNIST experiments.

This module intentionally contains only reusable infrastructure: dataset
construction, the MLP model, replay storage, evaluation, and CSV/JSON output.
The continual-learning algorithms themselves live in ``methods/``.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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

    Returns ``(image, training_label, original_digit)``. Domain-CL maps the pair
    to a local binary label; Class-CL and Task-CL keep the original digit label.
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
        mapped = self._local[original] if self.scenario == "domain" else original
        return x, torch.tensor(mapped, dtype=torch.long), torch.tensor(original, dtype=torch.long)


class MLP(nn.Module):
    """Small fully connected network used by all neural methods."""

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
    """Class-balanced CPU replay buffer built only from training data."""

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


def next_batch(loader: DataLoader, iterator: Optional[Iterable]):
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def build_split_mnist(args):
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


def supervised_context_loss(logits: torch.Tensor, labels: torch.Tensor, args, context_index: int) -> torch.Tensor:
    """Cross-entropy for the active scenario.

    Task-CL uses task identity, so the supervised loss is restricted to the two
    allowed classes for the current context. Class-CL and Domain-CL use the full
    active output space.
    """

    if args.scenario != "task":
        return F.cross_entropy(logits, labels)
    allowed = DIGIT_CONTEXTS[context_index]
    active_logits = logits[:, [allowed[0], allowed[1]]]
    active_labels = labels - allowed[0]
    return F.cross_entropy(active_logits, active_labels)


def evaluate_neural(model: MLP, test_contexts: Sequence[SplitMNISTContext], args, device: torch.device, acc_n: Optional[int]) -> float:
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


def evaluate_separate(models: Sequence[MLP], test_contexts: Sequence[SplitMNISTContext], args, device: torch.device, acc_n: Optional[int]) -> float:
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


def record_eval(history: List[Dict[str, object]], args, method_name: str, iteration: int, context: int, accuracy: float) -> None:
    row = {
        "method": method_name,
        "scenario": args.scenario,
        "iteration": iteration,
        "context": context,
        "accuracy": accuracy,
    }
    history.append(row)
    print(f"[eval] method={method_name} scenario={args.scenario} iter={iteration} context={context} acc={accuracy:.4f}", flush=True)
