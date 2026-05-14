"""Simple diagonal-Gaussian Generative Classifier."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence, Tuple

import torch

from core import SplitMNISTContext, display_method, make_loader, mask_task_logits, move_batch, record_eval


class GaussianGenerativeClassifier:
    """Class-conditional diagonal Gaussian over flattened pixels."""

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


def evaluate_generative(classifier: GaussianGenerativeClassifier, test_contexts: Sequence[SplitMNISTContext], args, device: torch.device, acc_n):
    correct = 0
    total = 0
    with torch.no_grad():
        for context_index, dataset in enumerate(test_contexts):
            eval_dataset = dataset
            if acc_n is not None and acc_n > 0 and acc_n < len(dataset):
                from torch.utils.data import Subset

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


def train_generative(args, train_contexts: Sequence[SplitMNISTContext], test_contexts: Sequence[SplitMNISTContext], output_dim: int, device: torch.device) -> Tuple[float, List[Dict[str, object]], float]:
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
