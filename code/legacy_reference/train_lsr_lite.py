#!/usr/bin/env python3
"""Prototype runner for LSR-lite on task-based continual-learning experiments.

This script intentionally keeps the experimental LSR code separate from the
repository's established methods. It reuses the repository data/model/eval
utilities, while adding a real-sample replay buffer that stores labels, teacher
logits, and penultimate features at insertion time.
"""

import argparse
import csv
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import tqdm

import utils
from data.load import get_context_set
from data.manipulate import SubDataset
from eval import evaluate
from models import define_models as define
from params.param_values import set_default_values


class LSRBuffer:
    """Class-balanced exemplar buffer with stored teacher signals."""

    def __init__(self, samples_per_class):
        self.samples_per_class = samples_per_class
        self.data = {}

    def __len__(self):
        return sum(entry["x"].size(0) for entry in self.data.values())

    def add_class(self, class_id, x, y, logits, features):
        n = min(self.samples_per_class, x.size(0))
        self.data[class_id] = {
            "x": x[:n].detach().cpu().clone(),
            "y": y[:n].detach().cpu().clone(),
            "logits": logits[:n].detach().cpu().clone(),
            "features": features[:n].detach().cpu().clone(),
        }

    def sample(self, batch_size, device):
        if len(self) == 0:
            return None
        entries = list(self.data.values())
        xs = torch.cat([entry["x"] for entry in entries], dim=0)
        ys = torch.cat([entry["y"] for entry in entries], dim=0)
        logits = torch.cat([entry["logits"] for entry in entries], dim=0)
        features = torch.cat([entry["features"] for entry in entries], dim=0)
        idx = torch.randint(0, xs.size(0), (min(batch_size, xs.size(0)),))
        return (
            xs[idx].to(device),
            ys[idx].to(device),
            logits[idx].to(device),
            features[idx].to(device),
        )


def build_args():
    parser = argparse.ArgumentParser(description="Run the LSR-lite prototype.")
    parser.add_argument("--experiment", type=str, default="splitMNIST")
    parser.add_argument("--scenario", type=str, default="class", choices=["task", "domain", "class"])
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--acc-n", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="./store/datasets", dest="d_dir")
    parser.add_argument("--results-dir", type=str, default="./results/lsr-lite-class", dest="r_dir")
    parser.add_argument("--model-dir", type=str, default="./results/models", dest="m_dir")
    parser.add_argument("--budget", type=int, default=100, help="stored exemplars per class")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--replay-ce-weight", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=2.0, help="KD temperature")
    parser.add_argument("--fourier", action="store_true", help="enable auxiliary feature-spectrum anchoring")
    parser.add_argument("--fourier-weight", type=float, default=0.1)
    parser.add_argument("--asw", action="store_true", help="enable adaptive stability weighting")
    parser.add_argument("--asw-epsilon", type=float, default=1e-8)
    parser.add_argument("--asw-min", type=float, default=0.5)
    parser.add_argument("--asw-max", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-history-file", type=str, default=None)
    parser.add_argument("--eval-history-method", type=str, default=None)
    parser.add_argument("--no-gpus", action="store_false", dest="gpu")
    parser.set_defaults(gpu=True)
    args = parser.parse_args()

    # Minimal model defaults expected by define_models.py.
    defaults = dict(
        normalize=False,
        depth=None,
        fc_lay=None,
        channels=None,
        rl=None,
        gp=False,
        conv_type="standard",
        n_blocks=2,
        conv_bn="yes",
        conv_nl="relu",
        fc_units=None,
        fc_drop=0.0,
        fc_bn="no",
        fc_nl="relu",
        optimizer="adam",
        lr=None,
        pre_convE=False,
        freeze_convE=False,
        convE_ltag="e100",
        seed_to_ltag=False,
        separate_networks=False,
        feedback=False,
        gen_classifier=False,
        fisher_kfac=False,
        singlehead=False,
        neg_samples="all",
    )
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    set_default_values(args, also_hyper_params=False)
    return args


def kd_loss(student_logits, teacher_logits, temp):
    log_probs = F.log_softmax(student_logits / temp, dim=1)
    target_probs = F.softmax(teacher_logits / temp, dim=1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean") * (temp ** 2)


def fourier_feature_loss(current_features, stored_features):
    current_spec = torch.log1p(torch.abs(torch.fft.rfft(current_features.float(), dim=1)))
    stored_spec = torch.log1p(torch.abs(torch.fft.rfft(stored_features.float(), dim=1)))
    return F.mse_loss(current_spec, stored_spec)


def collect_class_exemplars(model, dataset, class_id, n, batch_size, device):
    loader = utils.get_data_loader(dataset, batch_size=batch_size, cuda=(device.type == "cuda"), drop_last=False)
    xs, ys, logits, features = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model.classify(x, no_prototypes=True)
            feat = model.feature_extractor(x)
            xs.append(x.cpu())
            ys.append(y.cpu())
            logits.append(out.cpu())
            features.append(feat.cpu())
            if sum(chunk.size(0) for chunk in xs) >= n:
                break
    model.train()
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(logits, dim=0), torch.cat(features, dim=0)


def evaluate_average(model, test_datasets, test_size, batch_size, current_context=None):
    accs = []
    datasets = test_datasets[:current_context] if current_context is not None else test_datasets
    for i, dataset in enumerate(datasets):
        allowed_classes = None
        if model.scenario == "task" and not getattr(model, "singlehead", False):
            allowed_classes = list(
                range(model.classes_per_context * i, model.classes_per_context * (i + 1))
            )
        accs.append(
            evaluate.test_acc(
                model, dataset, batch_size=batch_size, test_size=test_size,
                verbose=False, context_id=i, allowed_classes=allowed_classes
            )
        )
    return sum(accs) / len(accs), accs


def append_eval_history(file_path, method, iteration, context, accuracy):
    if file_path is None:
        return
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "iteration", "context", "accuracy"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "method": method,
            "iteration": iteration,
            "context": context,
            "accuracy": accuracy,
        })


def main():
    args = build_args()
    if args.scenario not in ("class", "domain", "task"):
        raise ValueError("This prototype runner supports Class-CL, Domain-CL, and Task-CL.")

    Path(args.r_dir).mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    (train_datasets, test_datasets), config = get_context_set(
        name=args.experiment,
        scenario=args.scenario,
        contexts=args.contexts,
        data_dir=args.d_dir,
        normalize=False,
        exception=(args.seed == 0),
    )

    model = define.define_classifier(args=args, config=config, device=device, depth=args.depth)
    define.init_params(model, args)
    model.scenario = args.scenario
    model.classes_per_context = config["classes_per_context"]
    model.singlehead = False
    model.neg_samples = "all"
    model.optimizer = optim.Adam(
        [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr}],
        betas=(0.9, 0.999),
    )

    buffer = LSRBuffer(samples_per_class=args.budget)
    final_losses = {}
    adaptive_factors = []
    method = "LSR-lite"
    if args.fourier:
        method += "+Fourier"
    if args.asw:
        method += "+ASW"
    eval_history_method = args.eval_history_method or method

    for context, train_dataset in enumerate(train_datasets, 1):
        data_loader = None
        iters_left = 0
        progress = tqdm.tqdm(range(1, args.iters + 1))
        for batch_index in progress:
            iters_left -= 1
            if iters_left <= 0:
                data_loader = iter(utils.get_data_loader(train_dataset, args.batch, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)
            x, y = next(data_loader)
            x, y = x.to(device), y.to(device)

            model.train()
            if model.convE.frozen:
                model.convE.eval()
            if model.fcE.frozen:
                model.fcE.eval()
            model.optimizer.zero_grad()

            logits_cur = model.classify(x, no_prototypes=True)
            loss_current = F.cross_entropy(logits_cur, y)
            loss = loss_current
            metrics = {
                "loss_current": loss_current.item(),
                "loss_replay_ce": 0.0,
                "loss_distill": 0.0,
                "loss_feature": 0.0,
                "loss_fourier": 0.0,
            }

            replay = buffer.sample(args.batch, device)
            if replay is not None:
                x_r, y_r, logits_teacher, features_teacher = replay
                logits_replay = model.classify(x_r, no_prototypes=True)
                features_replay = model.feature_extractor(x_r)
                loss_replay_ce = F.cross_entropy(logits_replay, y_r)
                loss_distill = kd_loss(logits_replay, logits_teacher, args.temp)
                loss_feature = F.mse_loss(features_replay, features_teacher)
                if args.asw:
                    with torch.no_grad():
                        old_loss = F.cross_entropy(logits_teacher, y_r)
                        adaptive_factor = old_loss / (loss_replay_ce.detach() + args.asw_epsilon)
                        adaptive_factor = torch.clamp(adaptive_factor, min=args.asw_min, max=args.asw_max)
                    adaptive_factors.append(adaptive_factor.item())
                else:
                    adaptive_factor = torch.tensor(1.0, device=device)
                lambda_kd_eff = args.distill_weight * adaptive_factor
                lambda_feat_eff = args.feature_weight * adaptive_factor
                loss = loss + args.replay_ce_weight * loss_replay_ce
                loss = loss + lambda_kd_eff * loss_distill
                loss = loss + lambda_feat_eff * loss_feature
                metrics.update(
                    loss_replay_ce=loss_replay_ce.item(),
                    loss_distill=loss_distill.item(),
                    loss_feature=loss_feature.item(),
                    adaptive_factor=adaptive_factor.item(),
                    lambda_kd_eff=lambda_kd_eff.item(),
                    lambda_feat_eff=lambda_feat_eff.item(),
                )
                if args.fourier:
                    loss_fourier = fourier_feature_loss(features_replay, features_teacher)
                    loss = loss + args.fourier_weight * loss_fourier
                    metrics["loss_fourier"] = loss_fourier.item()

            loss.backward()
            model.optimizer.step()
            final_losses = metrics
            global_iteration = (context - 1) * args.iters + batch_index
            if args.eval_every is not None and global_iteration % args.eval_every == 0:
                history_context = context if args.scenario == "task" else None
                history_acc, _ = evaluate_average(
                    model, test_datasets, args.acc_n, args.batch, current_context=history_context
                )
                append_eval_history(
                    args.eval_history_file, eval_history_method, global_iteration, context, history_acc
                )
            progress.set_description(
                "<LSR{}> | Context: {}/{} | loss: {:.3f} |".format(
                    "{}{}".format("+FFT" if args.fourier else "", "+ASW" if args.asw else ""),
                    context,
                    args.contexts,
                    loss.item(),
                )
            )
        progress.close()

        if context < len(train_datasets):
            if args.scenario == "domain":
                new_classes = list(range(config["classes_per_context"]))
            else:
                new_classes = list(
                    range(config["classes_per_context"] * (context - 1), config["classes_per_context"] * context)
                )
            for class_id in new_classes:
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                x_e, y_e, logits_e, features_e = collect_class_exemplars(
                    model, class_dataset, class_id, args.budget, args.batch, device
                )
                buffer_key = (context, class_id) if args.scenario == "domain" else class_id
                buffer.add_class(buffer_key, x_e, y_e, logits_e, features_e)

    # Match main.py's final metric: the repository writes final accuracy on the
    # full test split. The --acc-n option is used there for interim callbacks.
    average_acc, per_context_acc = evaluate_average(model, test_datasets, None, args.batch)

    result_name = (
        "acc-{}{}-{}--{}--i{}-b{}-bud{}-kd{}-feat{}{}.txt".format(
            args.experiment,
            args.contexts,
            args.scenario,
            method.replace("+", "-").replace(" ", ""),
            args.iters,
            args.batch,
            args.budget,
            args.distill_weight,
            args.feature_weight,
            "{}{}".format(
                "-fft{}".format(args.fourier_weight) if args.fourier else "",
                "-asw{}-{}-eps{}".format(args.asw_min, args.asw_max, args.asw_epsilon) if args.asw else "",
            ),
        )
    )
    result_path = Path(args.r_dir) / result_name
    result_path.write_text(str(average_acc), encoding="utf-8")

    metrics_path = Path(args.r_dir) / result_name.replace("acc-", "metrics-").replace(".txt", ".csv")
    asw_mean = float(np.mean(adaptive_factors)) if adaptive_factors else 1.0
    asw_min = float(np.min(adaptive_factors)) if adaptive_factors else 1.0
    asw_max = float(np.max(adaptive_factors)) if adaptive_factors else 1.0
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "average_accuracy", "per_context_accuracy", "buffer_size",
                "asw_mean", "asw_min", "asw_max", *final_losses.keys()
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": method,
                "average_accuracy": average_acc,
                "per_context_accuracy": ";".join(str(x) for x in per_context_acc),
                "buffer_size": len(buffer),
                "asw_mean": asw_mean,
                "asw_min": asw_min,
                "asw_max": asw_max,
                **final_losses,
            }
        )

    print("\n***************************** EVALUATION *****************************")
    for index, acc in enumerate(per_context_acc, 1):
        print(" - Context {}: {:.4f}".format(index, acc))
    print("=> average accuracy over all {} contexts: {:.4f}".format(args.contexts, average_acc))
    if args.asw:
        print("=> adaptive_factor mean/min/max: {:.4f}/{:.4f}/{:.4f}".format(asw_mean, asw_min, asw_max))
    print("result_file={}".format(result_path))
    print("metrics_file={}".format(metrics_path))


if __name__ == "__main__":
    main()
