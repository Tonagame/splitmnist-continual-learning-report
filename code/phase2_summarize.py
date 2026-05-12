#!/usr/bin/env python3
import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = [
    "None", "EWC", "LwF", "A-GEM", "LSR-lite", "LSR-lite + Fourier",
    "LSR-lite + ASW", "LSR-lite + Fourier + ASW", "Joint",
]

PYTHON = r"E:\conda-envs\continual\python.exe"
COMMON_MAIN = (
    "--experiment=splitMNIST --scenario=domain --contexts=5 --iters=2000 "
    "--batch=128 --acc-n=1024 --acc-log=100 --time --no-save --budget=100"
)
COMMON_LSR = (
    "--experiment=splitMNIST --scenario=domain --contexts=5 --iters=2000 "
    "--batch=128 --acc-n=1024 --budget=100 --eval-every=100"
)
COMMANDS = {
    "None": f'& "{PYTHON}" main.py {COMMON_MAIN} --eval-history-method=None',
    "EWC": f'& "{PYTHON}" main.py {COMMON_MAIN} --ewc --eval-history-method=EWC',
    "LwF": f'& "{PYTHON}" main.py {COMMON_MAIN} --lwf --eval-history-method=LwF',
    "A-GEM": f'& "{PYTHON}" main.py {COMMON_MAIN} --agem --eval-history-method=A-GEM',
    "LSR-lite": f'& "{PYTHON}" train_lsr_lite.py {COMMON_LSR} --eval-history-method=LSR-lite',
    "LSR-lite + Fourier": f'& "{PYTHON}" train_lsr_lite.py {COMMON_LSR} --fourier --eval-history-method="LSR-lite + Fourier"',
    "LSR-lite + ASW": f'& "{PYTHON}" train_lsr_lite.py {COMMON_LSR} --asw --distill-weight=1.0 --feature-weight=0.5 --temp=2 --eval-history-method="LSR-lite + ASW"',
    "LSR-lite + Fourier + ASW": f'& "{PYTHON}" train_lsr_lite.py {COMMON_LSR} --fourier --asw --distill-weight=1.0 --feature-weight=0.5 --fourier-weight=0.05 --temp=2 --eval-history-method="LSR-lite + Fourier + ASW"',
    "Joint": f'& "{PYTHON}" main.py --experiment=splitMNIST --scenario=domain --contexts=5 --iters=10000 --batch=128 --acc-n=1024 --acc-log=100 --time --no-save --budget=100 --joint --eval-history-method=Joint',
}
LOGS = {
    "None": "splitMNIST_domain_2000_none.log",
    "EWC": "splitMNIST_domain_2000_ewc.log",
    "LwF": "splitMNIST_domain_2000_lwf.log",
    "A-GEM": "splitMNIST_domain_2000_agem.log",
    "LSR-lite": "splitMNIST_domain_2000_lsr_lite.log",
    "LSR-lite + Fourier": "splitMNIST_domain_2000_lsr_lite_fourier.log",
    "LSR-lite + ASW": "splitMNIST_domain_2000_lsr_lite_asw.log",
    "LSR-lite + Fourier + ASW": "splitMNIST_domain_2000_lsr_lite_fourier_asw.log",
    "Joint": "splitMNIST_domain_2000_joint.log",
}


def latest(paths):
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def read_float(path):
    return float(path.read_text(encoding="utf-8").strip())


def find_result_files(results):
    return {
        "None": latest(results.glob("acc-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam.txt")),
        "EWC": latest(results.glob("acc-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam--PReg*-offline.txt")),
        "LwF": latest(results.glob("acc-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam--current-KD2.0.txt")),
        "A-GEM": latest(results.glob("acc-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam--buffer-A-GEM*.txt")),
        "LSR-lite": latest(results.glob("acc-splitMNIST5-domain--LSR-lite--i2000-b128-bud100-kd1.0-feat1.0.txt")),
        "LSR-lite + Fourier": latest(results.glob("acc-splitMNIST5-domain--LSR-lite-Fourier--i2000-b128-bud100-kd1.0-feat1.0-fft0.1.txt")),
        "LSR-lite + ASW": latest(results.glob("acc-splitMNIST5-domain--LSR-lite-ASW--i2000-b128-bud100-kd1.0-feat0.5-asw0.5-2.0-eps1e-08.txt")),
        "LSR-lite + Fourier + ASW": latest(results.glob("acc-splitMNIST5-domain--LSR-lite-Fourier-ASW--i2000-b128-bud100-kd1.0-feat0.5-fft0.05-asw0.5-2.0-eps1e-08.txt")),
        "Joint": latest(results.glob("acc-splitMNIST5-Joint-domain--F-784x400x400_c2--i10000-lr0.001-b128-adam.txt")),
    }


def find_time_files(results):
    return {
        "None": latest(results.glob("time-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam.txt")),
        "EWC": latest(results.glob("time-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam--PReg*-offline.txt")),
        "LwF": latest(results.glob("time-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam--current-KD2.0.txt")),
        "A-GEM": latest(results.glob("time-splitMNIST5-domain--F-784x400x400_c2--i2000-lr0.001-b128-adam--buffer-A-GEM*.txt")),
        "Joint": latest(results.glob("time-splitMNIST5-Joint-domain--F-784x400x400_c2--i10000-lr0.001-b128-adam.txt")),
    }


def runtime_from_log(results, method):
    log = results / "logs" / LOGS[method]
    if not log.exists():
        return ""
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return ""
    first = lines[0].lstrip("\ufeff")
    prefix = f"===== START {method} "
    if not first.startswith(prefix):
        return ""
    raw = first[len(prefix):].removesuffix(" =====")
    try:
        start = datetime.strptime(raw, "%m/%d/%Y %H:%M:%S")
    except ValueError:
        return ""
    return f"{(datetime.fromtimestamp(log.stat().st_mtime) - start).total_seconds():.3f}"


def load_status(results):
    path = results / "run_status.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def load_asw_stats(results, method):
    pattern = {
        "LSR-lite + ASW": "metrics-splitMNIST5-domain--LSR-lite-ASW--i2000-b128-bud100-kd1.0-feat0.5-asw0.5-2.0-eps1e-08.csv",
        "LSR-lite + Fourier + ASW": "metrics-splitMNIST5-domain--LSR-lite-Fourier-ASW--i2000-b128-bud100-kd1.0-feat0.5-fft0.05-asw0.5-2.0-eps1e-08.csv",
    }.get(method)
    if not pattern:
        return {}
    path = results / pattern
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def plot_final(rows, out):
    methods = [r["method"] for r in rows]
    vals = [float(r["final_accuracy"]) if r["final_accuracy"] else math.nan for r in rows]
    colors = ["#7a8599", "#2f80ed", "#27ae60", "#f2994a", "#0f766e", "#7c3aed", "#14b8a6", "#b45309", "#111827"]
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    bars = ax.bar(range(len(methods)), [0 if math.isnan(v) else v for v in vals], color=colors, width=0.68)
    for bar, val in zip(bars, vals):
        if math.isnan(val):
            bar.set_alpha(0.25)
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "failed/missing", ha="center", va="bottom", fontsize=8, rotation=90)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Final accuracy")
    ax.set_title("Split MNIST Domain-CL 2000: Final Accuracy")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=22, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_learning_curve(curve_path, graph_path, context_graph_path):
    if not curve_path.exists():
        return
    by_method = {}
    with curve_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_method.setdefault(row["method"], []).append((int(float(row["iteration"])), float(row["accuracy"]), int(float(row["context"]))))
    fig, ax = plt.subplots(figsize=(14, 7))
    for method in METHODS:
        points = sorted(by_method.get(method, []))
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], label=method, linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Domain-CL accuracy")
    ax.set_title("Split MNIST Domain-CL 2000: Learning Curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(graph_path, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    for method in METHODS:
        points = sorted(by_method.get(method, []))
        if not points:
            continue
        by_context = {}
        for _, acc, context in points:
            by_context.setdefault(context, []).append(acc)
        xs = sorted(by_context)
        ys = [sum(by_context[c]) / len(by_context[c]) for c in xs]
        ax.plot(xs, ys, marker="o", label=method, linewidth=1.8)
    ax.set_xlabel("Current context")
    ax.set_ylabel("Mean logged accuracy within context")
    ax.set_title("Split MNIST Domain-CL 2000: Context-Aggregated Learning Curve")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(context_graph_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results = Path(args.results_dir)
    status = load_status(results)
    files = find_result_files(results)
    time_files = find_time_files(results)
    accs = {m: read_float(p) for m, p in files.items() if p is not None}
    joint = accs.get("Joint")
    none = accs.get("None")
    rows = []
    for method in METHODS:
        acc = accs.get(method)
        st = status.get(method, {}).get("status", "missing")
        runtime = status.get(method, {}).get("runtime_seconds", "")
        if not runtime and time_files.get(method) is not None:
            runtime = f"{read_float(time_files[method]):.3f}"
        if not runtime:
            runtime = runtime_from_log(results, method)
        log_file = status.get(method, {}).get("log_file", "")
        if not log_file:
            log_file = str(results / "logs" / LOGS[method])
        rows.append({
            "method": method,
            "status": "success" if acc is not None and st != "failed" else st,
            "final_accuracy": "" if acc is None else f"{acc:.12f}",
            "gap_from_joint": "" if acc is None or joint is None else f"{joint - acc:.12f}",
            "improvement_over_none": "" if acc is None or none is None else f"{acc - none:.12f}",
            "runtime_seconds": runtime,
            "result_file": "" if files.get(method) is None else str(files[method]),
            "command": status.get(method, {}).get("command", COMMANDS[method]),
            "log_file": log_file,
        })
    with (results / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_final(rows, results / "splitMNIST_domain_2000_final_accuracy.png")
    plot_learning_curve(
        results / "learning_curve.csv",
        results / "splitMNIST_domain_2000_learning_curve.png",
        results / "splitMNIST_domain_2000_learning_curve_by_context.png",
    )
    successes = [r for r in rows if r["final_accuracy"]]
    closest = min([r for r in successes if r["method"] != "Joint"], key=lambda r: float(r["gap_from_joint"])) if joint is not None else None
    best = max([r for r in successes if r["method"] not in ("None", "Joint")], key=lambda r: float(r["improvement_over_none"])) if none is not None else None
    acc = {r["method"]: float(r["final_accuracy"]) for r in successes}
    lines = [
        "# PHASE 2 Report",
        "",
        "Experiment: Split MNIST Domain-CL, contexts=5, iters=2000, batch=128, acc-n=1024.",
        "Final accuracy is evaluated on the full test set. Generative Classifier and Separate Networks were skipped.",
        "Learning-curve evaluation was logged every 100 iterations, following the explicit Phase 2 evaluation-frequency setting.",
        "Note: Joint used the same comparator convention as Phase 1 (`--joint --iters=10000`), which the repository reports as 50,000 progress iterations in the learning curve.",
        "",
        "## Final Results",
        "",
        "| method | status | final accuracy | gap from Joint | improvement over None | runtime seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['method']} | {r['status']} | {r['final_accuracy']} | {r['gap_from_joint']} | {r['improvement_over_none']} | {r['runtime_seconds']} |")
    lines += ["", "## Commands", ""]
    for r in rows:
        lines += [f"### {r['method']}", "", "```powershell", r["command"], "```", ""]
    lines += [
        "## Comparisons",
        "",
        f"Closest method to Joint: {closest['method'] if closest else 'n/a'}",
        f"Best improvement over None: {best['method'] if best else 'n/a'}",
        f"Fourier helped vs LSR-lite: {acc.get('LSR-lite + Fourier', float('nan')) - acc.get('LSR-lite', float('nan')) if 'LSR-lite + Fourier' in acc and 'LSR-lite' in acc else 'n/a'}",
        f"ASW helped vs LSR-lite: {acc.get('LSR-lite + ASW', float('nan')) - acc.get('LSR-lite', float('nan')) if 'LSR-lite + ASW' in acc and 'LSR-lite' in acc else 'n/a'}",
        f"Fourier + ASW helped vs Fourier alone: {acc.get('LSR-lite + Fourier + ASW', float('nan')) - acc.get('LSR-lite + Fourier', float('nan')) if 'LSR-lite + Fourier + ASW' in acc and 'LSR-lite + Fourier' in acc else 'n/a'}",
        "",
        "## ASW Stats",
        "",
    ]
    for method in ["LSR-lite + ASW", "LSR-lite + Fourier + ASW"]:
        stats = load_asw_stats(results, method)
        if stats:
            lines.append(f"- {stats['method']}: mean={stats.get('asw_mean')}, min={stats.get('asw_min')}, max={stats.get('asw_max')}")
    failed = [r["method"] for r in rows if r["status"] != "success"]
    lines += [
        "",
        f"Failed methods: {', '.join(failed) if failed else 'none'}",
        "",
        "## Output Files",
        "",
        f"- summary.csv: {results / 'summary.csv'}",
        f"- learning_curve.csv: {results / 'learning_curve.csv'}",
        f"- final graph: {results / 'splitMNIST_domain_2000_final_accuracy.png'}",
        f"- learning curve: {results / 'splitMNIST_domain_2000_learning_curve.png'}",
        f"- context curve: {results / 'splitMNIST_domain_2000_learning_curve_by_context.png'}",
        f"- logs: {results / 'logs'}",
    ]
    (results / "PHASE2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(results / "summary.csv")
    print(results / "PHASE2_REPORT.md")


if __name__ == "__main__":
    main()
