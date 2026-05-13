#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = [
    "None", "EWC", "LwF", "A-GEM", "Generative Classifier",
    "LSR-lite", "LSR-lite + Fourier", "LSR-lite + ASW",
    "LSR-lite + Fourier + ASW", "Joint",
]


def latest(paths):
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def read_float(path):
    return float(path.read_text(encoding="utf-8").strip())


def find_result_files(results):
    return {
        "None": latest(results.glob("acc-splitMNIST5-class--F-784x400x400_c10--i2000-lr0.001-b128-adam-all.txt")),
        "EWC": latest(results.glob("acc-splitMNIST5-class--F-784x400x400_c10--i2000-lr0.001-b128-adam-all--PReg*-offline.txt")),
        "LwF": latest(results.glob("acc-splitMNIST5-class--F-784x400x400_c10--i2000-lr0.001-b128-adam-all--current-KD2.0.txt")),
        "A-GEM": latest(results.glob("acc-splitMNIST5-class--F-784x400x400_c10--i2000-lr0.001-b128-adam-all--buffer-A-GEM*.txt")),
        "Generative Classifier": latest(results.glob("acc-splitMNIST5-class--x10-VAE=F-784x400x400--z100--sigmoid--i2000-lr0.001-b128-adam-BCE--S50.txt")),
        "LSR-lite": latest(results.glob("acc-splitMNIST5-class--LSR-lite--i2000-b128-bud100-kd1.0-feat1.0.txt")),
        "LSR-lite + Fourier": latest(results.glob("acc-splitMNIST5-class--LSR-lite-Fourier--i2000-b128-bud100-kd1.0-feat1.0-fft0.1.txt")),
        "LSR-lite + ASW": latest(results.glob("acc-splitMNIST5-class--LSR-lite-ASW--i2000-b128-bud100-kd1.0-feat0.5-asw0.5-2.0-eps1e-08.txt")),
        "LSR-lite + Fourier + ASW": latest(results.glob("acc-splitMNIST5-class--LSR-lite-Fourier-ASW--i2000-b128-bud100-kd1.0-feat0.5-fft0.05-asw0.5-2.0-eps1e-08.txt")),
        "Joint": latest(results.glob("acc-splitMNIST5-Joint-class--F-784x400x400_c10--i10000-lr0.001-b128-adam-all.txt")),
    }


def load_status(results):
    status_path = results / "run_status.csv"
    if not status_path.exists():
        return {}
    with status_path.open(newline="", encoding="utf-8-sig") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def load_asw_stats(results, method):
    pattern = {
        "LSR-lite + ASW": "metrics-splitMNIST5-class--LSR-lite-ASW--i2000-b128-bud100-kd1.0-feat0.5-asw0.5-2.0-eps1e-08.csv",
        "LSR-lite + Fourier + ASW": "metrics-splitMNIST5-class--LSR-lite-Fourier-ASW--i2000-b128-bud100-kd1.0-feat0.5-fft0.05-asw0.5-2.0-eps1e-08.csv",
    }.get(method)
    if not pattern:
        return {}
    path = results / pattern
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {key: row.get(key, "") for key in ["asw_mean", "asw_min", "asw_max"]}


def plot_final(rows, out):
    methods = [r["method"] for r in rows]
    vals = [float(r["final_accuracy"]) if r["final_accuracy"] else math.nan for r in rows]
    colors = ["#7a8599", "#2f80ed", "#27ae60", "#f2994a", "#9b51e0",
              "#0f766e", "#7c3aed", "#14b8a6", "#b45309", "#111827"]
    fig, ax = plt.subplots(figsize=(15, 6.6))
    bars = ax.bar(range(len(methods)), [0 if math.isnan(v) else v for v in vals], color=colors, width=0.68)
    for bar, val in zip(bars, vals):
        if math.isnan(val):
            bar.set_alpha(0.25)
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "failed/missing",
                    ha="center", va="bottom", fontsize=8, rotation=90)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Final accuracy")
    ax.set_title("Split MNIST Class-CL 2000: Final Accuracy")
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
        if not points:
            continue
        ax.plot([p[0] for p in points], [p[1] for p in points], label=method, linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average accuracy")
    ax.set_title("Split MNIST Class-CL 2000: Learning Curve")
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
        for iteration, acc, context in points:
            by_context.setdefault(context, []).append(acc)
        xs = sorted(by_context)
        ys = [sum(by_context[c]) / len(by_context[c]) for c in xs]
        ax.plot(xs, ys, marker="o", label=method, linewidth=1.8)
    ax.set_xlabel("Current context")
    ax.set_ylabel("Mean logged accuracy within context")
    ax.set_title("Split MNIST Class-CL 2000: Context-Aggregated Learning Curve")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(context_graph_path, dpi=170)
    plt.close(fig)


def write_report(results, rows, status):
    joint = next((float(r["final_accuracy"]) for r in rows if r["method"] == "Joint" and r["final_accuracy"]), None)
    none = next((float(r["final_accuracy"]) for r in rows if r["method"] == "None" and r["final_accuracy"]), None)
    successes = [r for r in rows if r["final_accuracy"]]
    closest = min([r for r in successes if r["method"] != "Joint"], key=lambda r: float(r["gap_from_joint"])) if joint is not None else None
    best = max([r for r in successes if r["method"] not in ("None", "Joint")],
               key=lambda r: float(r["improvement_over_none"])) if none is not None else None
    acc = {r["method"]: float(r["final_accuracy"]) for r in successes}
    lines = [
        "# PHASE 1 Report",
        "",
        "Experiment: Split MNIST Class-CL, contexts=5, iters=2000, batch=128, acc-n=1024.",
        "Final accuracy is evaluated on the full test set with true Class-CL inference over all 10 classes.",
        "",
        "## Final Results",
        "",
        "| method | status | final accuracy | gap from Joint | improvement over None | runtime seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        st = status.get(r["method"], {})
        lines.append("| {method} | {status} | {acc} | {gap} | {imp} | {rt} |".format(
            method=r["method"], status=r["status"], acc=r["final_accuracy"] or "",
            gap=r["gap_from_joint"] or "", imp=r["improvement_over_none"] or "",
            rt=st.get("runtime_seconds", ""),
        ))
    lines += ["", "## Commands", ""]
    for r in rows:
        cmd = status.get(r["method"], {}).get("command") or r.get("command", "")
        lines += [f"### {r['method']}", "", "```powershell", cmd, "```", ""]
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
            lines.append(f"- {method}: mean={stats.get('asw_mean')}, min={stats.get('asw_min')}, max={stats.get('asw_max')}")
    failed = [r["method"] for r in rows if r["status"] != "success"]
    lines += [
        "",
        f"Failed or missing methods: {', '.join(failed) if failed else 'none'}",
        "",
        "## Output Files",
        "",
        f"- summary.csv: {results / 'summary.csv'}",
        f"- learning_curve.csv: {results / 'learning_curve.csv'}",
        f"- final graph: {results / 'splitMNIST_class_2000_final_accuracy.png'}",
        f"- learning curve: {results / 'splitMNIST_class_2000_learning_curve.png'}",
        f"- logs: {results / 'logs'}",
    ]
    (results / "PHASE1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results = Path(args.results_dir)
    status = load_status(results)
    files = find_result_files(results)
    accs = {m: read_float(p) for m, p in files.items() if p is not None}
    joint = accs.get("Joint")
    none = accs.get("None")
    rows = []
    for method in METHODS:
        acc = accs.get(method)
        st = status.get(method, {}).get("status", "missing")
        rows.append({
            "method": method,
            "status": "success" if acc is not None and st != "failed" else st,
            "final_accuracy": "" if acc is None else f"{acc:.12f}",
            "gap_from_joint": "" if acc is None or joint is None else f"{joint - acc:.12f}",
            "improvement_over_none": "" if acc is None or none is None else f"{acc - none:.12f}",
            "result_file": "" if files.get(method) is None else str(files[method]),
            "command": status.get(method, {}).get("command", ""),
            "log_file": status.get(method, {}).get("log_file", ""),
        })
    with (results / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_final(rows, results / "splitMNIST_class_2000_final_accuracy.png")
    plot_learning_curve(
        results / "learning_curve.csv",
        results / "splitMNIST_class_2000_learning_curve.png",
        results / "splitMNIST_class_2000_learning_curve_by_context.png",
    )
    write_report(results, rows, status)
    print(results / "summary.csv")
    print(results / "PHASE1_REPORT.md")


if __name__ == "__main__":
    main()
