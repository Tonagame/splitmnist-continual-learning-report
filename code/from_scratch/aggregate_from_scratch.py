#!/usr/bin/env python3
"""Aggregate from-scratch Split MNIST runs and create report graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCENARIO_ORDER = ["class", "domain", "task"]
METHOD_ORDER = [
    "None",
    "EWC",
    "LwF",
    "A-GEM",
    "Generative Classifier",
    "Separate Networks",
    "Joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder containing scenario result folders")
    parser.add_argument("--title", default="From-scratch Split MNIST classic methods")
    return parser.parse_args()


def ordered_categories(values, order):
    present = [item for item in order if item in set(values)]
    extra = sorted(set(values) - set(order))
    return present + extra


def read_summaries(root: Path) -> pd.DataFrame:
    frames = []
    for path in root.glob("splitmnist_*_*/summary.csv"):
        df = pd.read_csv(path, keep_default_na=False)
        df["source_dir"] = str(path.parent)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No summary.csv files found under {root}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["scenario", "method", "seed"], keep="last")
    df["accuracy_percent"] = df["final_accuracy"] * 100.0
    df["scenario"] = pd.Categorical(df["scenario"], ordered=True, categories=ordered_categories(df["scenario"], SCENARIO_ORDER))
    df["method"] = pd.Categorical(df["method"], ordered=True, categories=ordered_categories(df["method"], METHOD_ORDER))
    return df.sort_values(["scenario", "method"])


def read_curves(root: Path) -> pd.DataFrame | None:
    frames = []
    for path in root.glob("splitmnist_*_*/learning_curve.csv"):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path, keep_default_na=False)
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["scenario", "method", "iteration", "context"], keep="last")
    df["accuracy_percent"] = df["accuracy"] * 100.0
    return df


def plot_final_accuracy(df: pd.DataFrame, root: Path, title: str) -> Path:
    scenarios = ordered_categories(df["scenario"].astype(str), SCENARIO_ORDER)
    methods = ordered_categories(df["method"].astype(str), METHOD_ORDER)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.2 * len(scenarios), 6), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    colors = ["#4c6a88", "#14796f", "#b85b1e", "#6f42c1", "#5f7f2f", "#9b4d64", "#202938"]
    color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}

    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"].astype(str) == scenario].copy()
        sub = sub.sort_values("method")
        bars = ax.bar(
            sub["method"].astype(str),
            sub["accuracy_percent"],
            color=[color_map[m] for m in sub["method"].astype(str)],
        )
        ax.set_title(f"{scenario.capitalize()}-CL")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Final accuracy (%)")
    fig.suptitle(title)
    fig.tight_layout()
    out = root / "from_scratch_classic_final_accuracy.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_learning_curves(curves: pd.DataFrame, root: Path, title: str) -> Path:
    scenarios = ordered_categories(curves["scenario"].astype(str), SCENARIO_ORDER)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4.5 * len(scenarios)), sharex=False)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sub = curves[curves["scenario"].astype(str) == scenario].copy()
        for method, method_df in sub.groupby("method", sort=False):
            method_df = method_df.sort_values("iteration")
            ax.plot(method_df["iteration"], method_df["accuracy_percent"], label=str(method), linewidth=1.8)
        ax.set_title(f"{scenario.capitalize()}-CL")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title + " learning curves")
    fig.tight_layout()
    out = root / "from_scratch_classic_learning_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(df: pd.DataFrame, curves: pd.DataFrame | None, root: Path, final_graph: Path, curve_graph: Path | None) -> Path:
    report = root / "FROM_SCRATCH_CLASSIC_REPORT.md"
    lines = [
        "# From-Scratch Classic Split MNIST Report",
        "",
        "This report summarizes the independent implementation runs. LSR variants were not rerun in this batch by request.",
        "",
        "## Final Accuracy",
        "",
        "| Scenario | Method | Final accuracy | Runtime seconds |",
        "|---|---|---:|---:|",
    ]
    for row in df.itertuples():
        lines.append(f"| {row.scenario} | {row.method} | {row.final_accuracy:.6f} | {row.runtime_seconds:.1f} |")
    lines.extend([
        "",
        "## Files",
        "",
        f"- Combined summary: `{(root / 'combined_summary.csv').name}`",
        f"- Final accuracy graph: `{final_graph.name}`",
    ])
    if curve_graph is not None:
        lines.append(f"- Learning curve graph: `{curve_graph.name}`")
    if curves is not None:
        lines.append(f"- Combined learning curves: `{(root / 'combined_learning_curve.csv').name}`")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    summary = read_summaries(root)
    summary_path = root / "combined_summary.csv"
    summary.to_csv(summary_path, index=False)

    final_graph = plot_final_accuracy(summary, root, args.title)

    curves = read_curves(root)
    curve_graph = None
    if curves is not None:
        curves_path = root / "combined_learning_curve.csv"
        curves.to_csv(curves_path, index=False)
        curve_graph = plot_learning_curves(curves, root, args.title)

    report = write_report(summary, curves, root, final_graph, curve_graph)
    print(f"Saved {summary_path}")
    print(f"Saved {final_graph}")
    if curve_graph is not None:
        print(f"Saved {curve_graph}")
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
