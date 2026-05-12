#!/usr/bin/env python3
"""Create simple graphs from from-scratch summary.csv files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)
    df = df.drop_duplicates(subset=["scenario", "method", "seed"], keep="last")
    df["final_accuracy_percent"] = df["final_accuracy"] * 100.0
    df = df.sort_values(["scenario", "final_accuracy_percent"], ascending=[True, False])

    width = max(10, 0.65 * len(df))
    fig, ax = plt.subplots(figsize=(width, 6))
    labels = [f"{row.scenario}\n{row.method}" for row in df.itertuples()]
    bars = ax.bar(labels, df["final_accuracy_percent"], color="#14796f")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(args.title or "From-scratch Split MNIST final accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path = results_dir / "from_scratch_final_accuracy.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
