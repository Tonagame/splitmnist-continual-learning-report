#!/usr/bin/env python3
"""Compare paper, GMvandeVen-code runs, and our clean-room runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
FROM_SCRATCH = ROOT / "results_from_scratch" / "classic_no_lsr_2000"

PAPER_CSV = ASSETS / "paper_vs_ours_splitMNIST_common_methods.csv"
GM_CSV = ASSETS / "splitMNIST_2000_all_scenarios_summary.csv"
OUR_CSV = FROM_SCRATCH / "combined_summary.csv"
TASK_FIX_CSV = ROOT / "results_from_scratch" / "task_protocol_fix_2000" / "summary.csv"

OUT_CSV = ASSETS / "paper_vs_gmvandeven_vs_from_scratch.csv"
OUT_PNG = ASSETS / "paper_vs_gmvandeven_vs_from_scratch.png"
OUT_MERGED_OUR_CSV = ASSETS / "from_scratch_classic_no_lsr_2000_summary.csv"

SCENARIO_ORDER = ["Class-CL", "Domain-CL", "Task-CL"]
METHOD_ORDER = [
    "None",
    "EWC",
    "LwF",
    "A-GEM",
    "Generative Classifier",
    "Separate Networks",
    "Joint",
]
SOURCE_ORDER = ["Paper Table 2", "GMvandeVen code run", "Our from-scratch code"]
SOURCE_COLORS = {
    "Paper Table 2": "#627187",
    "GMvandeVen code run": "#14796f",
    "Our from-scratch code": "#b85b1e",
}


def normalize_scenario(value: str) -> str:
    key = value.strip().lower().replace("-cl", "")
    return {
        "class": "Class-CL",
        "domain": "Domain-CL",
        "task": "Task-CL",
    }.get(key, value)


def add_row(rows, scenario, method, source, accuracy_percent, note=""):
    rows.append(
        {
            "scenario": normalize_scenario(str(scenario)),
            "method": str(method),
            "source": source,
            "accuracy_percent": accuracy_percent,
            "note": note,
        }
    )


def build_comparison() -> pd.DataFrame:
    rows = []

    paper_df = pd.read_csv(PAPER_CSV, keep_default_na=False)
    for row in paper_df.itertuples():
        if str(row.paper_percent).strip():
            add_row(rows, row.scenario, row.method, "Paper Table 2", float(row.paper_percent))

    gm_df = pd.read_csv(GM_CSV, keep_default_na=False)
    for row in gm_df.itertuples():
        if row.method not in METHOD_ORDER:
            continue
        if str(row.accuracy).strip():
            add_row(rows, row.scenario, row.method, "GMvandeVen code run", float(row.accuracy) * 100.0)
        elif row.method == "Generative Classifier":
            add_row(rows, row.scenario, row.method, "GMvandeVen code run", np.nan, "failed or skipped")

    our_df = pd.read_csv(OUR_CSV, keep_default_na=False)
    our_df = apply_task_protocol_fix(our_df)
    for row in our_df.itertuples():
        add_row(rows, row.scenario, row.method, "Our from-scratch code", float(row.final_accuracy) * 100.0)

    df = pd.DataFrame(rows)
    df["scenario"] = pd.Categorical(df["scenario"], SCENARIO_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df["source"] = pd.Categorical(df["source"], SOURCE_ORDER, ordered=True)
    return df.sort_values(["scenario", "method", "source"])


def apply_task_protocol_fix(our_df: pd.DataFrame) -> pd.DataFrame:
    """Use corrected Task-CL None/EWC runs when available.

    The initial all-methods run used full 10-class CE during Task-CL training.
    The corrected runs use the active task's allowed classes during training.
    """

    if not TASK_FIX_CSV.exists():
        return our_df

    fixed = pd.read_csv(TASK_FIX_CSV, keep_default_na=False)
    fixed = fixed[fixed["method"].isin(["None", "EWC"])].copy()
    if "command" in fixed.columns:
        fixed = fixed[~fixed["command"].str.contains("--ewc-lambda", regex=False)]
    fixed = fixed.drop_duplicates(subset=["scenario", "method"], keep="last")

    merged = our_df.copy()
    for row in fixed.itertuples():
        scenario = normalize_scenario(str(row.scenario)).replace("-CL", "").lower()
        mask = (merged["scenario"].astype(str).str.lower() == scenario) & (merged["method"] == row.method)
        if mask.any():
            merged.loc[mask, "final_accuracy"] = row.final_accuracy
            merged.loc[mask, "runtime_seconds"] = row.runtime_seconds
        else:
            merged = pd.concat([merged, pd.DataFrame([row._asdict()])], ignore_index=True)
    return merged


def plot(df: pd.DataFrame) -> None:
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(df["scenario"].astype(str))]
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(16, 5.2 * len(scenarios)), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    width = 0.24
    offsets = {
        "Paper Table 2": -width,
        "GMvandeVen code run": 0.0,
        "Our from-scratch code": width,
    }

    for ax, scenario in zip(axes, scenarios):
        scenario_df = df[df["scenario"].astype(str) == scenario]
        methods = [method for method in METHOD_ORDER if method in set(scenario_df["method"].astype(str))]
        x = np.arange(len(methods))

        for source in SOURCE_ORDER:
            values = []
            notes = []
            for method in methods:
                match = scenario_df[
                    (scenario_df["method"].astype(str) == method)
                    & (scenario_df["source"].astype(str) == source)
                ]
                if match.empty:
                    values.append(np.nan)
                    notes.append("")
                else:
                    values.append(float(match.iloc[0]["accuracy_percent"]) if str(match.iloc[0]["accuracy_percent"]) != "nan" else np.nan)
                    notes.append(str(match.iloc[0].get("note", "")))

            bars = ax.bar(
                x + offsets[source],
                values,
                width=width,
                label=source,
                color=SOURCE_COLORS[source],
            )
            for bar, value, note in zip(bars, values, notes):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 1.0,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )
                elif note:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        4.0,
                        "failed/skipped",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                        color="#7a2330",
                    )

        ax.set_title(scenario)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Final accuracy (%)")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper left", ncols=3)
    fig.suptitle("Split MNIST reproduction comparison: paper vs GMvandeVen code vs our from-scratch code", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    df = build_comparison()
    df.to_csv(OUT_CSV, index=False)
    merged_ours = apply_task_protocol_fix(pd.read_csv(OUR_CSV, keep_default_na=False))
    merged_ours.to_csv(OUT_MERGED_OUR_CSV, index=False)
    plot(df)
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_MERGED_OUR_CSV}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
