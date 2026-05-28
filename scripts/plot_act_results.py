#!/usr/bin/env python
"""Build ACT training and D-D offline evaluation plots for HW3 task 2."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PROJECT = Path("/data/Vscode_project/Deeplearning/Homework_3")
DEFAULT_STORAGE = Path("/data/yzj/calvin_lerobot_act_hw2")

TRAIN_LINE_RE = re.compile(
    r"INFO (?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) .*"
    r" step:(?P<display_step>\S+) .*"
    r" epch:(?P<epoch>[0-9.]+) "
    r"loss:(?P<loss>[0-9.eE+-]+) "
    r"grdn:(?P<grad_norm>[0-9.eE+-]+) "
    r"lr:(?P<lr>[0-9.eE+-]+)"
)

COLORS = {
    "ACT-A": "#0B6E69",
    "ACT-ABC": "#B35C1E",
}


@dataclass(frozen=True)
class TrainSource:
    model: str
    path: Path
    start_step: int
    max_entries: int | None = None


def parse_train_source(source: TrainSource, log_freq: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with source.path.open("r", errors="replace") as handle:
        for line in handle:
            match = TRAIN_LINE_RE.search(line)
            if not match:
                continue
            if source.max_entries is not None and len(rows) >= source.max_entries:
                break
            local_index = len(rows) + 1
            rows.append(
                {
                    "model": source.model,
                    "step": source.start_step + local_index * log_freq,
                    "timestamp": f"{match.group('date')} {match.group('time')}",
                    "display_step": match.group("display_step"),
                    "epoch": float(match.group("epoch")),
                    "loss": float(match.group("loss")),
                    "grad_norm": float(match.group("grad_norm")),
                    "lr": float(match.group("lr")),
                    "source_log": str(source.path),
                }
            )
    return rows


def load_training_rows(storage: Path, log_freq: int) -> pd.DataFrame:
    log_dir = storage / "logs"
    sources = [
        TrainSource(
            model="ACT-A",
            path=log_dir / "train_act_calvin_a_clean_20260527_a_clean_bs64_200k_070441.log",
            start_step=0,
            max_entries=6500,
        ),
        TrainSource(
            model="ACT-A",
            path=log_dir / "train_act_calvin_a_clean_20260527_a_clean_bs64_200k_070441_resume_after_cache_migration.log",
            start_step=65000,
        ),
        TrainSource(
            model="ACT-ABC",
            path=log_dir / "train_act_calvin_abc_clean_20260526_abc_clean_bs64_200k_022517.log",
            start_step=0,
        ),
    ]

    rows: list[dict[str, object]] = []
    for source in sources:
        if not source.path.exists():
            raise FileNotFoundError(source.path)
        rows.extend(parse_train_source(source, log_freq))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No training log rows were parsed.")
    return df.sort_values(["model", "step"]).reset_index(drop=True)


def load_eval_rows(storage: Path, steps: list[int]) -> pd.DataFrame:
    eval_dir = storage / "eval_outputs"
    rows = []
    for model in ["ACT-A", "ACT-ABC"]:
        prefix = "ACTA" if model == "ACT-A" else "ACTABC"
        for step in steps:
            path = eval_dir / f"offline_eval_{prefix}_DD_{step:06d}_l1_200b.json"
            if not path.exists():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text())
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "dd_l1_mean": float(data["l1_loss_mean"]),
                    "dd_l1_last": float(data["l1_loss_last"]),
                    "num_batches": int(data["num_batches"]),
                    "num_samples": int(data["num_samples"]),
                    "elapsed_s": float(data["elapsed_s"]),
                    "json_path": str(path),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "step"]).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def add_checkpoint_lines(ax: plt.Axes, steps: list[int]) -> None:
    for step in steps:
        ax.axvline(step / 1000, color="#D6D6D6", linewidth=0.6, zorder=0)


def plot_training_loss(train_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.3), dpi=180)
    for model, group in train_df.groupby("model"):
        group = group.sort_values("step")
        smooth = group["loss"].rolling(window=200, min_periods=1).median()
        ax.plot(group["step"] / 1000, group["loss"], color=COLORS[model], alpha=0.08, linewidth=0.6)
        ax.plot(group["step"] / 1000, smooth, color=COLORS[model], linewidth=1.7, label=f"{model} median-200")
    ax.set_yscale("log")
    ax.set_xlabel("Training step (K)")
    ax.set_ylabel("Training loss, log scale")
    ax.set_title("ACT Training Loss")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "act_train_loss_full_log.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.3), dpi=180)
    late = train_df[train_df["step"] >= 5000].copy()
    for model, group in late.groupby("model"):
        group = group.sort_values("step")
        smooth = group["loss"].rolling(window=400, min_periods=1).median()
        ax.plot(group["step"] / 1000, smooth, color=COLORS[model], linewidth=1.9, label=f"{model} median-400")
    ax.set_xlabel("Training step (K)")
    ax.set_ylabel("Training loss")
    ax.set_title("ACT Training Loss After Warmup")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "act_train_loss_after_5k.png")
    plt.close(fig)


def plot_eval_curve(eval_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=180)
    for model, group in eval_df.groupby("model"):
        group = group.sort_values("step")
        ax.plot(
            group["step"] / 1000,
            group["dd_l1_mean"],
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            color=COLORS[model],
            label=model,
        )
    best = eval_df.loc[eval_df["dd_l1_mean"].idxmin()]
    ax.scatter(
        [best["step"] / 1000],
        [best["dd_l1_mean"]],
        s=90,
        marker="*",
        color="#222222",
        zorder=5,
        label=f"best {best['model']} {int(best['step']) // 1000}K",
    )
    ax.set_xlabel("Checkpoint step (K)")
    ax.set_ylabel("D-D offline Action L1 mean")
    ax.set_title("Zero-shot D-D Offline Action Error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "act_dd_l1_curve.png")
    plt.close(fig)


def plot_combined(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path, eval_steps: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=180)

    ax = axes[0]
    late = train_df[train_df["step"] >= 5000].copy()
    add_checkpoint_lines(ax, eval_steps)
    for model, group in late.groupby("model"):
        group = group.sort_values("step")
        smooth = group["loss"].rolling(window=400, min_periods=1).median()
        ax.plot(group["step"] / 1000, smooth, color=COLORS[model], linewidth=1.8, label=model)
    ax.set_xlabel("Training step (K)")
    ax.set_ylabel("Training loss")
    ax.set_title("Train loss, median-400")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    for model, group in eval_df.groupby("model"):
        group = group.sort_values("step")
        ax.plot(
            group["step"] / 1000,
            group["dd_l1_mean"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=COLORS[model],
            label=model,
        )
    ax.set_xlabel("Checkpoint step (K)")
    ax.set_ylabel("D-D Action L1 mean")
    ax.set_title("Zero-shot D-D proxy")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "act_train_and_dd_eval.png")
    plt.close(fig)


def write_summary(eval_df: pd.DataFrame, train_df: pd.DataFrame, out_dir: Path) -> None:
    final = eval_df[eval_df["step"] == 200000].set_index("model")
    best = eval_df.loc[eval_df.groupby("model")["dd_l1_mean"].idxmin()].sort_values("model")
    final_delta = final.loc["ACT-A", "dd_l1_mean"] - final.loc["ACT-ABC", "dd_l1_mean"]
    final_rel = final_delta / final.loc["ACT-A", "dd_l1_mean"] * 100

    lines = [
        "# ACT Task 2 Curve Summary",
        "",
        "## D-D Offline L1 By Checkpoint",
        "",
        "| model | step | D-D L1 mean | D-D L1 last | samples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in eval_df.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['step'])} | {row['dd_l1_mean']:.10f} | "
            f"{row['dd_l1_last']:.10f} | {int(row['num_samples'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Checkpoint On D-D Offline L1",
            "",
            "| model | best step | best D-D L1 mean |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in best.iterrows():
        lines.append(f"| {row['model']} | {int(row['step'])} | {row['dd_l1_mean']:.10f} |")

    lines.extend(
        [
            "",
            "## Final Checkpoint Comparison",
            "",
            f"At 200000 steps, ACT-A D-D L1 mean is {final.loc['ACT-A', 'dd_l1_mean']:.10f}.",
            f"At 200000 steps, ACT-ABC D-D L1 mean is {final.loc['ACT-ABC', 'dd_l1_mean']:.10f}.",
            f"ACT-ABC is lower by {final_delta:.10f} absolute L1, or {final_rel:.2f}% relative to ACT-A.",
            "",
            "## Parsed Training Log Rows",
            "",
            "| model | rows | min step | max step | final parsed loss |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, group in train_df.groupby("model"):
        group = group.sort_values("step")
        lines.append(
            f"| {model} | {len(group)} | {int(group['step'].min())} | "
            f"{int(group['step'].max())} | {group['loss'].iloc[-1]:.6f} |"
        )

    (out_dir / "act_curve_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ACT training and D-D offline evaluation curves.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=[25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.project / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_training_rows(args.storage, args.log_freq)
    eval_df = load_eval_rows(args.storage, args.eval_steps)

    save_csv(train_df, out_dir / "act_training_loss.csv")
    save_csv(eval_df, out_dir / "act_dd_l1_by_checkpoint.csv")
    plot_training_loss(train_df, out_dir)
    plot_eval_curve(eval_df, out_dir)
    plot_combined(train_df, eval_df, out_dir, args.eval_steps)
    write_summary(eval_df, train_df, out_dir)

    print(f"wrote {len(train_df)} training rows")
    print(f"wrote {len(eval_df)} eval rows")
    print(f"reports: {out_dir}")


if __name__ == "__main__":
    main()
