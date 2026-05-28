#!/usr/bin/env python
"""Import HW3 ACT scalar metrics into a local SwanLab run.

The training runs were executed with online logging disabled, so this script
replays the saved scalar CSVs into SwanLab. The resulting local run can be
opened with `swanlab watch` and used to export the required course figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import swanlab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("report/tables/act_training_loss.csv"),
        help="CSV parsed from the ACT training logs.",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("report/tables/act_dd_l1_by_checkpoint.csv"),
        help="CSV containing D-D offline Action L1 per checkpoint.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("swanlab_logs"),
        help="Directory where SwanLab local run files are written.",
    )
    parser.add_argument(
        "--project",
        default="hw3-act-calvin-generalization",
        help="SwanLab project name.",
    )
    parser.add_argument(
        "--experiment-name",
        default="act_a_vs_act_abc_replayed_scalars",
        help="SwanLab experiment name.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=400,
        help="Median smoothing window, in logged scalar points.",
    )
    return parser.parse_args()


def add_smoothed_loss(train_df: pd.DataFrame, window: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in train_df.sort_values(["model", "step"]).groupby("model"):
        group = group.copy()
        group["loss_median"] = group["loss"].rolling(window=window, min_periods=1).median()
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(["step", "model"])


def log_training(train_df: pd.DataFrame) -> None:
    for row in train_df.sort_values("step").itertuples(index=False):
        payload = {
            "train_loss_raw": float(row.loss),
            "train_loss_median_400": float(row.loss_median),
            "train_grad_norm": float(row.grad_norm),
            "learning_rate": float(row.lr),
        }
        if int(row.step) >= 5000:
            payload["train_loss_after_5k_median_400"] = float(row.loss_median)
        swanlab.log(payload, step=int(row.step))


def log_eval(eval_df: pd.DataFrame) -> None:
    for row in eval_df.sort_values("step").itertuples(index=False):
        swanlab.log(
            {
                "dd_action_l1_mean": float(row.dd_l1_mean),
                "dd_action_l1_last": float(row.dd_l1_last),
            },
            step=int(row.step),
        )


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    eval_df = pd.read_csv(args.eval_csv)
    train_df = add_smoothed_loss(train_df, args.smooth_window)

    args.logdir.mkdir(parents=True, exist_ok=True)
    for model, train_group in train_df.groupby("model", sort=True):
        eval_group = eval_df[eval_df["model"] == model].copy()
        swanlab.init(
            project=args.project,
            experiment_name=model,
            mode="local",
            logdir=str(args.logdir),
            tags=["hw3", "act", "calvin", "replayed-scalars", model],
            config={
                "assignment": "HW3 Task 2",
                "model": model,
                "train_data": "A only" if model == "ACT-A" else "A+B+C mixed",
                "zero_shot_test_env": "CALVIN D-D",
                "metric": "offline unnormalized Action L1",
                "train_csv": str(args.train_csv),
                "eval_csv": str(args.eval_csv),
                "smooth_window": args.smooth_window,
                "note": "Metrics replayed from original LeRobot logs and D-D evaluation JSONs.",
            },
        )
        log_training(train_group)
        log_eval(eval_group)
        swanlab.finish()
    print(f"SwanLab local runs saved under: {args.logdir.resolve()}")
    print(f"Open with: swanlab watch {args.logdir.resolve()}")


if __name__ == "__main__":
    main()
