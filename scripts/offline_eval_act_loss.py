#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_PROJECT = Path("/data/Vscode_project/Deeplearning/Homework_3")
DEFAULT_STORAGE = Path("/data/yzj/calvin_lerobot_act_hw2")
LEROBOT_SRC = Path(os.environ.get("LEROBOT_REPO", DEFAULT_PROJECT / "repos" / "lerobot_v033")) / "src"
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute offline ACT loss on a LeRobot dataset.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_STORAGE
        / "checkpoints"
        / "act_calvin_abc_full_20260525_bs64_041136"
        / "checkpoints"
        / "050000"
        / "pretrained_model",
    )
    parser.add_argument("--repo-id", default="fywang/calvin-task-D-D-lerobot")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_STORAGE / "processed_datasets" / "lerobot" / "fywang" / "calvin-task-D-D-lerobot",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode",
        choices=["inference_l1", "train_forward"],
        default="inference_l1",
        help=(
            "inference_l1 uses policy.predict_action_chunk in eval mode and compares unnormalized actions. "
            "train_forward uses policy.forward in train mode and reports LeRobot's normalized training loss."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_STORAGE / "eval_outputs" / "offline_eval_DD_050000_loss.json",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.device = str(device)

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta_timestamps)

    # Match the training path: the original train config used ImageNet stats for camera inputs.
    for key in dataset.meta.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    if args.mode == "inference_l1":
        policy.eval()
    else:
        # ACT's VAE branch computes KL only in training mode. This is useful for train-loss diagnostics,
        # but inference_l1 is the cleaner offline proxy for deployment behavior.
        policy.train()

    if hasattr(policy_cfg, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=policy_cfg.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    losses: list[float] = []
    l1_losses: list[float] = []
    kld_losses: list[float] = []
    start = time.perf_counter()
    iterator = cycle(dataloader)

    print(
        f"Evaluating {args.repo_id} with checkpoint={args.checkpoint} "
        f"batches={args.num_batches} batch_size={args.batch_size} device={device}",
        flush=True,
    )

    with torch.no_grad():
        for idx in range(args.num_batches):
            batch = next(iterator)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=device.type == "cuda")

            if args.mode == "inference_l1":
                pred_actions = policy.predict_action_chunk(batch)
                action_is_pad = batch["action_is_pad"].unsqueeze(-1)
                l1_loss = (F.l1_loss(pred_actions, batch["action"], reduction="none") * ~action_is_pad).mean()
                loss = l1_loss
                losses.append(float(loss.item()))
                l1_losses.append(float(l1_loss.item()))
            else:
                loss, output = policy.forward(batch)
                losses.append(float(loss.item()))
                l1_losses.append(float(output["l1_loss"]))
                if "kld_loss" in output:
                    kld_losses.append(float(output["kld_loss"]))

            if (idx + 1) % max(1, args.num_batches // 10) == 0 or idx == 0:
                print(
                    f"batch {idx + 1:04d}/{args.num_batches} "
                    f"loss={losses[-1]:.4f} mean_loss={mean(losses):.4f} "
                    f"mean_l1={mean(l1_losses):.4f}",
                    flush=True,
                )

    result = {
        "checkpoint": str(args.checkpoint),
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "num_frames": dataset.num_frames,
        "num_episodes": dataset.num_episodes,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_samples": len(losses) * args.batch_size,
        "device": str(device),
        "mode": args.mode,
        "loss_mean": mean(losses),
        "loss_last": losses[-1],
        "l1_loss_mean": mean(l1_losses),
        "l1_loss_last": l1_losses[-1],
        "kld_loss_mean": mean(kld_losses) if kld_losses else None,
        "kld_loss_last": kld_losses[-1] if kld_losses else None,
        "elapsed_s": time.perf_counter() - start,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved result to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
