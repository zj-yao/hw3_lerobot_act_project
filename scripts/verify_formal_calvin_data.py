#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import torch

DEFAULT_LEROBOT_REPO = Path("/data/Vscode_project/Deeplearning/Homework_3/repos/lerobot_v033")
LEROBOT_SRC = Path(os.environ.get("LEROBOT_REPO", DEFAULT_LEROBOT_REPO)) / "src"
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig


DATASETS = {
    "abc": {
        "repo_id": "fywang/calvin-task-ABC-D-lerobot",
        "expected_episodes": 18957,
        "expected_data_files": 18957,
    },
    "d": {
        "repo_id": "fywang/calvin-task-D-D-lerobot",
        "expected_episodes": 6135,
        "expected_data_files": 6135,
    },
}

REQUIRED_FEATURES = {
    "observation.images.top",
    "observation.images.wrist",
    "observation.state",
    "action",
    "task_index",
    "timestamp",
    "episode_index",
    "frame_index",
    "index",
}


def describe_value(value) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return f"shape={tuple(shape)}, dtype={dtype}"
    return f"type={type(value).__name__}, value={value!r}"


def count_parquet(root: Path) -> int:
    return sum(1 for _ in (root / "data").rglob("*.parquet"))


def verify_one(name: str, storage_root: Path, sample_episodes: list[int], chunk_size: int) -> None:
    spec = DATASETS[name]
    repo_id = spec["repo_id"]
    root = storage_root / repo_id

    print(f"=== {name}: {repo_id} ===", flush=True)
    print(f"root={root}", flush=True)
    if not root.exists():
        raise FileNotFoundError(root)

    parquet_count = count_parquet(root)
    print(f"parquet_count={parquet_count}", flush=True)
    if parquet_count != spec["expected_data_files"]:
        raise AssertionError(f"{repo_id}: expected {spec['expected_data_files']} parquet, got {parquet_count}")

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    print(f"LeRobot CODEBASE_VERSION={CODEBASE_VERSION}", flush=True)
    print(f"dataset_codebase_version={meta.info['codebase_version']}", flush=True)
    print(f"total_episodes={meta.total_episodes}", flush=True)
    print(f"total_frames={meta.total_frames}", flush=True)
    print(f"fps={meta.fps}", flush=True)
    print(f"robot_type={meta.robot_type}", flush=True)
    print(f"image_keys={meta.image_keys}", flush=True)

    if meta.total_episodes != spec["expected_episodes"]:
        raise AssertionError(f"{repo_id}: expected {spec['expected_episodes']} episodes, got {meta.total_episodes}")
    missing = REQUIRED_FEATURES.difference(meta.features)
    if missing:
        raise AssertionError(f"{repo_id}: missing features {sorted(missing)}")

    policy_cfg = ACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        pretrained_backbone_weights=None,
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    print(f"delta_timestamps={delta_timestamps}", flush=True)

    episodes = [ep for ep in sample_episodes if ep < meta.total_episodes]
    dataset = LeRobotDataset(repo_id, root=root, episodes=episodes, delta_timestamps=delta_timestamps)
    print(f"selected_episodes={episodes}", flush=True)
    print(f"selected_frames={len(dataset)}", flush=True)
    print(f"hf_features={dataset.hf_features}", flush=True)

    sample_indices = [0, min(len(dataset) - 1, max(1, chunk_size // 2)), len(dataset) - 1]
    for idx in sample_indices:
        sample = dataset[idx]
        print(f"sample[{idx}]", flush=True)
        for key in sorted(sample):
            print(f"  {key}: {describe_value(sample[key])}", flush=True)

        assert sample["observation.images.top"].shape == torch.Size([3, 200, 200])
        assert sample["observation.images.wrist"].shape == torch.Size([3, 84, 84])
        assert sample["observation.state"].shape == torch.Size([15])
        assert sample["action"].shape == torch.Size([chunk_size, 7])
        assert sample["action_is_pad"].shape == torch.Size([chunk_size])
        assert isinstance(sample["task"], str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/yzj/calvin_lerobot_act_hw2/processed_datasets/lerobot"),
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["abc", "d"])
    parser.add_argument(
        "--sample-episodes",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Use contiguous episode ids with LeRobot v0.3.3; sparse ids can trip selected-episode indexing.",
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    for name in args.datasets:
        verify_one(name, args.root, args.sample_episodes, args.chunk_size)


if __name__ == "__main__":
    main()
