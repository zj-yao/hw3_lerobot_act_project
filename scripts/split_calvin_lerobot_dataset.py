#!/usr/bin/env python
"""Split the local CALVIN ABC-D LeRobot dataset into clean ABC and A-only train sets."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_STORAGE = Path("/data/yzj/calvin_lerobot_act_hw2")
DEFAULT_LEROBOT_HOME = DEFAULT_STORAGE / "processed_datasets" / "lerobot"
DEFAULT_SOURCE_ROOT = DEFAULT_LEROBOT_HOME / "fywang" / "calvin-task-ABC-D-lerobot"
DEFAULT_METADATA_ROOT = (
    DEFAULT_STORAGE
    / "raw_datasets"
    / "calvin_metadata"
    / "task_ABC_D_official_annotations"
    / "task_ABC_D"
)
DEFAULT_ABC_REPO_ID = "local/calvin-task-ABC-train-lerobot"
DEFAULT_A_REPO_ID = "local/calvin-task-A-train-lerobot"
DEFAULT_ABC_OUTPUT_ROOT = DEFAULT_LEROBOT_HOME / DEFAULT_ABC_REPO_ID
DEFAULT_A_OUTPUT_ROOT = DEFAULT_LEROBOT_HOME / DEFAULT_A_REPO_ID


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def data_path_for_episode(info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))


def make_scalar_stats(values: np.ndarray | list[int] | range) -> dict[str, list[float | int]]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "min": [int(arr.min())],
        "max": [int(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.size)],
    }


def update_index_stats(
    stats: dict[str, Any],
    *,
    episode_index: int,
    global_frame_start: int,
    length: int,
    task_index: int,
) -> dict[str, Any]:
    updated = copy.deepcopy(stats)
    updated["episode_index"] = make_scalar_stats([episode_index] * length)
    updated["frame_index"] = make_scalar_stats(range(length))
    updated["index"] = make_scalar_stats(range(global_frame_start, global_frame_start + length))
    updated["task_index"] = make_scalar_stats([task_index] * length)
    return updated


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        return table
    return table.set_column(column_index, name, values)


def rewrite_episode_table(
    table: pa.Table,
    *,
    episode_index: int,
    global_frame_start: int,
    task_index: int,
) -> pa.Table:
    length = table.num_rows
    table = replace_column(table, "episode_index", pa.array([episode_index] * length, type=pa.int64()))
    table = replace_column(table, "frame_index", pa.array(range(length), type=pa.int64()))
    table = replace_column(
        table,
        "index",
        pa.array(range(global_frame_start, global_frame_start + length), type=pa.int64()),
    )
    table = replace_column(table, "task_index", pa.array([task_index] * length, type=pa.int64()))
    return table


def validate_v21_source(source_root: Path, source_info: dict[str, Any]) -> None:
    if source_info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected a v2.1 LeRobot dataset, got {source_info.get('codebase_version')}")
    if source_info.get("total_videos", 0) != 0:
        raise ValueError("This splitter currently handles embedded-image parquet datasets only.")
    if not (source_root / "meta/episodes.jsonl").is_file():
        raise FileNotFoundError(source_root / "meta/episodes.jsonl")
    if not (source_root / "meta/episodes_stats.jsonl").is_file():
        raise FileNotFoundError(source_root / "meta/episodes_stats.jsonl")


def build_task_mapping(source_episodes: list[dict[str, Any]], selected_source_episode_ids: list[int]) -> dict[str, int]:
    task_to_new_index: dict[str, int] = {}
    for source_episode_id in selected_source_episode_ids:
        tasks = source_episodes[source_episode_id]["tasks"]
        if not tasks:
            raise ValueError(f"Episode {source_episode_id} has no task text.")
        for task in tasks:
            if task not in task_to_new_index:
                task_to_new_index[task] = len(task_to_new_index)
    return task_to_new_index


def prepare_output_root(output_root: Path, overwrite: bool) -> Path:
    tmp_root = output_root.with_name(f"{output_root.name}.tmp-{os.getpid()}")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(output_root)
    tmp_root.mkdir(parents=True)
    return tmp_root


def commit_output_root(tmp_root: Path, output_root: Path) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.rename(output_root)


def write_subset_dataset(
    *,
    source_root: Path,
    output_root: Path,
    output_repo_id: str,
    selected_source_episode_ids: list[int],
    scene_by_source_episode: dict[int, str],
    overwrite: bool = False,
    progress_every: int = 250,
) -> None:
    source_info = read_json(source_root / "meta/info.json")
    validate_v21_source(source_root, source_info)
    source_episodes = read_jsonl(source_root / "meta/episodes.jsonl")
    source_stats_rows = read_jsonl(source_root / "meta/episodes_stats.jsonl")
    source_stats = {row["episode_index"]: row["stats"] for row in source_stats_rows}

    if [row["episode_index"] for row in source_episodes] != list(range(len(source_episodes))):
        raise ValueError("Source episodes.jsonl must be sorted and contiguous by episode_index.")
    if not selected_source_episode_ids:
        raise ValueError("No episodes selected.")

    task_to_new_index = build_task_mapping(source_episodes, selected_source_episode_ids)
    new_info = copy.deepcopy(source_info)
    total_frames = sum(int(source_episodes[ep_id]["length"]) for ep_id in selected_source_episode_ids)
    chunks_size = int(new_info["chunks_size"])
    new_info["total_episodes"] = len(selected_source_episode_ids)
    new_info["total_frames"] = total_frames
    new_info["total_tasks"] = len(task_to_new_index)
    new_info["total_chunks"] = (len(selected_source_episode_ids) + chunks_size - 1) // chunks_size
    new_info["splits"] = {"train": f"0:{len(selected_source_episode_ids)}"}

    tmp_root = prepare_output_root(output_root, overwrite)
    started_at = time.time()

    tasks_rows = [
        {"task_index": task_index, "task": task}
        for task, task_index in sorted(task_to_new_index.items(), key=lambda item: item[1])
    ]
    write_json(tmp_root / "meta/info.json", new_info)
    write_jsonl(tmp_root / "meta/tasks.jsonl", tasks_rows)

    new_episode_rows: list[dict[str, Any]] = []
    new_stats_rows: list[dict[str, Any]] = []
    source_map_rows: list[dict[str, Any]] = []
    global_frame_start = 0

    total = len(selected_source_episode_ids)
    for new_episode_index, source_episode_id in enumerate(selected_source_episode_ids):
        source_episode = source_episodes[source_episode_id]
        length = int(source_episode["length"])
        task_index = task_to_new_index[source_episode["tasks"][0]]

        src_rel = data_path_for_episode(source_info, source_episode_id)
        dst_rel = data_path_for_episode(new_info, new_episode_index)
        src_path = source_root / src_rel
        dst_path = tmp_root / dst_rel
        if not src_path.is_file():
            raise FileNotFoundError(src_path)

        table = pq.read_table(src_path)
        if table.num_rows != length:
            raise ValueError(f"{src_path} has {table.num_rows} rows, expected {length}.")
        table = rewrite_episode_table(
            table,
            episode_index=new_episode_index,
            global_frame_start=global_frame_start,
            task_index=task_index,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_path, compression="snappy")

        new_episode_rows.append(
            {
                "episode_index": new_episode_index,
                "tasks": source_episode["tasks"],
                "length": length,
            }
        )
        new_stats_rows.append(
            {
                "episode_index": new_episode_index,
                "stats": update_index_stats(
                    source_stats[source_episode_id],
                    episode_index=new_episode_index,
                    global_frame_start=global_frame_start,
                    length=length,
                    task_index=task_index,
                ),
            }
        )
        source_map_rows.append(
            {
                "episode_index": new_episode_index,
                "source_episode_index": source_episode_id,
                "scene": scene_by_source_episode.get(source_episode_id, "unknown"),
            }
        )
        global_frame_start += length

        if progress_every and ((new_episode_index + 1) % progress_every == 0 or new_episode_index + 1 == total):
            elapsed = time.time() - started_at
            print(
                f"[{output_repo_id}] wrote {new_episode_index + 1}/{total} episodes "
                f"({global_frame_start}/{total_frames} frames) in {elapsed / 60:.1f} min",
                flush=True,
            )

    write_jsonl(tmp_root / "meta/episodes.jsonl", new_episode_rows)
    write_jsonl(tmp_root / "meta/episodes_stats.jsonl", new_stats_rows)
    write_jsonl(tmp_root / "meta/source_episode_map.jsonl", source_map_rows)

    manifest = {
        "output_repo_id": output_repo_id,
        "output_root": str(output_root),
        "source_root": str(source_root),
        "source_total_episodes": source_info["total_episodes"],
        "selected_source_episode_count": len(selected_source_episode_ids),
        "selected_frame_count": total_frames,
        "scene_counts": dict(Counter(scene_by_source_episode.get(ep, "unknown") for ep in selected_source_episode_ids)),
        "created_at_unix": int(time.time()),
    }
    write_json(tmp_root / "meta/split_manifest.json", manifest)

    if global_frame_start != total_frames:
        raise RuntimeError(f"Frame count mismatch: wrote {global_frame_start}, expected {total_frames}.")
    commit_output_root(tmp_root, output_root)


def load_official_train_scene_labels(
    *,
    source_root: Path,
    metadata_root: Path,
    strict_text: bool = True,
) -> tuple[list[int], list[int], dict[int, str], dict[str, tuple[int, int]]]:
    source_episodes = read_jsonl(source_root / "meta/episodes.jsonl")
    train_ann_path = metadata_root / "training/lang_annotations/auto_lang_ann.npy"
    val_ann_path = metadata_root / "validation/lang_annotations/auto_lang_ann.npy"
    scene_info_path = metadata_root / "training/scene_info.npy"

    train_ann = np.load(train_ann_path, allow_pickle=True).item()
    val_ann = np.load(val_ann_path, allow_pickle=True).item()
    scene_info_raw = np.load(scene_info_path, allow_pickle=True).item()
    scene_ranges = {
        scene_name.replace("calvin_scene_", ""): (int(bounds[0]), int(bounds[1]))
        for scene_name, bounds in scene_info_raw.items()
    }

    train_intervals = train_ann["info"]["indx"]
    train_tasks = train_ann["language"]["task"]
    train_texts = train_ann["language"]["ann"]
    train_count = len(train_intervals)
    val_count = len(val_ann["info"]["indx"])

    if len(source_episodes) != train_count + val_count:
        raise ValueError(
            f"Source episode count {len(source_episodes)} does not match official "
            f"train+validation count {train_count}+{val_count}."
        )

    abc_episode_ids: list[int] = []
    a_episode_ids: list[int] = []
    scene_by_source_episode: dict[int, str] = {}

    for episode_index, ((start, end), task, text) in enumerate(
        zip(train_intervals, train_tasks, train_texts, strict=True)
    ):
        start = int(start)
        end = int(end)
        length = end - start + 1
        source_episode = source_episodes[episode_index]
        if source_episode["episode_index"] != episode_index:
            raise ValueError(f"Unexpected source episode index at row {episode_index}: {source_episode}")
        if int(source_episode["length"]) != length:
            raise ValueError(
                f"Length mismatch for episode {episode_index}: "
                f"source={source_episode['length']} official={length}"
            )
        official_task_text = f"{task}: {text}"
        if strict_text and source_episode["tasks"][0] != official_task_text:
            raise ValueError(
                f"Task text mismatch for episode {episode_index}: "
                f"source={source_episode['tasks'][0]!r} official={official_task_text!r}"
            )

        matched_scenes = [
            scene_name
            for scene_name, (scene_start, scene_end) in scene_ranges.items()
            if scene_start <= start and end <= scene_end
        ]
        if len(matched_scenes) != 1:
            raise ValueError(f"Episode {episode_index} interval {(start, end)} matched scenes {matched_scenes}.")
        scene = matched_scenes[0]
        scene_by_source_episode[episode_index] = scene
        abc_episode_ids.append(episode_index)
        if scene == "A":
            a_episode_ids.append(episode_index)

    return abc_episode_ids, a_episode_ids, scene_by_source_episode, scene_ranges


def ensure_output_under_data(path: Path, allow_outside_data: bool) -> None:
    if allow_outside_data:
        return
    resolved = path.resolve()
    data_root = Path("/data/yzj").resolve()
    if data_root not in [resolved, *resolved.parents]:
        raise ValueError(f"Refusing to write outside /data/yzj: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--abc-output-root", type=Path, default=DEFAULT_ABC_OUTPUT_ROOT)
    parser.add_argument("--a-output-root", type=Path, default=DEFAULT_A_OUTPUT_ROOT)
    parser.add_argument("--abc-repo-id", default=DEFAULT_ABC_REPO_ID)
    parser.add_argument("--a-repo-id", default=DEFAULT_A_REPO_ID)
    parser.add_argument("--splits", nargs="+", choices=["abc", "a"], default=["abc", "a"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit episodes per split for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--no-strict-text", action="store_true")
    parser.add_argument("--allow-output-outside-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_under_data(args.abc_output_root, args.allow_output_outside_data)
    ensure_output_under_data(args.a_output_root, args.allow_output_outside_data)

    abc_episode_ids, a_episode_ids, scene_by_source_episode, scene_ranges = load_official_train_scene_labels(
        source_root=args.source_root,
        metadata_root=args.metadata_root,
        strict_text=not args.no_strict_text,
    )
    split_specs = {
        "abc": (args.abc_repo_id, args.abc_output_root, abc_episode_ids),
        "a": (args.a_repo_id, args.a_output_root, a_episode_ids),
    }

    print(f"source_root={args.source_root}", flush=True)
    print(f"metadata_root={args.metadata_root}", flush=True)
    print(f"official_scene_ranges={scene_ranges}", flush=True)
    print(f"abc_episodes={len(abc_episode_ids)}", flush=True)
    print(f"a_episodes={len(a_episode_ids)}", flush=True)

    if args.dry_run:
        for split_name in args.splits:
            repo_id, output_root, episode_ids = split_specs[split_name]
            if args.limit is not None:
                episode_ids = episode_ids[: args.limit]
            scene_counts = Counter(scene_by_source_episode.get(ep, "unknown") for ep in episode_ids)
            print(
                f"dry_run split={split_name} repo_id={repo_id} output_root={output_root} "
                f"episodes={len(episode_ids)} scene_counts={dict(scene_counts)}",
                flush=True,
            )
        return

    for split_name in args.splits:
        repo_id, output_root, episode_ids = split_specs[split_name]
        if args.limit is not None:
            episode_ids = episode_ids[: args.limit]
        print(f"writing split={split_name} repo_id={repo_id} output_root={output_root}", flush=True)
        write_subset_dataset(
            source_root=args.source_root,
            output_root=output_root,
            output_repo_id=repo_id,
            selected_source_episode_ids=episode_ids,
            scene_by_source_episode=scene_by_source_episode,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
        print(f"finished split={split_name} output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
