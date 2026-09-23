from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

def future_max(values: np.ndarray) -> np.ndarray:
    """Compute the inclusive future maximum within one event segment.

    For a segment g[0:L],

        future_max_g[t] = max(g[t], ..., g[L-1]).
    """
    if values.ndim != 1:
        raise ValueError(f"Expected 1-D safety values, got shape {values.shape}.")

    return np.maximum.accumulate(values[::-1])[::-1]

def get_segment_boundaries(push_events: np.ndarray) -> list[tuple[int, int, bool]]:
    """Return [start, end) boundaries separated by push events.

    A push marker at index k means that state k is the first recorded
    state of the new post-push segment.

    """
    push_events = np.asarray(push_events).reshape(-1).astype(bool)

    length = len(push_events)
    push_indices = np.flatnonzero(push_events)

    boundaries = np.concatenate((np.array([0], dtype=np.int64), push_indices.astype(np.int64), np.array([length], dtype=np.int64)))

    segments = []
    for segment_idx in range(len(boundaries)-1):
        start = int(boundaries[segment_idx])
        end = int(boundaries[segment_idx + 1])

        starts_after_push = segment_idx > 0

        segments.append((start, end, starts_after_push))

    return segments

def split_episode_keys(
    episode_keys: list[str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split raw episodes before temporal segmentation."""
    total_ratio = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}.")

    rng = np.random.default_rng(seed)
    episode_keys = np.asarray(sorted(episode_keys), dtype=object)
    episode_keys = episode_keys[rng.permutation(len(episode_keys))]

    num_episodes = len(episode_keys)
    num_test = int(num_episodes * test_ratio)
    num_validation = int(num_episodes * validation_ratio)
    num_train = num_episodes - num_validation - num_test
    validation_end = num_train + num_validation

    return {
        "train": episode_keys[:num_train].tolist(),
        "validation": episode_keys[num_train:validation_end].tolist(),
        "test": episode_keys[validation_end:].tolist(),
    }

def process_episode(episode_key: str, episode_group: h5py.Group) -> list[dict]:
    """Split one raw episode by push events and compute segment-wise future maximum."""
    states = np.asarray(episode_group["safety_states"], dtype=np.float32)
    g_values = np.asarray(episode_group["safety_values"], dtype=np.float32)
    push_events = np.asarray(episode_group["push_events"], dtype=bool)
    terminated = np.asarray(episode_group["terminated"], dtype=bool)
    truncated = np.asarray(episode_group["truncated"], dtype=bool)

    if not np.all(np.isfinite(states)):
        raise ValueError(f"{episode_key}: safety_states contains NaN or Inf.")
    if not np.all(np.isfinite(g_values)):
        raise ValueError(f"{episode_key}: safety_values contains NaN or Inf.")

    boundaries = get_segment_boundaries(push_events)
    segments = []

    for segment_index, (start, end, starts_after_push) in enumerate(boundaries):
        if end - start < 2:
            continue

        segment_states = states[start:end]
        segment_g = g_values[start:end]
        segment_terminated = terminated[start:end]
        segment_truncated = truncated[start:end]

        segments.append({
            "states": segment_states,
            "g_values": segment_g,
            "future_max_g": future_max(segment_g),
            "terminated": segment_terminated,
            "truncated": segment_truncated,
            "source_episode": episode_key,
            "segment_index": segment_index,
            "source_start_index": start,
            "source_end_index": end,
            "starts_after_push": starts_after_push,
        })

    return segments

def write_segment(
    split_group: h5py.Group,
    segment_name: str,
    segment: dict,
    source_env_id: int,
    compression: str | None = "lzf",
) -> None:
    """Write one processed trajectory segment."""
    group = split_group.create_group(segment_name)

    group.create_dataset("states", data=segment["states"], compression=compression)
    group.create_dataset("g_values", data=segment["g_values"], compression=compression)
    group.create_dataset("future_max_g", data=segment["future_max_g"], compression=compression)
    group.create_dataset("terminated", data=segment["terminated"], compression=compression)
    group.create_dataset("truncated", data=segment["truncated"], compression=compression)

    group.attrs["source_episode"] = segment["source_episode"]
    group.attrs["source_env_id"] = source_env_id
    group.attrs["segment_index"] = segment["segment_index"]
    group.attrs["source_start_index"] = segment["source_start_index"]
    group.attrs["source_end_index"] = segment["source_end_index"]
    group.attrs["starts_after_push"] = segment["starts_after_push"]
    group.attrs["length"] = len(segment["g_values"])
    group.attrs["ends_with_termination"] = bool(segment["terminated"][-1])
    group.attrs["ends_with_truncation"] = bool(segment["truncated"][-1])

def update_statistics(stats: dict, segment: dict) -> None:
    length = len(segment["g_values"])
    future_max_for_training = segment["future_max_g"][:-1]
    risk = future_max_for_training > 0

    stats["segments"] += 1
    stats["states"] += length
    stats["transitions"] += length - 1
    stats["risk_states"] += int(np.sum(risk))
    stats["safe_states"] += int(np.sum(~risk))
    stats["terminated_segments"] += int(segment["terminated"][-1])
    stats["truncated_segments"] += int(segment["truncated"][-1])
    stats["push_segments"] += int(segment["starts_after_push"])

def print_statistics(statistics: dict[str, dict], output_path: str) -> None:
    print("=" * 80)
    print("SAFETY DATASET PREPROCESSING")
    print("=" * 80)

    for split_name, stats in statistics.items():
        transitions = stats["transitions"]
        risk_ratio = 100.0 * stats["risk_states"] / transitions if transitions > 0 else 0.0

        print(f"[{split_name.upper()}]")
        print(f"Episodes              : {stats['episodes']}")
        print(f"Segments              : {stats['segments']}")
        print(f"Push-start segments   : {stats['push_segments']}")
        print(f"Terminated segments   : {stats['terminated_segments']}")
        print(f"Truncated segments    : {stats['truncated_segments']}")
        print(f"Short segments skipped: {stats['short_segments_skipped']}")
        print(f"States                : {stats['states']}")
        print(f"Transitions           : {stats['transitions']}")
        print(f"Future-risk states    : {stats['risk_states']}")
        print(f"Future-safe states    : {stats['safe_states']}")
        print(f"Future-risk ratio     : {risk_ratio:.2f}%")
        print("-" * 80)

    print(f"[INFO] Processed dataset: {output_path}")

def preprocess_dataset(
    raw_dataset_path: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    compression: str | None = "lzf",
) -> None:
    raw_dataset_path = os.path.abspath(raw_dataset_path)
    output_path = os.path.join(os.path.dirname(raw_dataset_path), 'data_processed.hbf5')

    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Raw dataset not found: {raw_dataset_path}")
    if os.path.exists(output_path):
        raise FileExistsError(f"Processed dataset already exists: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(raw_dataset_path, "r") as raw_file:
        raw_data = raw_file["data"]
        episode_keys = sorted(raw_data.keys())
        safety_state_dim = int(raw_file.attrs["safety_state_dim"])

        # Split raw data. (units: episode)
        splits = split_episode_keys(
            episode_keys=episode_keys,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        with h5py.File(output_path, "w") as out_file:
            # metadata oif processed dataset
            out_file.attrs["source_dataset"] = raw_dataset_path
            out_file.attrs["safety_state_dim"] = safety_state_dim
            out_file.attrs["split_seed"] = seed
            out_file.attrs["train_ratio"] = train_ratio
            out_file.attrs["validation_ratio"] = validation_ratio
            out_file.attrs["test_ratio"] = test_ratio

            statistics = {}
            global_segment_index = 0

            for split_name, split_keys in splits.items():
                # [train, validation, test]
                split_group = out_file.create_group(split_name)
                stats = {
                            "episodes": 0,
                            "segments": 0,
                            "states": 0,
                            "transitions": 0,
                            "risk_states": 0,
                            "safe_states": 0,
                            "terminated_segments": 0,
                            "truncated_segments": 0,
                            "push_segments": 0,
                            "short_segments_skipped": 0,
                        }
                stats["episodes"] = len(split_keys)

                # find segments in each episode
                for episode_key in split_keys:
                    episode_group = raw_data[episode_key]
                    source_env_id = int(episode_group.attrs.get("env_id", -1))

                    # skip short horizon boundary (unvalid)
                    push_events = np.asarray(episode_group["push_events"], dtype=bool)
                    boundaries = get_segment_boundaries(push_events)
                    stats["short_segments_skipped"] += sum((end - start) < 2 for start, end, _ in boundaries)

                    # get finite-horizon boundary using push event marker
                    segments = process_episode(episode_key, episode_group)
                    for segment in segments:
                        segment_name = f"segment_{global_segment_index:08d}"
                        write_segment(
                            split_group=split_group,
                            segment_name=segment_name,
                            segment=segment,
                            source_env_id=source_env_id,
                            compression=compression,
                        )
                        update_statistics(stats, segment)
                        global_segment_index += 1

                statistics[split_name] = stats
                for key, value in stats.items():
                    split_group.attrs[key] = value

            out_file.attrs["num_segments"] = global_segment_index

    print_statistics(statistics, output_path)

if __name__ == "__main__":
    """
    Data Architecture: 

        data_processed.hdf5
        │
        ├── attrs (metadata)
        │   ├── source_dataset
        │   ├── safety_state_dim
        │   ├── split_seed
        │   ├── train_ratio
        │   ├── validation_ratio
        │   ├── test_ratio
        │   └── num_segments
        │
        ├── train/ (segment-wise data for training)
        │   ├── segment_00000000/
        │   │   ├── states
        │   │   ├── g_values
        │   │   ├── future_max_g
        │   │   ├── terminated
        │   │   ├── truncated
        │   │   └── attrs ...
        │   ├── segment_00000001/
        │   └── ...
        │
        ├── validation/ (segment-wise data for validation)
        │   └── ...
        │
        └── test/ (segment-wise data for test)
            └── ...
    
    """
    parser = argparse.ArgumentParser(description="Preprocess offline safety-value rollout data.")
    parser.add_argument("--raw_dataset", type=str, required=True, help="Path to data_raw.hdf5.")
    parser.add_argument("--seed", type=int, default=42, help="Episode-level split seed.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    raw_dataset_path = os.path.abspath(args.raw_dataset)

    preprocess_dataset(
        raw_dataset_path=raw_dataset_path,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )