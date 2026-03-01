import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import robocasa.utils.lerobot_utils as LU


def main(args):
    dataset_path = Path(args.dataset)
    assert dataset_path.exists(), f"Dataset path does not exist: {dataset_path}"

    print("\n=== Loading LeRobot Dataset ===")
    ds = LeRobotDataset(repo_id="robocasa365", root=str(dataset_path))

    # --------------------------
    # Environment metadata
    # --------------------------
    print("\n=== Environment Metadata ===")
    env_meta = LU.get_env_metadata(dataset_path)
    print(json.dumps(env_meta, indent=4))

    # --------------------------
    # Dataset metadata
    # --------------------------
    print("\n=== Dataset Metadata ===")
    print(f"Total episodes: {len(ds.meta.episodes)}")

    print("\nAvailable feature keys:")
    for k in ds.meta.features:
        print(f"  - {k}: {ds.meta.features[k]}")

    # --------------------------
    # Inspect first episode
    # --------------------------
    print("\n=== Inspecting First Episode ===")
    ep_idx = 0
    start = int(ds.episode_data_index["from"][ep_idx])
    sample = ds[start]

    print("\nSample keys:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")

    # --------------------------
    # Inspect state vector specifically
    # --------------------------
    if "observation.state" in sample:
        state = sample["observation.state"]
        print("\n=== observation.state ===")
        print(f"Shape: {state.shape}")
        print(f"Dtype: {state.dtype}")
        print(f"First 5 values: {state[:5]}")
    else:
        print("\nNo 'observation.state' key found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to lerobot dataset directory",
    )
    args = parser.parse_args()
    main(args)
