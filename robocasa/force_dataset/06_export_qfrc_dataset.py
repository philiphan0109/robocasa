#!/usr/bin/env python3
"""
Export qfrc-based force features into a new LeRobot dataset root.

This script is non-destructive:
- reads from --dataset (source root),
- writes to --output-root (new root),
- never edits source files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import robosuite
except ImportError as e:
    raise SystemExit(
        "robosuite is required for this script. Activate the RoboCasa environment first."
    ) from e

try:
    import robocasa.utils.lerobot_utils as LU
except ImportError as e:
    raise SystemExit(
        "robocasa is not importable. Set PYTHONPATH (e.g. PYTHONPATH=/home/phan07/robocasa)."
    ) from e


KEY_QFRC_ARM = "observation.force.qfrc_constraint_arm"
KEY_QFRC_ARM_L2 = "observation.force.qfrc_constraint_arm_l2"
KEY_VALID = "diagnostic.force_valid"
KEY_DIV = "diagnostic.force_divergence_l2"
SIDECAR_REL_DIR = Path("extras") / "force_qfrc_full"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def parse_episodes_spec(spec: str, total_episodes: int) -> list[int]:
    s = spec.strip().lower()
    if s == "all":
        return list(range(total_episodes))

    out: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if b < a:
                a, b = b, a
            for ep in range(a, b + 1):
                if ep < 0 or ep >= total_episodes:
                    raise ValueError(
                        f"Episode {ep} out of bounds [0, {total_episodes - 1}]"
                    )
                out.add(ep)
        else:
            ep = int(part)
            if ep < 0 or ep >= total_episodes:
                raise ValueError(
                    f"Episode {ep} out of bounds [0, {total_episodes - 1}]"
                )
            out.add(ep)
    return sorted(out)


def get_total_episodes(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    info = load_json(info_path)
    total = info.get("total_episodes")
    if not isinstance(total, int):
        raise ValueError(f"Invalid total_episodes in {info_path}")
    return total


def copy_dataset_root(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    if src_root.resolve() == dst_root.resolve():
        raise ValueError("output-root cannot be the same as source dataset root")

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root exists: {dst_root}. Pass --overwrite-output-root to replace it."
            )
        shutil.rmtree(dst_root)

    shutil.copytree(src_root, dst_root)


def find_episode_parquet(dataset_root: Path, episode: int) -> Path:
    matches = list(dataset_root.glob(f"data/*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode_{episode:06d} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple parquet files for episode {episode}: {matches}")
    return matches[0]


def reset_to(env, state):
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}

        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)

        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()

    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def make_env(dataset_root: Path):
    env_meta = LU.get_env_metadata(dataset_root)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def select_arm_dofs(model) -> tuple[list[str], list[int]]:
    joint_names = [model.joint_id2name(i) for i in range(model.njnt)]
    dof_jntid = np.asarray(model.dof_jntid, dtype=np.int64)
    target_joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
    selected_joint_names = [n for n in target_joint_names if n in joint_names]
    if len(selected_joint_names) != 7:
        raise RuntimeError(
            "Expected 7 arm joints robot0_joint1..robot0_joint7, "
            f"found {selected_joint_names}"
        )

    selected_dof_indices: list[int] = []
    for jname in selected_joint_names:
        jid = model.joint_name2id(jname)
        dof_idx = np.where(dof_jntid == jid)[0].tolist()
        selected_dof_indices.extend(dof_idx)
    selected_dof_indices = sorted(set(selected_dof_indices))

    if len(selected_dof_indices) != 7:
        raise RuntimeError(
            f"Expected 7 arm DoFs from robot0 joints, got {len(selected_dof_indices)}"
        )
    return selected_joint_names, selected_dof_indices


def collect_episode_raw_forces(
    env,
    dataset_root: Path,
    episode: int,
    replay_mode: str,
) -> dict[str, Any]:
    if replay_mode != "action_shifted":
        raise ValueError(
            f"Unsupported replay-mode: {replay_mode}. Only action_shifted is supported."
        )

    states = LU.get_episode_states(dataset_root, episode)
    actions = LU.get_episode_actions(dataset_root, episode, abs_actions=False)
    if len(states) != len(actions):
        raise RuntimeError(
            f"states/actions mismatch for episode {episode}: "
            f"{len(states)} vs {len(actions)}"
        )

    initial_state = {
        "states": states[0],
        "model": LU.get_episode_model_xml(dataset_root, episode),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset_root, episode)),
    }
    reset_to(env, initial_state)

    model = env.sim.model
    arm_joint_names, arm_dof_indices = select_arm_dofs(model)

    qfrc = []
    divergence = np.full((len(states),), np.nan, dtype=np.float64)
    for t in range(len(states)):
        reset_to(env, {"states": states[t]})
        env.step(actions[t])  # restore-state -> play-action -> sample-force

        q = np.asarray(env.sim.data.qfrc_constraint, dtype=np.float64).copy()
        qfrc.append(q)

        if t < len(states) - 1:
            s_next = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            divergence[t] = float(np.linalg.norm(s_next - states[t + 1]))

    qfrc = np.stack(qfrc, axis=0)
    qfrc_arm = qfrc[:, arm_dof_indices]
    qfrc_arm_l2 = np.linalg.norm(qfrc_arm, axis=1)

    return {
        "episode": int(episode),
        "T": int(len(states)),
        "nv": int(model.nv),
        "qfrc_raw": qfrc,
        "qfrc_arm_raw": qfrc_arm,
        "qfrc_arm_l2_raw": qfrc_arm_l2,
        "divergence_raw": divergence,
        "arm_joint_names": arm_joint_names,
        "arm_dof_indices": arm_dof_indices,
        "dof_joint_ids": np.asarray(model.dof_jntid, dtype=np.int64).tolist(),
    }


def shift_causal(arr: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    x = np.asarray(arr)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = fill_value
    out[1:] = x[:-1]
    return out


def apply_alignment(raw_pack: dict[str, Any], div_thresh: float) -> dict[str, Any]:
    t_horizon = raw_pack["T"]
    q_shift = shift_causal(raw_pack["qfrc_raw"], fill_value=np.nan)
    arm_shift = shift_causal(raw_pack["qfrc_arm_raw"], fill_value=np.nan)
    arm_l2_shift = shift_causal(raw_pack["qfrc_arm_l2_raw"], fill_value=np.nan)
    div_shift = shift_causal(raw_pack["divergence_raw"], fill_value=np.nan)

    valid = np.zeros((t_horizon,), dtype=bool)
    div_raw = np.asarray(raw_pack["divergence_raw"], dtype=np.float64)
    valid[1:] = np.isfinite(div_raw[:-1]) & (div_raw[:-1] <= float(div_thresh))

    return {
        "episode": raw_pack["episode"],
        "T": t_horizon,
        "nv": raw_pack["nv"],
        "qfrc": q_shift,
        "qfrc_arm": arm_shift,
        "qfrc_arm_l2": arm_l2_shift,
        "divergence": div_shift,
        "valid": valid,
        "arm_joint_names": raw_pack["arm_joint_names"],
        "arm_dof_indices": raw_pack["arm_dof_indices"],
        "dof_joint_ids": raw_pack["dof_joint_ids"],
    }


def write_episode_parquet(parquet_path: Path, pack: dict[str, Any]) -> None:
    df = pd.read_parquet(parquet_path)
    if len(df) != pack["T"]:
        raise RuntimeError(
            f"Frame mismatch for {parquet_path.name}: parquet has {len(df)}, "
            f"force pack has {pack['T']}"
        )

    df[KEY_QFRC_ARM] = [x for x in pack["qfrc_arm"]]
    df[KEY_QFRC_ARM_L2] = [
        np.asarray([x], dtype=np.float64) for x in pack["qfrc_arm_l2"]
    ]
    df[KEY_VALID] = [np.asarray([x], dtype=bool) for x in pack["valid"]]
    df[KEY_DIV] = [np.asarray([x], dtype=np.float64) for x in pack["divergence"]]

    df.to_parquet(parquet_path, index=False)


def write_full_qfrc_sidecar(dataset_root: Path, pack: dict[str, Any]) -> Path:
    sidecar_dir = dataset_root / SIDECAR_REL_DIR
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / f"episode_{pack['episode']:06d}.npz"
    np.savez_compressed(
        sidecar_path,
        qfrc_constraint=np.asarray(pack["qfrc"], dtype=np.float64),
        nv=np.asarray([int(pack["nv"])], dtype=np.int64),
        arm_dof_indices=np.asarray(pack["arm_dof_indices"], dtype=np.int64),
        dof_joint_ids=np.asarray(pack["dof_joint_ids"], dtype=np.int64),
    )
    return sidecar_path


def update_info_json(meta_dir: Path) -> None:
    info_path = meta_dir / "info.json"
    info = load_json(info_path)
    features = info.setdefault("features", {})

    features[KEY_QFRC_ARM] = {"dtype": "float64", "shape": [7]}
    features[KEY_QFRC_ARM_L2] = {"dtype": "float64", "shape": [1]}
    features[KEY_VALID] = {"dtype": "bool", "shape": [1]}
    features[KEY_DIV] = {"dtype": "float64", "shape": [1]}

    dump_json(info_path, info)


def update_modality_json(meta_dir: Path) -> None:
    mod_path = meta_dir / "modality.json"
    modality = load_json(mod_path)

    force = modality.setdefault("force", {})
    force["qfrc_constraint_arm"] = {"original_key": KEY_QFRC_ARM}
    force["qfrc_constraint_arm_l2"] = {"original_key": KEY_QFRC_ARM_L2}

    diagnostic = modality.setdefault("diagnostic", {})
    diagnostic["force_valid"] = {"original_key": KEY_VALID}
    diagnostic["force_divergence_l2"] = {"original_key": KEY_DIV}

    dump_json(mod_path, modality)


def stat_dict(mat: np.ndarray) -> dict[str, list[float]]:
    # Force keys contain padded NaNs at row 0 by design; use nan-safe reducers.
    return {
        "mean": np.nanmean(mat, axis=0).tolist(),
        "std": np.nanstd(mat, axis=0).tolist(),
        "min": np.nanmin(mat, axis=0).tolist(),
        "max": np.nanmax(mat, axis=0).tolist(),
        "q01": np.nanquantile(mat, 0.01, axis=0).tolist(),
        "q99": np.nanquantile(mat, 0.99, axis=0).tolist(),
    }


def recompute_stats(dataset_root: Path, episodes: list[int]) -> None:
    keys = [KEY_QFRC_ARM, KEY_QFRC_ARM_L2, KEY_VALID, KEY_DIV]
    accum: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    for ep in episodes:
        p = find_episode_parquet(dataset_root, ep)
        df = pd.read_parquet(p)
        for key in keys:
            if key not in df.columns:
                raise RuntimeError(f"Missing {key} in {p}")
            rows = []
            for value in df[key].tolist():
                row = np.asarray(value, dtype=np.float64).reshape(-1)
                rows.append(row)
            accum[key].append(np.vstack(rows))

    stats_path = dataset_root / "meta" / "stats.json"
    stats = load_json(stats_path)
    for key, parts in accum.items():
        mat = np.vstack(parts)
        stats[key] = stat_dict(mat)
    dump_json(stats_path, stats)


def write_force_schema(
    meta_dir: Path,
    replay_mode: str,
    div_thresh: float,
    arm_joint_names: list[str],
    episodes: list[int],
    per_episode_nv: dict[str, int],
    per_episode_sidecar: dict[str, str],
) -> None:
    schema = {
        "force_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "replay_mode": replay_mode,
        "alignment": {
            "type": "action_shifted_to_observation_timeline",
            "description": (
                "Recorded signal at row t is sampled from replay step (t-1); "
                "row 0 is padded with NaN/False."
            ),
            "row0_policy": {
                KEY_QFRC_ARM: "NaN",
                KEY_QFRC_ARM_L2: "NaN",
                KEY_DIV: "NaN",
                KEY_VALID: False,
            },
            "divergence_threshold": float(div_thresh),
        },
        "keys": {
            KEY_QFRC_ARM: {"dtype": "float64", "shape": [7]},
            KEY_QFRC_ARM_L2: {"dtype": "float64", "shape": [1]},
            KEY_VALID: {"dtype": "bool", "shape": [1]},
            KEY_DIV: {"dtype": "float64", "shape": [1]},
        },
        "full_qfrc_sidecar": {
            "relative_dir": str(SIDECAR_REL_DIR),
            "file_pattern": "episode_<episode_id>.npz",
            "array_key": "qfrc_constraint",
            "shape": "[T, nv_episode]",
            "notes": "Per-episode full qfrc stored here because nv is demo-dependent.",
        },
        "arm_joint_names": arm_joint_names,
        "episodes": episodes,
        "per_episode_nv": per_episode_nv,
        "per_episode_sidecar": per_episode_sidecar,
    }
    dump_json(meta_dir / "force_schema.json", schema)


def default_report_path(output_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("artifacts")
        / "qfrc_export_reports"
        / f"{output_root.name}_export_report_{ts}.json"
    )


def progress_log(msg: str, pbar: Any | None) -> None:
    if pbar is not None and hasattr(pbar, "write"):
        pbar.write(msg)
    else:
        print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Source lerobot root"
    )
    parser.add_argument(
        "--output-root", type=str, required=True, help="Output augmented lerobot root"
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode spec: all | 0-103 | 0,1,2",
    )
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="action_shifted",
        choices=["action_shifted"],
        help="Replay mode used for extraction and causal alignment",
    )
    parser.add_argument(
        "--div-thresh",
        type=float,
        default=0.1,
        help="Divergence threshold for diagnostic.force_valid",
    )
    parser.add_argument(
        "--overwrite-output-root",
        action="store_true",
        help="Overwrite output root if it exists",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Reserved. Current implementation is deterministic single-worker only.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip recomputing stats.json entries for new force keys",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional export report path; default under artifacts/qfrc_export_reports/",
    )
    args = parser.parse_args()

    if args.num_workers != 1:
        raise SystemExit("Only --num-workers 1 is supported in this v1 implementation.")

    src_root = Path(args.dataset).resolve()
    dst_root = Path(args.output_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src_root}")
    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Invalid dataset root (missing meta/info.json): {src_root}"
        )

    total_episodes = get_total_episodes(src_root)
    episodes = parse_episodes_spec(args.episodes, total_episodes)
    report_path = (
        Path(args.report_json) if args.report_json else default_report_path(dst_root)
    )

    start_time = time.time()
    copy_dataset_root(src_root, dst_root, overwrite=args.overwrite_output_root)

    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(src_root),
        "output_dataset": str(dst_root),
        "episodes_requested": episodes,
        "replay_mode": args.replay_mode,
        "div_thresh": float(args.div_thresh),
        "processed": [],
        "failed": [],
        "metadata": {},
    }

    env = make_env(src_root)
    pbar: Any | None = None
    ep_iter: Any = episodes
    if tqdm is not None:
        pbar = tqdm(
            episodes,
            total=len(episodes),
            desc="Export episodes",
            unit="ep",
            dynamic_ncols=True,
        )
        ep_iter = pbar
    else:
        print(f"Processing {len(episodes)} episodes (install tqdm for progress bar).")

    meta_ref: dict[str, Any] | None = None
    per_episode_nv: dict[str, int] = {}
    per_episode_sidecar: dict[str, str] = {}
    try:
        for ep in ep_iter:
            ep_t0 = time.time()
            try:
                raw = collect_episode_raw_forces(
                    env=env,
                    dataset_root=src_root,
                    episode=ep,
                    replay_mode=args.replay_mode,
                )
                aligned = apply_alignment(raw, div_thresh=args.div_thresh)
                out_parquet = find_episode_parquet(dst_root, ep)
                write_episode_parquet(out_parquet, aligned)
                sidecar_path = write_full_qfrc_sidecar(dst_root, aligned)

                if meta_ref is None:
                    meta_ref = {
                        "arm_joint_names": aligned["arm_joint_names"],
                    }
                per_episode_nv[f"{ep:06d}"] = int(aligned["nv"])
                per_episode_sidecar[f"{ep:06d}"] = str(
                    sidecar_path.relative_to(dst_root)
                )

                div = np.asarray(aligned["divergence"], dtype=np.float64)
                valid = np.asarray(aligned["valid"], dtype=bool)
                finite_div = div[np.isfinite(div)]
                report["processed"].append(
                    {
                        "episode": ep,
                        "frames": int(aligned["T"]),
                        "nv": int(aligned["nv"]),
                        "valid_ratio": float(np.mean(valid)),
                        "div_mean": float(np.mean(finite_div))
                        if finite_div.size
                        else None,
                        "div_p95": float(np.quantile(finite_div, 0.95))
                        if finite_div.size
                        else None,
                        "elapsed_sec": round(time.time() - ep_t0, 4),
                    }
                )
                progress_log(f"[ok] episode_{ep:06d}", pbar)
            except Exception as e:
                report["failed"].append({"episode": ep, "error": str(e)})
                progress_log(f"[fail] episode_{ep:06d}: {e}", pbar)
            if pbar is not None and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    ok=len(report["processed"]),
                    fail=len(report["failed"]),
                    last=f"{ep:06d}",
                )
    finally:
        if pbar is not None:
            pbar.close()
        env.close()

    if meta_ref is not None:
        update_info_json(dst_root / "meta")
        update_modality_json(dst_root / "meta")
        write_force_schema(
            meta_dir=dst_root / "meta",
            replay_mode=args.replay_mode,
            div_thresh=args.div_thresh,
            arm_joint_names=meta_ref["arm_joint_names"],
            episodes=episodes,
            per_episode_nv=per_episode_nv,
            per_episode_sidecar=per_episode_sidecar,
        )
    if not args.skip_stats and report["processed"]:
        recompute_stats(dst_root, [x["episode"] for x in report["processed"]])

    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_sec"] = round(time.time() - start_time, 4)
    report["metadata"] = {
        "processed_count": len(report["processed"]),
        "failed_count": len(report["failed"]),
        "skip_stats": bool(args.skip_stats),
    }
    dump_json(report_path, report)

    print(f"Export report: {report_path}")
    print(f"Output dataset: {dst_root}")
    if report["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
