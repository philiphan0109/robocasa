#!/usr/bin/env python3
"""
Validate qfrc-augmented LeRobot dataset exported by 06_export_qfrc_dataset.py.
"""

from __future__ import annotations

import argparse
import json
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
REQ_KEYS = [KEY_QFRC_ARM, KEY_QFRC_ARM_L2, KEY_VALID, KEY_DIV]
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
    info = load_json(dataset_root / "meta" / "info.json")
    total = info.get("total_episodes")
    if not isinstance(total, int):
        raise ValueError("meta/info.json missing valid total_episodes")
    return total


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


def shift_causal(arr: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    x = np.asarray(arr)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = fill_value
    out[1:] = x[:-1]
    return out


def collect_raw_replay(env, dataset_root: Path, episode: int) -> dict[str, np.ndarray]:
    states = LU.get_episode_states(dataset_root, episode)
    actions = LU.get_episode_actions(dataset_root, episode, abs_actions=False)
    if len(states) != len(actions):
        raise RuntimeError(
            f"states/actions mismatch for episode {episode}: {len(states)} vs {len(actions)}"
        )

    initial_state = {
        "states": states[0],
        "model": LU.get_episode_model_xml(dataset_root, episode),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset_root, episode)),
    }
    reset_to(env, initial_state)

    model = env.sim.model
    _, arm_dof_indices = select_arm_dofs(model)

    qfrc = []
    divergence = np.full((len(states),), np.nan, dtype=np.float64)
    for t in range(len(states)):
        reset_to(env, {"states": states[t]})
        env.step(actions[t])
        q = np.asarray(env.sim.data.qfrc_constraint, dtype=np.float64).copy()
        qfrc.append(q)

        if t < len(states) - 1:
            s_next = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            divergence[t] = float(np.linalg.norm(s_next - states[t + 1]))

    qfrc = np.stack(qfrc, axis=0)
    qfrc_arm = qfrc[:, arm_dof_indices]
    qfrc_arm_l2 = np.linalg.norm(qfrc_arm, axis=1)
    return {
        "qfrc_raw": qfrc,
        "qfrc_arm_raw": qfrc_arm,
        "qfrc_arm_l2_raw": qfrc_arm_l2,
        "divergence_raw": divergence,
    }


def read_force_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    arm = np.vstack(
        [np.asarray(v, dtype=np.float64).reshape(-1) for v in df[KEY_QFRC_ARM].tolist()]
    )
    arm_l2 = np.asarray(
        [
            np.asarray(v, dtype=np.float64).reshape(-1)[0]
            for v in df[KEY_QFRC_ARM_L2].tolist()
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [bool(np.asarray(v).reshape(-1)[0]) for v in df[KEY_VALID].tolist()],
        dtype=bool,
    )
    div = np.asarray(
        [np.asarray(v, dtype=np.float64).reshape(-1)[0] for v in df[KEY_DIV].tolist()],
        dtype=np.float64,
    )
    return {
        "arm": arm,
        "arm_l2": arm_l2,
        "valid": valid,
        "div": div,
    }


def validate_metadata(dataset_root: Path) -> list[str]:
    errors: list[str] = []

    info = load_json(dataset_root / "meta" / "info.json")
    features = info.get("features", {})
    exp_shapes = {
        KEY_QFRC_ARM: [7],
        KEY_QFRC_ARM_L2: [1],
        KEY_VALID: [1],
        KEY_DIV: [1],
    }
    exp_dtypes = {
        KEY_QFRC_ARM: "float64",
        KEY_QFRC_ARM_L2: "float64",
        KEY_VALID: "bool",
        KEY_DIV: "float64",
    }
    for key in REQ_KEYS:
        if key not in features:
            errors.append(f"meta/info.json missing feature {key}")
            continue
        feat = features[key]
        if feat.get("dtype") != exp_dtypes[key]:
            errors.append(
                f"meta/info.json dtype mismatch for {key}: {feat.get('dtype')} vs {exp_dtypes[key]}"
            )
        if feat.get("shape") != exp_shapes[key]:
            errors.append(
                f"meta/info.json shape mismatch for {key}: {feat.get('shape')} vs {exp_shapes[key]}"
            )

    modality = load_json(dataset_root / "meta" / "modality.json")
    force = modality.get("force", {})
    diagnostic = modality.get("diagnostic", {})
    if force.get("qfrc_constraint_arm", {}).get("original_key") != KEY_QFRC_ARM:
        errors.append("meta/modality.json missing force.qfrc_constraint_arm mapping")
    if force.get("qfrc_constraint_arm_l2", {}).get("original_key") != KEY_QFRC_ARM_L2:
        errors.append("meta/modality.json missing force.qfrc_constraint_arm_l2 mapping")
    if diagnostic.get("force_valid", {}).get("original_key") != KEY_VALID:
        errors.append("meta/modality.json missing diagnostic.force_valid mapping")
    if diagnostic.get("force_divergence_l2", {}).get("original_key") != KEY_DIV:
        errors.append(
            "meta/modality.json missing diagnostic.force_divergence_l2 mapping"
        )

    stats = load_json(dataset_root / "meta" / "stats.json")
    for key in REQ_KEYS:
        if key not in stats:
            errors.append(f"meta/stats.json missing key {key}")
            continue
        row = stats[key]
        for stat_name in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat_name not in row:
                errors.append(f"meta/stats.json missing {stat_name} for {key}")

    schema_path = dataset_root / "meta" / "force_schema.json"
    if not schema_path.exists():
        errors.append("meta/force_schema.json missing")

    return errors


def validate_sidecar(dataset_root: Path, episode: int, expected_t: int) -> list[str]:
    errs: list[str] = []
    sidecar = dataset_root / SIDECAR_REL_DIR / f"episode_{episode:06d}.npz"
    if not sidecar.exists():
        return [f"Missing full-qfrc sidecar: {sidecar}"]
    try:
        with np.load(sidecar) as z:
            if "qfrc_constraint" not in z:
                errs.append(f"sidecar missing key qfrc_constraint ({sidecar})")
            else:
                q = np.asarray(z["qfrc_constraint"], dtype=np.float64)
                if q.ndim != 2:
                    errs.append(f"sidecar qfrc_constraint ndim != 2 ({sidecar})")
                else:
                    if q.shape[0] != expected_t:
                        errs.append(
                            f"sidecar T mismatch ({sidecar}): {q.shape[0]} vs {expected_t}"
                        )
                    if q.shape[1] <= 0:
                        errs.append(f"sidecar nv invalid ({sidecar}): {q.shape[1]}")
                    if not np.all(np.isnan(q[0])):
                        errs.append(f"sidecar row0 not padded NaN ({sidecar})")
    except Exception as e:
        errs.append(f"sidecar read failed ({sidecar}): {e}")
    return errs


def get_div_threshold_from_schema(dataset_root: Path, fallback: float = 0.1) -> float:
    schema_path = dataset_root / "meta" / "force_schema.json"
    if not schema_path.exists():
        return float(fallback)
    try:
        schema = load_json(schema_path)
        return float(schema.get("alignment", {}).get("divergence_threshold", fallback))
    except Exception:
        return float(fallback)


def default_report_path(dataset_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("artifacts")
        / "qfrc_validation_reports"
        / f"{dataset_root.name}_validation_report_{ts}.json"
    )


def progress_log(msg: str, pbar: Any | None) -> None:
    if pbar is not None and hasattr(pbar, "write"):
        pbar.write(msg)
    else:
        print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Augmented dataset root"
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode spec: all | 0-103 | 0,1,2",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Return non-zero if any validation error is found",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always exit zero and rely on report contents",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Validation report path; default under artifacts/qfrc_validation_reports/",
    )
    parser.add_argument(
        "--check-alignment-replay",
        dest="check_alignment_replay",
        action="store_true",
        default=True,
        help="Replay-check causal alignment against stored values",
    )
    parser.add_argument(
        "--skip-alignment-replay",
        dest="check_alignment_replay",
        action="store_false",
        help="Skip replay-based equality checks",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for replay alignment value checks",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for replay alignment value checks",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Invalid dataset root: {dataset_root}")

    total_episodes = get_total_episodes(dataset_root)
    episodes = parse_episodes_spec(args.episodes, total_episodes)
    report_path = (
        Path(args.report_json)
        if args.report_json
        else default_report_path(dataset_root)
    )

    div_thresh = get_div_threshold_from_schema(dataset_root, fallback=0.1)
    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_root),
        "episodes": episodes,
        "strict": bool(args.strict),
        "check_alignment_replay": bool(args.check_alignment_replay),
        "div_thresh": float(div_thresh),
        "episode_results": [],
        "metadata_errors": [],
        "loader_check": {},
        "errors": [],
    }

    t0 = time.time()
    env = make_env(dataset_root) if args.check_alignment_replay else None
    pbar: Any | None = None
    ep_iter: Any = episodes
    if tqdm is not None:
        pbar = tqdm(
            episodes,
            total=len(episodes),
            desc="Validate episodes",
            unit="ep",
            dynamic_ncols=True,
        )
        ep_iter = pbar
    else:
        print(f"Validating {len(episodes)} episodes (install tqdm for progress bar).")

    try:
        for ep in ep_iter:
            ep_errs: list[str] = []
            ep_t0 = time.time()
            try:
                parquet_path = find_episode_parquet(dataset_root, ep)
                df = pd.read_parquet(parquet_path)
                for key in REQ_KEYS:
                    if key not in df.columns:
                        ep_errs.append(f"missing column {key}")

                if ep_errs:
                    report["episode_results"].append(
                        {
                            "episode": ep,
                            "ok": False,
                            "errors": ep_errs,
                            "elapsed_sec": round(time.time() - ep_t0, 4),
                        }
                    )
                    continue

                cols = read_force_columns(df)
                t_horizon = len(df)

                if cols["arm"].shape[1] != 7:
                    ep_errs.append(
                        f"{KEY_QFRC_ARM} second dim mismatch: {cols['arm'].shape[1]} vs 7"
                    )
                if cols["arm_l2"].shape[0] != t_horizon:
                    ep_errs.append(f"{KEY_QFRC_ARM_L2} length mismatch")
                if cols["valid"].shape[0] != t_horizon:
                    ep_errs.append(f"{KEY_VALID} length mismatch")
                if cols["div"].shape[0] != t_horizon:
                    ep_errs.append(f"{KEY_DIV} length mismatch")

                ep_errs.extend(validate_sidecar(dataset_root, ep, expected_t=t_horizon))

                # Row-0 causal pad checks
                if not np.all(np.isnan(cols["arm"][0])):
                    ep_errs.append(f"{KEY_QFRC_ARM}[0] is not all NaN")
                if not np.isnan(cols["arm_l2"][0]):
                    ep_errs.append(f"{KEY_QFRC_ARM_L2}[0] is not NaN")
                if bool(cols["valid"][0]):
                    ep_errs.append(f"{KEY_VALID}[0] should be False")
                if not np.isnan(cols["div"][0]):
                    ep_errs.append(f"{KEY_DIV}[0] should be NaN")

                # Row count consistency against replay states
                states = LU.get_episode_states(dataset_root, ep)
                if len(states) != t_horizon:
                    ep_errs.append(
                        f"row count mismatch: parquet={t_horizon} states={len(states)}"
                    )

                # Replay alignment consistency
                if args.check_alignment_replay:
                    assert env is not None
                    raw = collect_raw_replay(env, dataset_root, ep)
                    exp_arm = shift_causal(raw["qfrc_arm_raw"], fill_value=np.nan)
                    exp_l2 = shift_causal(raw["qfrc_arm_l2_raw"], fill_value=np.nan)
                    exp_div = shift_causal(raw["divergence_raw"], fill_value=np.nan)
                    exp_valid = np.zeros((t_horizon,), dtype=bool)
                    exp_valid[1:] = np.isfinite(raw["divergence_raw"][:-1]) & (
                        raw["divergence_raw"][:-1] <= float(div_thresh)
                    )

                    if not np.array_equal(np.isnan(cols["arm"]), np.isnan(exp_arm)):
                        ep_errs.append(f"{KEY_QFRC_ARM} NaN mask mismatch vs replay")
                    else:
                        arm_mask = np.isfinite(exp_arm)
                        if not np.allclose(
                            cols["arm"][arm_mask],
                            exp_arm[arm_mask],
                            rtol=args.rtol,
                            atol=args.atol,
                        ):
                            ep_errs.append(f"{KEY_QFRC_ARM} values mismatch vs replay")

                    if not np.array_equal(np.isnan(cols["arm_l2"]), np.isnan(exp_l2)):
                        ep_errs.append(f"{KEY_QFRC_ARM_L2} NaN mask mismatch vs replay")
                    else:
                        l2_mask = np.isfinite(exp_l2)
                        if not np.allclose(
                            cols["arm_l2"][l2_mask],
                            exp_l2[l2_mask],
                            rtol=args.rtol,
                            atol=args.atol,
                        ):
                            ep_errs.append(
                                f"{KEY_QFRC_ARM_L2} values mismatch vs replay"
                            )

                    if not np.array_equal(cols["valid"], exp_valid):
                        ep_errs.append(f"{KEY_VALID} values mismatch vs replay")

                    if not np.array_equal(np.isnan(cols["div"]), np.isnan(exp_div)):
                        ep_errs.append(f"{KEY_DIV} NaN mask mismatch vs replay")
                    else:
                        div_mask = np.isfinite(exp_div)
                        if not np.allclose(
                            cols["div"][div_mask],
                            exp_div[div_mask],
                            rtol=args.rtol,
                            atol=args.atol,
                        ):
                            ep_errs.append(f"{KEY_DIV} values mismatch vs replay")

                report["episode_results"].append(
                    {
                        "episode": ep,
                        "ok": len(ep_errs) == 0,
                        "errors": ep_errs,
                        "frames": t_horizon,
                        "elapsed_sec": round(time.time() - ep_t0, 4),
                    }
                )
                if ep_errs:
                    progress_log(
                        f"[fail] episode_{ep:06d}: {len(ep_errs)} validation error(s)",
                        pbar,
                    )
                else:
                    progress_log(f"[ok] episode_{ep:06d}", pbar)
            except Exception as e:
                report["episode_results"].append(
                    {
                        "episode": ep,
                        "ok": False,
                        "errors": [str(e)],
                        "elapsed_sec": round(time.time() - ep_t0, 4),
                    }
                )
                progress_log(f"[fail] episode_{ep:06d}: {e}", pbar)
            if pbar is not None and hasattr(pbar, "set_postfix"):
                ep_failures_now = sum(
                    1 for x in report["episode_results"] if not bool(x.get("ok", False))
                )
                pbar.set_postfix(
                    ok=len(report["episode_results"]) - ep_failures_now,
                    fail=ep_failures_now,
                    last=f"{ep:06d}",
                )
    finally:
        if pbar is not None:
            pbar.close()
        if env is not None:
            env.close()

    if not report["episode_results"]:
        report["errors"].append("No episodes were validated.")
    else:
        report["metadata_errors"] = validate_metadata(dataset_root)
        report["errors"].extend(report["metadata_errors"])

    # Loader compatibility smoke test
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id="robocasa365", root=str(dataset_root))
        idx0 = int(ds.episode_data_index["from"][episodes[0]])
        sample = ds[idx0]
        missing = [k for k in REQ_KEYS if k not in sample]
        if missing:
            report["loader_check"] = {
                "ok": False,
                "error": f"missing sample keys: {missing}",
            }
            report["errors"].append(f"Loader check missing sample keys: {missing}")
        else:
            report["loader_check"] = {"ok": True, "error": None}
    except Exception as e:
        report["loader_check"] = {"ok": False, "error": str(e)}
        report["errors"].append(f"Loader check failed: {e}")

    ep_failures = [x for x in report["episode_results"] if not x["ok"]]
    report["summary"] = {
        "validated_count": len(report["episode_results"]),
        "episode_failures": len(ep_failures),
        "metadata_failures": len(report["metadata_errors"]),
        "has_errors": len(report["errors"]) > 0 or len(ep_failures) > 0,
        "elapsed_sec": round(time.time() - t0, 4),
    }
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    dump_json(report_path, report)
    print(f"Validation report: {report_path}")
    print(
        f"Validated {report['summary']['validated_count']} episodes; "
        f"episode failures={report['summary']['episode_failures']}; "
        f"metadata failures={report['summary']['metadata_failures']}"
    )

    if args.strict and report["summary"]["has_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
