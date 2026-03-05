#!/usr/bin/env python3
"""
Label force phases in an augmented LeRobot dataset.

Phase labels:
  -1: invalid
   0: free_motion
   1: precontact
   2: contact
"""

from __future__ import annotations

import argparse
import json
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


KEY_SIGNAL = "observation.force.qfrc_constraint_arm_l2"
KEY_FORCE_VALID = "diagnostic.force_valid"

KEY_PHASE_LABEL = "diagnostic.force_phase.label"
KEY_PHASE_VALID = "diagnostic.force_phase.valid"
KEY_SIGNAL_SMOOTH = "diagnostic.force_phase.signal_smoothed"


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


def progress_log(msg: str, pbar: Any | None) -> None:
    if pbar is not None and hasattr(pbar, "write"):
        pbar.write(msg)
    else:
        print(msg)


def scalar_from_cell(x, dtype=np.float64):
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    if arr.size == 0:
        return np.nan
    return arr[0]


def ema_causal(signal: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full(signal.shape, np.nan, dtype=np.float64)
    init = False
    prev = np.nan
    a = float(alpha)
    for t in range(signal.shape[0]):
        if not valid[t] or not np.isfinite(signal[t]):
            continue
        x = float(signal[t])
        if not init:
            prev = x
            init = True
        else:
            prev = a * x + (1.0 - a) * prev
        out[t] = prev
    return out


def find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = mask.shape[0]
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def merge_contact_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    runs = find_true_runs(mask)
    if not runs:
        return mask.copy()
    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        pstart, pend = merged[-1]
        gap = start - pend
        if gap <= max_gap:
            merged[-1] = (pstart, end)
        else:
            merged.append((start, end))

    out = np.zeros_like(mask, dtype=bool)
    for start, end in merged:
        out[start:end] = True
    return out


def drop_short_contact_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for start, end in find_true_runs(mask):
        if end - start >= min_len:
            out[start:end] = True
    return out


def hysteresis_contact(
    smooth: np.ndarray, valid: np.ndarray, tau_on: float, tau_off: float
) -> np.ndarray:
    in_contact = False
    out = np.zeros_like(valid, dtype=bool)
    for t in range(valid.shape[0]):
        if not valid[t] or not np.isfinite(smooth[t]):
            in_contact = False
            out[t] = False
            continue

        s = float(smooth[t])
        if in_contact:
            if s <= tau_off:
                in_contact = False
        else:
            if s >= tau_on:
                in_contact = True
        out[t] = in_contact
    return out


def build_precontact_mask(
    contact_mask: np.ndarray, valid_mask: np.ndarray, pre_window: int
) -> np.ndarray:
    pre = np.zeros_like(contact_mask, dtype=bool)
    for start, _end in find_true_runs(contact_mask):
        a = max(0, start - pre_window)
        b = start
        if b <= a:
            continue
        idx = np.arange(a, b)
        allow = valid_mask[idx] & (~contact_mask[idx])
        pre[idx[allow]] = True
    return pre


def read_episode_signal(
    dataset_root: Path, episode: int
) -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    p = find_episode_parquet(dataset_root, episode)
    df = pd.read_parquet(p)
    if KEY_SIGNAL not in df.columns:
        raise RuntimeError(f"Missing required signal key {KEY_SIGNAL} in {p}")
    if KEY_FORCE_VALID not in df.columns:
        raise RuntimeError(f"Missing required validity key {KEY_FORCE_VALID} in {p}")

    signal = np.asarray(
        [scalar_from_cell(v, dtype=np.float64) for v in df[KEY_SIGNAL].tolist()],
        dtype=np.float64,
    )
    force_valid = np.asarray(
        [
            bool(scalar_from_cell(v, dtype=np.bool_))
            for v in df[KEY_FORCE_VALID].tolist()
        ],
        dtype=bool,
    )
    return p, df, signal, force_valid


def fit_thresholds(
    dataset_root: Path,
    episodes: list[int],
    ema_alpha: float,
    tau_on_q: float,
    tau_off_q: float,
    min_hysteresis_gap: float,
) -> dict[str, float]:
    pooled = []
    for ep in episodes:
        _p, _df, signal, force_valid = read_episode_signal(dataset_root, ep)
        valid = force_valid & np.isfinite(signal)
        smooth = ema_causal(signal, valid, alpha=ema_alpha)
        vals = smooth[valid & np.isfinite(smooth)]
        if vals.size:
            pooled.append(vals)

    if not pooled:
        raise RuntimeError("No valid force samples found for threshold calibration.")

    x = np.concatenate(pooled, axis=0)
    tau_on = float(np.quantile(x, tau_on_q))
    tau_off = float(np.quantile(x, tau_off_q))
    if tau_on <= tau_off:
        tau_on = float(tau_off + min_hysteresis_gap)
    return {"tau_on": tau_on, "tau_off": tau_off}


def label_episode(
    signal: np.ndarray,
    force_valid: np.ndarray,
    threshold_mode: str,
    contact_threshold: float,
    tau_on: float,
    tau_off: float,
    ema_alpha: float,
    min_contact_frames: int,
    merge_gap_frames: int,
    precontact_window: int,
    invalid_label: int,
    free_label: int,
    precontact_label: int,
    contact_label: int,
) -> dict[str, np.ndarray]:
    phase_valid = force_valid & np.isfinite(signal)
    smooth = ema_causal(signal, phase_valid, alpha=ema_alpha)

    if threshold_mode == "absolute":
        # User-driven rule: any valid frame with force > threshold is contact.
        contact = phase_valid & (signal > float(contact_threshold))
    elif threshold_mode == "quantile":
        contact = hysteresis_contact(
            smooth, phase_valid, tau_on=tau_on, tau_off=tau_off
        )
        contact = merge_contact_gaps(contact, max_gap=int(merge_gap_frames))
        contact = drop_short_contact_runs(contact, min_len=int(min_contact_frames))
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    precontact = build_precontact_mask(
        contact_mask=contact, valid_mask=phase_valid, pre_window=int(precontact_window)
    )
    precontact = precontact & (~contact)

    labels = np.full((signal.shape[0],), int(invalid_label), dtype=np.int64)
    labels[phase_valid] = int(free_label)
    labels[precontact] = int(precontact_label)
    labels[contact] = int(contact_label)

    return {
        "labels": labels,
        "phase_valid": phase_valid,
        "smooth": smooth,
        "contact_mask": contact,
        "precontact_mask": precontact,
    }


def write_episode_labels(
    parquet_path: Path,
    df: pd.DataFrame,
    labels_pack: dict[str, np.ndarray],
    write_debug_smoothed_signal: bool,
    overwrite_labels: bool,
) -> None:
    if not overwrite_labels:
        for key in [KEY_PHASE_LABEL, KEY_PHASE_VALID, KEY_SIGNAL_SMOOTH]:
            if key in df.columns:
                raise RuntimeError(
                    f"{parquet_path}: column {key} already exists; use --overwrite-labels"
                )

    labels = labels_pack["labels"]
    phase_valid = labels_pack["phase_valid"]
    smooth = labels_pack["smooth"]

    df[KEY_PHASE_LABEL] = [np.asarray([v], dtype=np.int64) for v in labels]
    df[KEY_PHASE_VALID] = [np.asarray([v], dtype=bool) for v in phase_valid]
    if write_debug_smoothed_signal:
        df[KEY_SIGNAL_SMOOTH] = [np.asarray([v], dtype=np.float64) for v in smooth]

    df.to_parquet(parquet_path, index=False)


def update_info_json(meta_dir: Path, write_debug_smoothed_signal: bool) -> None:
    info_path = meta_dir / "info.json"
    info = load_json(info_path)
    features = info.setdefault("features", {})

    features[KEY_PHASE_LABEL] = {"dtype": "int64", "shape": [1]}
    features[KEY_PHASE_VALID] = {"dtype": "bool", "shape": [1]}
    if write_debug_smoothed_signal:
        features[KEY_SIGNAL_SMOOTH] = {"dtype": "float64", "shape": [1]}

    dump_json(info_path, info)


def update_modality_json(meta_dir: Path, write_debug_smoothed_signal: bool) -> None:
    modality_path = meta_dir / "modality.json"
    modality = load_json(modality_path)
    diagnostic = modality.setdefault("diagnostic", {})

    diagnostic["force_phase_label"] = {"original_key": KEY_PHASE_LABEL}
    diagnostic["force_phase_valid"] = {"original_key": KEY_PHASE_VALID}
    if write_debug_smoothed_signal:
        diagnostic["force_phase_signal_smoothed"] = {"original_key": KEY_SIGNAL_SMOOTH}

    dump_json(modality_path, modality)


def stat_dict(arr: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(arr, dtype=np.float64)
    return {
        "mean": np.nanmean(x, axis=0).tolist(),
        "std": np.nanstd(x, axis=0).tolist(),
        "min": np.nanmin(x, axis=0).tolist(),
        "max": np.nanmax(x, axis=0).tolist(),
        "q01": np.nanquantile(x, 0.01, axis=0).tolist(),
        "q99": np.nanquantile(x, 0.99, axis=0).tolist(),
    }


def recompute_stats(
    dataset_root: Path,
    episodes: list[int],
    write_debug_smoothed_signal: bool,
) -> None:
    keys = [KEY_PHASE_LABEL, KEY_PHASE_VALID]
    if write_debug_smoothed_signal:
        keys.append(KEY_SIGNAL_SMOOTH)

    accum: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for ep in episodes:
        p = find_episode_parquet(dataset_root, ep)
        df = pd.read_parquet(p)
        for key in keys:
            if key not in df.columns:
                raise RuntimeError(f"Missing {key} in {p}")
            rows = [
                np.asarray(v, dtype=np.float64).reshape(-1) for v in df[key].tolist()
            ]
            accum[key].append(np.vstack(rows))

    stats_path = dataset_root / "meta" / "stats.json"
    stats = load_json(stats_path)
    for key, parts in accum.items():
        stats[key] = stat_dict(np.vstack(parts))
    dump_json(stats_path, stats)


def write_phase_schema(
    meta_dir: Path,
    config: dict[str, Any],
    thresholds: dict[str, float],
    episodes: list[int],
    calibration_episodes: list[int],
) -> None:
    schema = {
        "force_phase_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": {
            "invalid": int(config["invalid_label"]),
            "free_motion": int(config["free_label"]),
            "precontact": int(config["precontact_label"]),
            "contact": int(config["contact_label"]),
        },
        "thresholds": thresholds,
        "config": config,
        "episodes_labeled": episodes,
        "calibration_episodes": calibration_episodes,
        "keys": {
            KEY_PHASE_LABEL: {"dtype": "int64", "shape": [1]},
            KEY_PHASE_VALID: {"dtype": "bool", "shape": [1]},
            KEY_SIGNAL_SMOOTH: {"dtype": "float64", "shape": [1]},
        },
    }
    dump_json(meta_dir / "force_phase_schema.json", schema)


def default_report_path(dataset_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("artifacts")
        / "force_phase_reports"
        / f"{dataset_root.name}_force_phase_report_{ts}.json"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Augmented dataset root"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fit_apply",
        choices=["fit", "apply", "fit_apply"],
        help="fit: thresholds only, apply: labels only from schema, fit_apply: both",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episodes to label/apply: all | 0-103 | 0,1,2",
    )
    parser.add_argument(
        "--calibration-episodes",
        type=str,
        default=None,
        help="Episodes for threshold fitting. Default: same as --episodes",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional report path; default under artifacts/force_phase_reports/",
    )

    # Thresholding strategy
    parser.add_argument(
        "--threshold-mode",
        type=str,
        choices=["absolute", "quantile"],
        default="absolute",
        help="absolute: contact if raw arm_l2 > contact-threshold; quantile: fitted hysteresis",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=20.0,
        help="Used in threshold-mode=absolute; contact if arm_l2 > this value",
    )

    # Smoothing + quantile mode params
    parser.add_argument("--ema-alpha", type=float, default=0.20)
    parser.add_argument("--tau-on-quantile", type=float, default=0.88)
    parser.add_argument("--tau-off-quantile", type=float, default=0.72)
    parser.add_argument("--min-hysteresis-gap", type=float, default=1e-6)
    parser.add_argument("--min-contact-frames", type=int, default=4)
    parser.add_argument("--merge-gap-frames", type=int, default=2)
    parser.add_argument("--precontact-window", type=int, default=25)

    parser.add_argument("--invalid-label", type=int, default=-1)
    parser.add_argument("--free-label", type=int, default=0)
    parser.add_argument("--precontact-label", type=int, default=1)
    parser.add_argument("--contact-label", type=int, default=2)

    parser.add_argument(
        "--overwrite-labels",
        action="store_true",
        help="Allow overwriting existing label/debug columns",
    )
    parser.add_argument(
        "--write-debug-smoothed-signal",
        action="store_true",
        help="Write diagnostic.force_phase.signal_smoothed column",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip stats.json updates for phase keys",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Invalid dataset root: {dataset_root}")

    total_episodes = get_total_episodes(dataset_root)
    episodes = parse_episodes_spec(args.episodes, total_episodes)
    calibration_spec = (
        args.calibration_episodes
        if args.calibration_episodes is not None
        else args.episodes
    )
    calibration_episodes = parse_episodes_spec(calibration_spec, total_episodes)

    report_path = (
        Path(args.report_json)
        if args.report_json
        else default_report_path(dataset_root)
    )
    t0 = time.time()

    config = {
        "threshold_mode": args.threshold_mode,
        "contact_threshold": float(args.contact_threshold),
        "ema_alpha": float(args.ema_alpha),
        "tau_on_quantile": float(args.tau_on_quantile),
        "tau_off_quantile": float(args.tau_off_quantile),
        "min_hysteresis_gap": float(args.min_hysteresis_gap),
        "min_contact_frames": int(args.min_contact_frames),
        "merge_gap_frames": int(args.merge_gap_frames),
        "precontact_window": int(args.precontact_window),
        "invalid_label": int(args.invalid_label),
        "free_label": int(args.free_label),
        "precontact_label": int(args.precontact_label),
        "contact_label": int(args.contact_label),
        "key_signal": KEY_SIGNAL,
        "key_force_valid": KEY_FORCE_VALID,
    }

    schema_path = dataset_root / "meta" / "force_phase_schema.json"
    thresholds: dict[str, float] | None = None
    threshold_mode_used = args.threshold_mode
    contact_threshold_used = float(args.contact_threshold)
    if args.mode in ("fit", "fit_apply"):
        if args.threshold_mode == "absolute":
            thresholds = {
                "tau_on": float(args.contact_threshold),
                "tau_off": float(args.contact_threshold),
            }
        else:
            thresholds = fit_thresholds(
                dataset_root=dataset_root,
                episodes=calibration_episodes,
                ema_alpha=args.ema_alpha,
                tau_on_q=args.tau_on_quantile,
                tau_off_q=args.tau_off_quantile,
                min_hysteresis_gap=args.min_hysteresis_gap,
            )
    if args.mode in ("apply", "fit_apply"):
        if thresholds is None:
            if not schema_path.exists():
                raise FileNotFoundError(
                    "No fitted thresholds available. Run with --mode fit or fit_apply first."
                )
            schema = load_json(schema_path)
            if "thresholds" not in schema:
                raise RuntimeError(f"Missing thresholds in {schema_path}")
            thresholds = {
                "tau_on": float(schema["thresholds"]["tau_on"]),
                "tau_off": float(schema["thresholds"]["tau_off"]),
            }
            threshold_mode_used = schema.get("config", {}).get(
                "threshold_mode", args.threshold_mode
            )
            contact_threshold_used = float(
                schema.get("config", {}).get(
                    "contact_threshold", args.contact_threshold
                )
            )
    config["threshold_mode"] = threshold_mode_used
    config["contact_threshold"] = contact_threshold_used

    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_root),
        "mode": args.mode,
        "episodes": episodes,
        "calibration_episodes": calibration_episodes,
        "config": config,
        "thresholds": thresholds,
        "episode_results": [],
    }

    if args.mode in ("apply", "fit_apply"):
        pbar: Any | None = None
        ep_iter: Any = episodes
        if tqdm is not None:
            pbar = tqdm(
                episodes,
                total=len(episodes),
                desc="Label episodes",
                unit="ep",
                dynamic_ncols=True,
            )
            ep_iter = pbar

        try:
            for ep in ep_iter:
                ep_t0 = time.time()
                try:
                    parquet_path, df, signal, force_valid = read_episode_signal(
                        dataset_root, ep
                    )
                    assert thresholds is not None
                    labels_pack = label_episode(
                        signal=signal,
                        force_valid=force_valid,
                        threshold_mode=threshold_mode_used,
                        contact_threshold=contact_threshold_used,
                        tau_on=float(thresholds["tau_on"]),
                        tau_off=float(thresholds["tau_off"]),
                        ema_alpha=args.ema_alpha,
                        min_contact_frames=args.min_contact_frames,
                        merge_gap_frames=args.merge_gap_frames,
                        precontact_window=args.precontact_window,
                        invalid_label=args.invalid_label,
                        free_label=args.free_label,
                        precontact_label=args.precontact_label,
                        contact_label=args.contact_label,
                    )

                    write_episode_labels(
                        parquet_path=parquet_path,
                        df=df,
                        labels_pack=labels_pack,
                        write_debug_smoothed_signal=args.write_debug_smoothed_signal,
                        overwrite_labels=args.overwrite_labels,
                    )

                    labels = labels_pack["labels"]
                    report["episode_results"].append(
                        {
                            "episode": ep,
                            "ok": True,
                            "frames": int(labels.shape[0]),
                            "counts": {
                                str(args.invalid_label): int(
                                    np.sum(labels == args.invalid_label)
                                ),
                                str(args.free_label): int(
                                    np.sum(labels == args.free_label)
                                ),
                                str(args.precontact_label): int(
                                    np.sum(labels == args.precontact_label)
                                ),
                                str(args.contact_label): int(
                                    np.sum(labels == args.contact_label)
                                ),
                            },
                            "elapsed_sec": round(time.time() - ep_t0, 4),
                        }
                    )
                    progress_log(f"[ok] episode_{ep:06d}", pbar)
                except Exception as e:
                    report["episode_results"].append(
                        {
                            "episode": ep,
                            "ok": False,
                            "error": str(e),
                            "elapsed_sec": round(time.time() - ep_t0, 4),
                        }
                    )
                    progress_log(f"[fail] episode_{ep:06d}: {e}", pbar)
                if pbar is not None and hasattr(pbar, "set_postfix"):
                    fail_count = sum(
                        1 for r in report["episode_results"] if not r["ok"]
                    )
                    pbar.set_postfix(
                        ok=len(report["episode_results"]) - fail_count,
                        fail=fail_count,
                        last=f"{ep:06d}",
                    )
        finally:
            if pbar is not None:
                pbar.close()

        if args.mode in ("fit_apply", "apply"):
            update_info_json(
                dataset_root / "meta",
                write_debug_smoothed_signal=args.write_debug_smoothed_signal,
            )
            update_modality_json(
                dataset_root / "meta",
                write_debug_smoothed_signal=args.write_debug_smoothed_signal,
            )
            if not args.skip_stats and report["episode_results"]:
                ok_eps = [r["episode"] for r in report["episode_results"] if r["ok"]]
                if ok_eps:
                    recompute_stats(
                        dataset_root=dataset_root,
                        episodes=ok_eps,
                        write_debug_smoothed_signal=args.write_debug_smoothed_signal,
                    )

    if thresholds is not None and args.mode in ("fit", "fit_apply"):
        write_phase_schema(
            meta_dir=dataset_root / "meta",
            config=config,
            thresholds=thresholds,
            episodes=episodes,
            calibration_episodes=calibration_episodes,
        )

    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_sec"] = round(time.time() - t0, 4)
    report["summary"] = {
        "episodes_total": len(episodes),
        "episodes_ok": sum(1 for r in report["episode_results"] if r.get("ok", False)),
        "episodes_failed": sum(
            1 for r in report["episode_results"] if not r.get("ok", False)
        ),
    }
    dump_json(report_path, report)

    print(f"Phase report: {report_path}")
    print(
        f"Labeled episodes: ok={report['summary']['episodes_ok']} "
        f"failed={report['summary']['episodes_failed']}"
    )

    if report["summary"]["episodes_failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
