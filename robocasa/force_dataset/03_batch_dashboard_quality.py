#!/usr/bin/env python3
"""
Batch quality-gated force dashboard pipeline.

This script orchestrates episode processing across one or more LeRobot datasets using:
- force_dataset/02_export_forces_single_episode.py (dashboard + summary)
- force_dataset/01_probe_qfrc_single_episode.py (QC-only mode when --skip-video is set)

It does NOT modify datasets. It writes artifacts and reports under --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def get_task_name(dataset_root: Path) -> str:
    meta_path = dataset_root / "extras" / "dataset_meta.json"
    if meta_path.exists():
        data = load_json(meta_path)
        env_args = data.get("env_args", {})
        env_name = env_args.get("env_name")
        if isinstance(env_name, str) and env_name:
            return env_name
    return (
        dataset_root.parent.parent.name
        if dataset_root.parent.parent.name
        else dataset_root.name
    )


def get_total_episodes(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    info = load_json(info_path)
    total = info.get("total_episodes")
    if not isinstance(total, int):
        raise ValueError(f"Invalid total_episodes in {info_path}")
    return total


def parse_episodes_spec(spec: str, total_episodes: int) -> list[int]:
    if spec.strip().lower() == "all":
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


def extract_metrics(
    raw_summary: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Returns:
      divergence_stats, ncon_stats, selected_cfrc_topk_body_names
    Works for both dashboard summaries and probe summaries.
    """
    # Dashboard summary path
    if "divergence_l2_stats" in raw_summary:
        div = raw_summary.get("divergence_l2_stats", {})
        ncon = raw_summary.get("ncon_stats", {})
        topk = raw_summary.get("selected_cfrc_topk_body_names", [])
        return div, ncon, topk if isinstance(topk, list) else []

    # Probe summary path
    div = {}
    ncon = {}
    topk = []

    action_replay = raw_summary.get("action_replay", {})
    if isinstance(action_replay, dict):
        div_raw = action_replay.get("divergence_l2", {})
        if isinstance(div_raw, dict):
            div = {
                "count": div_raw.get("count"),
                "mean": div_raw.get("mean"),
                "max": div_raw.get("max"),
                "p95": div_raw.get("p95"),
                "valid_ratio": None,
                "threshold": None,
            }

        summary_rows = action_replay.get("summary", [])
        if isinstance(summary_rows, list):
            for row in summary_rows:
                if isinstance(row, dict) and row.get("name") == "ncon":
                    ncon = {
                        "mean": row.get("mean_abs"),
                        "std": None,
                        "min": None,
                        "max": row.get("max_abs"),
                    }
                    break

    return div, ncon, topk


def classify_quality(
    divergence_stats: dict[str, Any],
    green_thresh: float,
    yellow_thresh: float,
) -> tuple[str, list[str]]:
    p95 = divergence_stats.get("p95")
    dmax = divergence_stats.get("max")
    valid_ratio = divergence_stats.get("valid_ratio")

    reasons: list[str] = []

    if p95 is None:
        tier = "red"
        reasons.append("missing_p95")
        return tier, reasons

    if p95 <= green_thresh:
        tier = "green"
    elif p95 <= yellow_thresh:
        tier = "yellow"
    else:
        tier = "red"

    if dmax is not None and dmax > 2.5:
        reasons.append("spike_outlier")
    if valid_ratio is not None and valid_ratio < 0.90:
        reasons.append("low_valid_ratio")

    return tier, reasons


def run_subprocess(cmd: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def make_runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    script_path = Path(__file__).resolve()
    # /home/alfredo/robocasa/robocasa/force_dataset/03_...
    workspace_root = script_path.parents[2]  # /home/alfredo/robocasa
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(workspace_root) if not old else f"{workspace_root}:{old}"
    return env


def normalize_summary(
    raw_summary: dict[str, Any],
    dataset_root: Path,
    task_name: str,
    episode: int,
    output_video: Path | None,
    green_thresh: float,
    yellow_thresh: float,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    divergence_stats, ncon_stats, topk_names = extract_metrics(raw_summary)
    tier, reasons = classify_quality(divergence_stats, green_thresh, yellow_thresh)

    base = dict(raw_summary)
    base["dataset_root"] = str(dataset_root)
    base["task_name"] = task_name
    base["episode"] = int(episode)
    base["drift_metric"] = "one_step_l2"
    base["quality_tier"] = tier
    base["quality_reasons"] = reasons
    base["status"] = status
    base["error"] = error
    base["output_video"] = str(output_video) if output_video else None

    # Ensure standardized keys exist even in probe mode
    if "divergence_l2_stats" not in base:
        base["divergence_l2_stats"] = divergence_stats
    if "ncon_stats" not in base:
        base["ncon_stats"] = ncon_stats
    if "selected_cfrc_topk_body_names" not in base:
        base["selected_cfrc_topk_body_names"] = topk_names

    return base


def process_one_episode(job: dict[str, Any]) -> dict[str, Any]:
    dataset_root: Path = job["dataset_root"]
    task_name: str = job["task_name"]
    episode: int = job["episode"]
    task_out_dir: Path = job["task_out_dir"]
    args = job["args"]
    env = job["env"]
    script_dir: Path = job["script_dir"]

    video_path = task_out_dir / f"force_dashboard_ep{episode:06d}.mp4"
    summary_path = task_out_dir / f"force_dashboard_ep{episode:06d}_summary.json"

    if args.resume and summary_path.exists():
        existing = load_json(summary_path)
        if existing.get("status") == "ok":
            return {
                "dataset_root": str(dataset_root),
                "task_name": task_name,
                "episode": episode,
                "summary_path": str(summary_path),
                "status": "skipped",
            }

    task_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.skip_video:
            probe_script = script_dir / "01_probe_qfrc_single_episode.py"
            cmd = [
                sys.executable,
                str(probe_script),
                "--dataset",
                str(dataset_root),
                "--episode",
                str(episode),
                "--replay-mode",
                "action",
                "--out-json",
                str(summary_path),
            ]
            code, stdout, stderr = run_subprocess(cmd, env=env)
            if code != 0:
                raise RuntimeError(f"probe failed: {stderr[-2000:]}")
            raw = load_json(summary_path)
            normalized = normalize_summary(
                raw_summary=raw,
                dataset_root=dataset_root,
                task_name=task_name,
                episode=episode,
                output_video=None,
                green_thresh=args.div_thresh_green,
                yellow_thresh=args.div_thresh_yellow,
            )
            dump_json(summary_path, normalized)
        else:
            dash_script = script_dir / "02_export_forces_single_episode.py"
            cmd = [
                sys.executable,
                str(dash_script),
                "--dataset",
                str(dataset_root),
                "--episode",
                str(episode),
                "--output-video",
                str(video_path),
                "--camera-name",
                args.camera_name,
                "--fps",
                str(args.fps),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--topk-cfrc",
                str(args.topk_cfrc),
                "--div-thresh",
                str(args.div_thresh_green),
            ]
            if args.include_gripper:
                cmd.append("--include-gripper")
            if args.max_steps is not None:
                cmd.extend(["--max-steps", str(args.max_steps)])

            code, stdout, stderr = run_subprocess(cmd, env=env)
            if code != 0:
                raise RuntimeError(f"dashboard failed: {stderr[-2000:]}")

            raw = load_json(summary_path)
            normalized = normalize_summary(
                raw_summary=raw,
                dataset_root=dataset_root,
                task_name=task_name,
                episode=episode,
                output_video=video_path,
                green_thresh=args.div_thresh_green,
                yellow_thresh=args.div_thresh_yellow,
            )
            dump_json(summary_path, normalized)

    except Exception as e:
        fallback = normalize_summary(
            raw_summary={},
            dataset_root=dataset_root,
            task_name=task_name,
            episode=episode,
            output_video=None if args.skip_video else video_path,
            green_thresh=args.div_thresh_green,
            yellow_thresh=args.div_thresh_yellow,
            status="error",
            error=str(e),
        )
        dump_json(summary_path, fallback)

    return {
        "dataset_root": str(dataset_root),
        "task_name": task_name,
        "episode": episode,
        "summary_path": str(summary_path),
        "status": load_json(summary_path).get("status", "error"),
    }


def percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals_sorted = sorted(vals)
    idx = (len(vals_sorted) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals_sorted) - 1)
    frac = idx - lo
    return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac


def build_aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        by_dataset[s.get("dataset_root", "unknown")].append(s)

    dataset_reports = {}
    cross = []

    for ds, rows in by_dataset.items():
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        tiers = Counter(r.get("quality_tier", "unknown") for r in ok_rows)
        statuses = Counter(r.get("status", "unknown") for r in rows)

        p95_vals = [
            r.get("divergence_l2_stats", {}).get("p95")
            for r in ok_rows
            if r.get("divergence_l2_stats", {}).get("p95") is not None
        ]
        p95_vals = [float(x) for x in p95_vals]

        ncon_means = [
            r.get("ncon_stats", {}).get("mean")
            for r in ok_rows
            if r.get("ncon_stats", {}).get("mean") is not None
        ]
        ncon_means = [float(x) for x in ncon_means]

        body_counter = Counter()
        for r in ok_rows:
            for name in r.get("selected_cfrc_topk_body_names", []) or []:
                body_counter[name] += 1

        report = {
            "dataset_root": ds,
            "task_name": ok_rows[0].get("task_name")
            if ok_rows
            else rows[0].get("task_name"),
            "num_rows": len(rows),
            "status_counts": dict(statuses),
            "quality_tier_counts": dict(tiers),
            "drift_p95_distribution": {
                "count": len(p95_vals),
                "mean": mean(p95_vals) if p95_vals else None,
                "p50": percentile(p95_vals, 0.50),
                "p95": percentile(p95_vals, 0.95),
                "max": max(p95_vals) if p95_vals else None,
            },
            "ncon_mean_distribution": {
                "count": len(ncon_means),
                "mean": mean(ncon_means) if ncon_means else None,
                "p50": percentile(ncon_means, 0.50),
                "p95": percentile(ncon_means, 0.95),
                "max": max(ncon_means) if ncon_means else None,
            },
            "top_recurring_high_force_bodies": body_counter.most_common(20),
        }
        dataset_reports[ds] = report

        ok_n = max(1, len(ok_rows))
        green_ratio = tiers.get("green", 0) / ok_n
        cross.append(
            {
                "dataset_root": ds,
                "task_name": report["task_name"],
                "green_ratio": green_ratio,
                "mean_p95": report["drift_p95_distribution"]["mean"],
                "num_ok": len(ok_rows),
            }
        )

    cross_sorted = sorted(
        cross, key=lambda x: (-x["green_ratio"], (x["mean_p95"] or 1e9))
    )
    return {
        "generated_at": datetime.now().isoformat(),
        "total_summaries": len(summaries),
        "dataset_reports": dataset_reports,
        "cross_dataset_comparison": cross_sorted,
    }


def write_episode_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    headers = [
        "dataset_root",
        "task_name",
        "episode",
        "status",
        "quality_tier",
        "drift_p95",
        "drift_max",
        "valid_ratio",
        "ncon_mean",
        "ncon_max",
        "output_video",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for s in summaries:
            div = s.get("divergence_l2_stats", {}) or {}
            ncon = s.get("ncon_stats", {}) or {}
            writer.writerow(
                {
                    "dataset_root": s.get("dataset_root"),
                    "task_name": s.get("task_name"),
                    "episode": s.get("episode"),
                    "status": s.get("status"),
                    "quality_tier": s.get("quality_tier"),
                    "drift_p95": div.get("p95"),
                    "drift_max": div.get("max"),
                    "valid_ratio": div.get("valid_ratio"),
                    "ncon_mean": ncon.get("mean"),
                    "ncon_max": ncon.get("max"),
                    "output_video": s.get("output_video"),
                    "summary_path": s.get("_summary_path"),
                }
            )


def build_training_handoff(
    summaries: list[dict[str, Any]]
) -> tuple[dict[str, list[int]], dict[str, dict[str, float]]]:
    green_episodes: dict[str, list[int]] = defaultdict(list)
    weights: dict[str, dict[str, float]] = defaultdict(dict)

    tier_weight = {"green": 1.0, "yellow": 0.5, "red": 0.1}

    for s in summaries:
        if s.get("status") != "ok":
            continue
        ds = s.get("dataset_root")
        ep = s.get("episode")
        tier = s.get("quality_tier", "red")
        if ds is None or ep is None:
            continue

        if tier == "green":
            green_episodes[ds].append(int(ep))

        weights[ds][str(ep)] = float(tier_weight.get(tier, 0.1))

    for ds in green_episodes:
        green_episodes[ds] = sorted(set(green_episodes[ds]))

    # deterministic ordering
    green_ordered = {k: green_episodes[k] for k in sorted(green_episodes)}
    weights_ordered = {
        k: dict(sorted(weights[k].items(), key=lambda kv: int(kv[0])))
        for k in sorted(weights)
    }
    return green_ordered, weights_ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="One or more absolute dataset roots",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode selection: all | 0,1,2 | 0-50 | mixed",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output run directory. Default: artifacts/quality_runs/<timestamp>",
    )
    parser.add_argument("--camera-name", type=str, default="robot0_agentview_right")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--topk-cfrc", type=int, default=8)
    parser.add_argument("--div-thresh-green", type=float, default=0.10)
    parser.add_argument("--div-thresh-yellow", type=float, default=0.20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    env = make_runtime_env()

    if args.out_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("artifacts") / "quality_runs" / run_id
    else:
        run_dir = Path(args.out_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ds in args.datasets:
        ds_root = Path(ds).resolve()
        total = get_total_episodes(ds_root)
        eps = parse_episodes_spec(args.episodes, total)
        task_name = get_task_name(ds_root)
        task_out_dir = run_dir / task_name

        for ep in eps:
            jobs.append(
                {
                    "dataset_root": ds_root,
                    "task_name": task_name,
                    "episode": ep,
                    "task_out_dir": task_out_dir,
                    "args": args,
                    "env": env,
                    "script_dir": script_dir,
                }
            )

    print(f"Running {len(jobs)} episode jobs across {len(args.datasets)} dataset(s)")
    print(f"Output dir: {run_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process_one_episode, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(
                f"[{r['status']}] {r['task_name']} ep{int(r['episode']):06d} -> {r['summary_path']}"
            )

    summary_files = [Path(r["summary_path"]) for r in results]
    summaries = []
    for p in summary_files:
        s = load_json(p)
        s["_summary_path"] = str(p)
        summaries.append(s)

    aggregate = build_aggregate(summaries)
    aggregate_path = run_dir / "aggregate_report.json"
    dump_json(aggregate_path, aggregate)

    csv_path = run_dir / "episode_quality.csv"
    write_episode_csv(csv_path, summaries)

    green_episodes, episode_weights = build_training_handoff(summaries)
    dump_json(run_dir / "green_episodes.json", green_episodes)
    dump_json(run_dir / "episode_weights.json", episode_weights)

    print(f"Saved aggregate report: {aggregate_path}")
    print(f"Saved CSV index: {csv_path}")
    print(f"Saved allowlist: {run_dir / 'green_episodes.json'}")
    print(f"Saved weights: {run_dir / 'episode_weights.json'}")


if __name__ == "__main__":
    main()
