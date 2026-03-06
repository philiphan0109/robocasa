#!/usr/bin/env python3
"""
Sweep hysteresis (high, low) thresholds and report region distribution:
free motion, precontact, contact.

This script does not modify datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SWEEP = [
    (0.30, 0.10),
    (0.30, 0.05),
    (0.20, 0.10),
    (0.20, 0.05),
    (0.15, 0.10),
    (0.15, 0.05),
]


@dataclass
class SweepStats:
    input_high: float
    input_low: float
    eff_high: float
    eff_low: float
    swapped: bool
    free_count: int = 0
    pre_count: int = 0
    contact_count: int = 0
    frames_total: int = 0
    episodes_used: int = 0

    def add_counts(self, free_n: int, pre_n: int, contact_n: int):
        self.free_count += int(free_n)
        self.pre_count += int(pre_n)
        self.contact_count += int(contact_n)
        self.frames_total += int(free_n + pre_n + contact_n)

    def as_dict(self) -> dict:
        total = max(self.frames_total, 1)
        return {
            "input_high": self.input_high,
            "input_low": self.input_low,
            "effective_high": self.eff_high,
            "effective_low": self.eff_low,
            "swapped_high_low": self.swapped,
            "episodes_used": self.episodes_used,
            "frames_total": self.frames_total,
            "free_count": self.free_count,
            "precontact_count": self.pre_count,
            "contact_count": self.contact_count,
            "free_ratio": float(self.free_count / total),
            "precontact_ratio": float(self.pre_count / total),
            "contact_ratio": float(self.contact_count / total),
        }


def find_episode_parquet(dataset_root: Path, episode: int) -> Path:
    matches = list(dataset_root.glob(f"data/*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode_{episode:06d} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple parquet files for episode {episode}: {matches}")
    return matches[0]


def parse_episodes_spec(spec: str, total_episodes: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total_episodes))

    out: set[int] = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
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


def parse_sweep_pairs(s: str | None) -> list[tuple[float, float]]:
    if s is None or not s.strip():
        return list(DEFAULT_SWEEP)
    out = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if ":" not in part:
            raise ValueError(
                f"Invalid sweep pair '{part}'. Use format high:low,high:low"
            )
        hi_s, lo_s = part.split(":", 1)
        out.append((float(hi_s), float(lo_s)))
    return out


def to_1d_contact(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim > 1:
        x = np.abs(x).sum(axis=-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.reshape(-1)


def normalize_contact_1d(contact: np.ndarray, mode: str) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float64)
    if mode == "none":
        return c
    if mode != "episode":
        raise ValueError(f"Unsupported normalize mode: {mode}")
    lo = float(np.min(c))
    hi = float(np.max(c))
    if hi > lo:
        return (c - lo) / (hi - lo)
    return np.ones_like(c) if hi > 0 else np.zeros_like(c)


def ema_smooth(signal: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    a = float(alpha)
    for i in range(1, x.shape[0]):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def hysteresis_contact_binary(
    signal_1d: np.ndarray, threshold_high: float, threshold_low: float
) -> np.ndarray:
    x = np.asarray(signal_1d, dtype=np.float64)
    out = np.zeros((x.shape[0],), dtype=np.int8)
    on = False
    for i in range(x.shape[0]):
        s = x[i]
        if on:
            if s < threshold_low:
                on = False
        else:
            if s > threshold_high:
                on = True
        out[i] = 1 if on else 0
    return out


def build_pre_mask(con_mask: np.ndarray, precontact_frames: int) -> np.ndarray:
    c = np.asarray(con_mask, dtype=np.int8).reshape(-1)
    pre = np.zeros_like(c, dtype=np.int8)
    n = c.shape[0]
    for i in range(n):
        if c[i] == 1 and (i == 0 or c[i - 1] == 0):
            a = max(0, i - int(precontact_frames))
            pre[a:i] = 1
    return pre


def build_labels(con_mask: np.ndarray, pre_mask: np.ndarray) -> np.ndarray:
    labels = np.zeros((con_mask.shape[0],), dtype=np.int8)  # 0=free
    labels[con_mask == 1] = 2  # contact has priority
    labels[(pre_mask == 1) & (con_mask == 0)] = 1  # precontact
    return labels


def smooth_episode_signal(
    parquet_path: Path,
    contact_keys: list[str],
    normalize_mode: str,
    ema_alpha: float,
) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=contact_keys)

    sigs: list[np.ndarray] = []
    for key in contact_keys:
        if key not in df.columns:
            raise RuntimeError(f"Missing key in {parquet_path}: {key}")
        rows = [np.asarray(v, dtype=np.float64).reshape(-1) for v in df[key].tolist()]
        sigs.append(to_1d_contact(np.vstack(rows)))

    if not sigs:
        raise RuntimeError(f"No contact signals loaded from {parquet_path}")

    contact_raw = np.sum(np.vstack(sigs), axis=0)
    contact_norm = normalize_contact_1d(contact_raw, mode=normalize_mode)
    return ema_smooth(contact_norm, alpha=ema_alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode subset: all | 0-99 | 0,1,2",
    )
    parser.add_argument(
        "--contact-keys",
        type=str,
        default="observation.force.qfrc_constraint_arm_l2",
        help="Comma-separated contact keys",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=["episode", "none"],
        default="episode",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--precontact-frames", type=int, default=30)
    parser.add_argument(
        "--sweep-pairs",
        type=str,
        default=None,
        help="Override defaults. Format: high:low,high:low,...",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/phase_threshold_sweeps",
        help="Directory for CSV/JSON outputs",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset info file: {info_path}")
    with open(info_path, "r") as f:
        info = json.load(f)
    total_episodes = int(info["total_episodes"])

    episodes = parse_episodes_spec(args.episodes, total_episodes)
    contact_keys = [k.strip() for k in args.contact_keys.split(",") if k.strip()]
    sweep_pairs = parse_sweep_pairs(args.sweep_pairs)

    sweep_stats: list[SweepStats] = []
    for hi, lo in sweep_pairs:
        eff_hi = float(hi)
        eff_lo = float(lo)
        swapped = False
        if eff_hi < eff_lo:
            eff_hi, eff_lo = eff_lo, eff_hi
            swapped = True
        sweep_stats.append(
            SweepStats(
                input_high=float(hi),
                input_low=float(lo),
                eff_high=eff_hi,
                eff_low=eff_lo,
                swapped=swapped,
            )
        )

    skipped: list[dict] = []
    for i, ep in enumerate(episodes):
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(episodes):
            print(f"[sweep] episode {i + 1}/{len(episodes)} (ep={ep:06d})")

        try:
            parquet = find_episode_parquet(dataset_root, ep)
            signal = smooth_episode_signal(
                parquet_path=parquet,
                contact_keys=contact_keys,
                normalize_mode=args.normalize_mode,
                ema_alpha=args.ema_alpha,
            )
            for st in sweep_stats:
                con = hysteresis_contact_binary(
                    signal_1d=signal,
                    threshold_high=st.eff_high,
                    threshold_low=st.eff_low,
                )
                pre = build_pre_mask(
                    con_mask=con, precontact_frames=args.precontact_frames
                )
                labels = build_labels(con_mask=con, pre_mask=pre)
                binc = np.bincount(labels, minlength=3)
                st.add_counts(
                    free_n=int(binc[0]), pre_n=int(binc[1]), contact_n=int(binc[2])
                )
                st.episodes_used += 1
        except Exception as exc:
            skipped.append({"episode": int(ep), "error": str(exc)})

    rows = [st.as_dict() for st in sweep_stats]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"phase_threshold_sweep_{stamp}.csv"
    json_path = out_dir / f"phase_threshold_sweep_{stamp}.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_high",
                "input_low",
                "effective_high",
                "effective_low",
                "swapped_high_low",
                "episodes_used",
                "frames_total",
                "free_count",
                "precontact_count",
                "contact_count",
                "free_ratio",
                "precontact_ratio",
                "contact_ratio",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = {
        "dataset": str(dataset_root),
        "episodes_requested": len(episodes),
        "episodes_used_per_pair": [r["episodes_used"] for r in rows],
        "contact_keys": contact_keys,
        "normalize_mode": args.normalize_mode,
        "ema_alpha": float(args.ema_alpha),
        "precontact_frames": int(args.precontact_frames),
        "results": rows,
        "skipped_episodes": skipped,
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSweep results (ratios):")
    for r in rows:
        print(
            f"hi={r['input_high']:.3f}, lo={r['input_low']:.3f}"
            f" -> free={r['free_ratio']:.4f}, pre={r['precontact_ratio']:.4f}, contact={r['contact_ratio']:.4f}"
            f"{' [swapped]' if r['swapped_high_low'] else ''}"
        )
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    if skipped:
        print(f"Skipped episodes: {len(skipped)}")


if __name__ == "__main__":
    main()
