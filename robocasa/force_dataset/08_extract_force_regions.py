#!/usr/bin/env python3
"""
Region extraction pipeline (no weighted sampling).

Implements, per episode:
1) load force/contact channel(s)
2) collapse to 1D abs-sum magnitude + NaN/Inf cleanup
3) normalize to [0, 1] (optional)
4) EMA smoothing
5) contact mask via single threshold OR hysteresis
6) final labels: 0=free, 1=precontact, 2=contact

Outputs are written to artifacts/ by default (JSON + NPZ), without mutating dataset files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


LABEL_FREE = 0
LABEL_PRE = 1
LABEL_CONTACT = 2


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


def to_1d_contact(x: np.ndarray) -> np.ndarray:
    contact = np.asarray(x, dtype=np.float64)
    if contact.ndim > 1:
        contact = np.abs(contact).sum(axis=-1)
    contact = np.nan_to_num(contact, nan=0.0, posinf=0.0, neginf=0.0)
    return contact.reshape(-1)


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


def ema_smooth(contact: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(contact, dtype=np.float64)
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


def contact_binary(
    signal_1d: np.ndarray,
    threshold: float,
    hysteresis_enabled: bool,
    threshold_high: float | None,
    threshold_low: float | None,
) -> np.ndarray:
    x = np.asarray(signal_1d, dtype=np.float64)
    if not hysteresis_enabled:
        return (x > float(threshold)).astype(np.int8)
    hi = float(threshold if threshold_high is None else threshold_high)
    lo = float(threshold if threshold_low is None else threshold_low)
    if hi < lo:
        hi, lo = lo, hi
    return hysteresis_contact_binary(x, threshold_high=hi, threshold_low=lo)


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
    labels = np.zeros((con_mask.shape[0],), dtype=np.int8)
    labels[(con_mask == 1) & (pre_mask == 0)] = LABEL_CONTACT
    labels[pre_mask == 1] = LABEL_PRE
    return labels


def read_key_as_rows(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df.columns:
        raise RuntimeError(f"Missing key in episode parquet: {key}")
    rows = [np.asarray(v, dtype=np.float64).reshape(-1) for v in df[key].tolist()]
    if not rows:
        raise RuntimeError(f"No rows found for key: {key}")
    return np.vstack(rows)


def compute_regions_from_df(
    df: pd.DataFrame,
    contact_keys: list[str],
    normalize_mode: str,
    ema_alpha: float,
    contact_threshold: float,
    hysteresis_enabled: bool,
    threshold_high: float | None,
    threshold_low: float | None,
    precontact_frames: int,
) -> dict[str, np.ndarray]:
    sigs: list[np.ndarray] = []
    for key in contact_keys:
        sigs.append(to_1d_contact(read_key_as_rows(df, key)))
    if not sigs:
        raise RuntimeError("No contact keys were loaded.")

    # Aggregate channels by sum after 1D conversion.
    contact_raw = np.sum(np.vstack(sigs), axis=0)
    contact_norm = normalize_contact_1d(contact_raw, mode=normalize_mode)
    contact_smooth = ema_smooth(contact_norm, alpha=ema_alpha)

    con_mask = contact_binary(
        signal_1d=contact_smooth,
        threshold=contact_threshold,
        hysteresis_enabled=hysteresis_enabled,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
    )
    pre_mask = build_pre_mask(con_mask=con_mask, precontact_frames=precontact_frames)
    labels = build_labels(con_mask=con_mask, pre_mask=pre_mask)

    return {
        "contact_raw": contact_raw,
        "contact_norm": contact_norm,
        "contact_smooth": contact_smooth,
        "con_mask": con_mask.astype(np.int8),
        "pre_mask": pre_mask.astype(np.int8),
        "labels": labels.astype(np.int8),
    }


def counts(labels: np.ndarray) -> dict[str, int]:
    l = np.asarray(labels, dtype=np.int8)
    return {
        "free": int(np.sum(l == LABEL_FREE)),
        "pre_contact": int(np.sum(l == LABEL_PRE)),
        "contact": int(np.sum(l == LABEL_CONTACT)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument(
        "--episodes", type=str, default="0", help="Episode spec: all | 0-9 | 0,1,2"
    )
    parser.add_argument(
        "--contact-keys",
        type=str,
        default="observation.force.qfrc_constraint_arm_l2",
        help="Comma-separated parquet keys to combine",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=["episode", "none"],
        default="episode",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--contact-threshold", type=float, default=0.3)
    parser.add_argument(
        "--hysteresis-enabled",
        dest="hysteresis_enabled",
        action="store_true",
        default=True,
        help="Enable hysteresis thresholding (default: enabled)",
    )
    parser.add_argument(
        "--no-hysteresis",
        dest="hysteresis_enabled",
        action="store_false",
        help="Disable hysteresis and use single-threshold only",
    )
    parser.add_argument("--threshold-high", type=float, default=None)
    parser.add_argument("--threshold-low", type=float, default=None)
    parser.add_argument("--precontact-frames", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/force_region_extraction",
        help="Output directory for JSON/NPZ",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Invalid dataset root: {dataset_root}")
    total = get_total_episodes(dataset_root)
    episodes = parse_episodes_spec(args.episodes, total)
    keys = [k.strip() for k in args.contact_keys.split(",") if k.strip()]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset": str(dataset_root),
        "episodes": episodes,
        "contact_keys": keys,
        "normalize_mode": args.normalize_mode,
        "ema_alpha": float(args.ema_alpha),
        "contact_threshold": float(args.contact_threshold),
        "hysteresis_enabled": bool(args.hysteresis_enabled),
        "threshold_high": args.threshold_high,
        "threshold_low": args.threshold_low,
        "precontact_frames": int(args.precontact_frames),
        "results": [],
    }

    ep_iter: Any = episodes
    pbar: Any | None = None
    if tqdm is not None:
        pbar = tqdm(episodes, desc="Extract regions", unit="ep", dynamic_ncols=True)
        ep_iter = pbar

    for ep in ep_iter:
        p = find_episode_parquet(dataset_root, ep)
        df = pd.read_parquet(p)
        out = compute_regions_from_df(
            df=df,
            contact_keys=keys,
            normalize_mode=args.normalize_mode,
            ema_alpha=args.ema_alpha,
            contact_threshold=args.contact_threshold,
            hysteresis_enabled=args.hysteresis_enabled,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
            precontact_frames=args.precontact_frames,
        )
        npz_path = out_dir / f"episode_{ep:06d}_regions.npz"
        json_path = out_dir / f"episode_{ep:06d}_regions.json"
        np.savez_compressed(npz_path, **out)

        cnt = counts(out["labels"])
        dump_json(
            json_path,
            {
                "episode": ep,
                "parquet": str(p),
                "npz": str(npz_path),
                "counts": cnt,
                "signal_len": int(out["labels"].shape[0]),
            },
        )
        summary["results"].append(
            {
                "episode": ep,
                "npz": str(npz_path),
                "json": str(json_path),
                "counts": cnt,
            }
        )

    if pbar is not None:
        pbar.close()

    summary_path = out_dir / "region_extraction_summary.json"
    dump_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
