#!/usr/bin/env python3
"""
Generate single-threshold vs hysteresis example figure for one episode.

Figure layout:
1) Signal with thresholds
2) Single-threshold binary contact mask
3) Hysteresis binary contact mask
4) Final regions (free unshaded, precontact green, contact red)
   with single dotted black border lines at state transitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_episode_parquet(dataset_root: Path, episode: int) -> Path:
    matches = list(dataset_root.glob(f"data/*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode_{episode:06d} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple parquet files for episode {episode}: {matches}")
    return matches[0]


def scalar_from_cell(x, dtype=np.float64):
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    if arr.size == 0:
        return np.nan
    return arr[0]


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
    labels = np.zeros((con_mask.shape[0],), dtype=np.int8)  # free
    labels[con_mask == 1] = 2  # contact
    labels[(pre_mask == 1) & (con_mask == 0)] = 1  # precontact
    return labels


def find_transitions(labels: np.ndarray) -> np.ndarray:
    l = np.asarray(labels, dtype=np.int8).reshape(-1)
    if l.size <= 1:
        return np.asarray([], dtype=np.int64)
    return np.where(l[1:] != l[:-1])[0] + 1


def load_signal(df: pd.DataFrame, contact_keys: list[str]) -> np.ndarray:
    sigs: list[np.ndarray] = []
    for key in contact_keys:
        if key not in df.columns:
            raise RuntimeError(f"Missing key in parquet: {key}")
        rows = [np.asarray(v, dtype=np.float64).reshape(-1) for v in df[key].tolist()]
        sigs.append(to_1d_contact(np.vstack(rows)))
    if not sigs:
        raise RuntimeError("No contact keys loaded.")
    return np.sum(np.vstack(sigs), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--contact-keys",
        type=str,
        default="observation.force.qfrc_constraint_arm_l2",
        help="Comma-separated keys to aggregate",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=["episode", "none"],
        default="episode",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument(
        "--single-threshold",
        type=float,
        default=0.3,
        help="Threshold for single-threshold mask",
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=0.3,
        help="Hysteresis high threshold",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.1,
        help="Hysteresis low threshold",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default=None,
        help="Output PNG path. Default artifacts/hysteresis_example_epXXXXXX.png",
    )
    parser.add_argument(
        "--precontact-frames",
        type=int,
        default=30,
        help="Frames before each contact onset to mark as precontact",
    )
    args = parser.parse_args()

    keys = [k.strip() for k in args.contact_keys.split(",") if k.strip()]
    dataset_root = Path(args.dataset).resolve()
    parquet = find_episode_parquet(dataset_root, args.episode)
    df = pd.read_parquet(parquet)

    raw = load_signal(df, keys)
    norm = normalize_contact_1d(raw, mode=args.normalize_mode)
    smooth = ema_smooth(norm, alpha=args.ema_alpha)

    single = (smooth > float(args.single_threshold)).astype(np.int8)
    hys = hysteresis_contact_binary(
        smooth,
        threshold_high=float(args.threshold_high),
        threshold_low=float(args.threshold_low),
    )
    pre = build_pre_mask(hys, precontact_frames=args.precontact_frames)
    labels = build_labels(con_mask=hys, pre_mask=pre)  # 0 free, 1 pre, 2 contact
    transitions = find_transitions(labels)

    t = np.arange(smooth.shape[0])
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, dpi=130)

    # Top: signal + thresholds
    ax0 = axes[0]
    ax0.plot(t, smooth, color="#1f77b4", linewidth=1.8, label="smoothed signal")
    ax0.axhline(
        float(args.single_threshold),
        color="#2ca02c",
        linestyle="--",
        linewidth=1.4,
        label=f"single threshold={args.single_threshold:g}",
    )
    ax0.axhline(
        float(args.threshold_high),
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"hys high={args.threshold_high:g}",
    )
    ax0.axhline(
        float(args.threshold_low),
        color="#9467bd",
        linestyle="--",
        linewidth=1.4,
        label=f"hys low={args.threshold_low:g}",
    )
    ax0.set_ylabel("signal")
    ax0.set_title(
        f"Episode {args.episode:06d} | signal={','.join(keys)} | normalize={args.normalize_mode} | ema={args.ema_alpha:g}"
    )
    ax0.grid(alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)

    # Middle: single threshold mask
    ax1 = axes[1]
    ax1.step(t, single, where="post", color="#2ca02c", linewidth=1.8)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.set_ylabel("single")
    ax1.set_title("Single-threshold binary contact")
    ax1.grid(alpha=0.2)

    # Bottom: hysteresis mask
    ax2 = axes[2]
    ax2.step(t, hys, where="post", color="#d62728", linewidth=1.8)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_ylabel("hysteresis")
    ax2.set_title("Hysteresis binary contact")
    ax2.grid(alpha=0.2)

    # Final regions panel (free unshaded, pre green, contact red)
    ax3 = axes[3]
    ax3.plot(t, smooth, color="#1f77b4", linewidth=1.5, label="smoothed signal")
    for i in range(labels.shape[0]):
        if labels[i] == 1:  # precontact
            ax3.axvspan(i, i + 1, color="#2ca02c", alpha=0.30, linewidth=0)
        elif labels[i] == 2:  # contact
            ax3.axvspan(i, i + 1, color="#d62728", alpha=0.30, linewidth=0)
        # free (0) intentionally unshaded
    for idx in transitions:
        # Single dotted border per boundary.
        ax3.axvline(
            idx,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            zorder=6,
        )
    ax3.set_ylabel("signal")
    ax3.set_xlabel("frame")
    ax3.set_title(
        f"Regions: free(unshaded), precontact(green), contact(red), pre_frames={args.precontact_frames}"
    )
    ax3.grid(alpha=0.2)

    plt.tight_layout()

    output = (
        Path(args.output_fig)
        if args.output_fig
        else Path("artifacts") / f"hysteresis_example_ep{args.episode:06d}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print(f"Saved figure: {output}")
    print(f"Parquet: {parquet}")


if __name__ == "__main__":
    main()
