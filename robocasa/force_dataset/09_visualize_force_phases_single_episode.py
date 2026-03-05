#!/usr/bin/env python3
"""
Create a phase-shaded force dashboard video for one episode.

Inputs:
- Existing camera video from dataset/videos/
- Existing phase labels from parquet:
  diagnostic.force_phase.label
- Existing force signals from parquet:
  observation.force.qfrc_constraint_arm_l2
  diagnostic.force_phase.signal_smoothed (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


KEY_LABEL = "diagnostic.force_phase.label"
KEY_SIGNAL_RAW = "observation.force.qfrc_constraint_arm_l2"
KEY_SIGNAL_SMOOTH = "diagnostic.force_phase.signal_smoothed"

LABEL_NAMES = {
    -1: "invalid",
    0: "free_motion",
    1: "precontact",
    2: "contact",
}

LABEL_COLORS = {
    -1: "#bdbdbd",  # gray
    0: "#cfe8ff",  # light blue
    1: "#d8bf00",  # darker yellow
    2: "#d64545",  # darker red
}


def scalar_from_cell(x, dtype=np.float64):
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    if arr.size == 0:
        return np.nan
    return arr[0]


def find_episode_parquet(dataset_root: Path, episode: int) -> Path:
    matches = list(dataset_root.glob(f"data/*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode_{episode:06d} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple parquet files for episode {episode}: {matches}")
    return matches[0]


def find_episode_video(dataset_root: Path, episode: int, camera_key: str) -> Path:
    rel = f"videos/*/observation.images.{camera_key}/episode_{episode:06d}.mp4"
    matches = list(dataset_root.glob(rel))
    if not matches:
        raise FileNotFoundError(
            f"No video found for episode_{episode:06d}, camera={camera_key} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple videos found for episode {episode}, camera={camera_key}: {matches}"
        )
    return matches[0]


def find_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if labels.size == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for i in range(1, labels.size):
        if int(labels[i]) != current:
            runs.append((start, i, current))
            start = i
            current = int(labels[i])
    runs.append((start, labels.size, current))
    return runs


def load_arrays(parquet_path: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    for key in [KEY_LABEL, KEY_SIGNAL_RAW]:
        if key not in df.columns:
            raise RuntimeError(f"Missing required key {key} in {parquet_path}")

    labels = np.asarray(
        [int(scalar_from_cell(v, dtype=np.int64)) for v in df[KEY_LABEL].tolist()],
        dtype=np.int64,
    )
    raw = np.asarray(
        [
            float(scalar_from_cell(v, dtype=np.float64))
            for v in df[KEY_SIGNAL_RAW].tolist()
        ],
        dtype=np.float64,
    )

    smooth = None
    if KEY_SIGNAL_SMOOTH in df.columns:
        smooth = np.asarray(
            [
                float(scalar_from_cell(v, dtype=np.float64))
                for v in df[KEY_SIGNAL_SMOOTH].tolist()
            ],
            dtype=np.float64,
        )

    return {"labels": labels, "raw": raw, "smooth": smooth}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Augmented dataset root"
    )
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--camera-key",
        type=str,
        default="robot0_agentview_right",
        help="Camera key suffix from videos folder",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video path. Default: artifacts/force_phase_dashboard_epXXXXXX.mp4",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override output fps")
    parser.add_argument(
        "--line-mode",
        type=str,
        default="both",
        choices=["raw", "smooth", "both"],
        help="Which force signal lines to draw",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional frame cap"
    )
    parser.add_argument(
        "--shade-alpha",
        type=float,
        default=0.60,
        help="Opacity for shaded phase regions",
    )
    parser.add_argument(
        "--transition-bar-width",
        type=float,
        default=2.0,
        help="Line width for phase transition bars",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    output_video = (
        Path(args.output_video)
        if args.output_video
        else Path("artifacts") / f"force_phase_dashboard_ep{args.episode:06d}.mp4"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = find_episode_parquet(dataset_root, args.episode)
    video_path = find_episode_video(dataset_root, args.episode, args.camera_key)
    arr = load_arrays(parquet_path)
    labels = arr["labels"]
    raw = arr["raw"]
    smooth = arr["smooth"]

    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = float(args.fps) if args.fps is not None else float(meta.get("fps", 20.0))

    total_t = len(labels)
    if args.max_frames is not None:
        total_t = min(total_t, int(args.max_frames))

    # Setup dashboard
    fig, (ax_cam, ax_plot) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    ax_cam.axis("off")
    ax_cam.set_title(args.camera_key)

    x = np.arange(total_t)
    y_candidates = [raw[:total_t][np.isfinite(raw[:total_t])]]
    if smooth is not None:
        y_candidates.append(smooth[:total_t][np.isfinite(smooth[:total_t])])
    y_cat = np.concatenate([y for y in y_candidates if y.size > 0], axis=0)
    if y_cat.size == 0:
        ymin, ymax = -1.0, 1.0
    else:
        ymin = float(np.nanmin(y_cat))
        ymax = float(np.nanmax(y_cat))
        if abs(ymax - ymin) < 1e-9:
            ymin -= 1.0
            ymax += 1.0

    # Shade phase regions
    legend_labels_added: set[int] = set()
    legend_patches = []
    runs = find_runs(labels[:total_t])
    for start, end, lab in runs:
        # Do not shade free motion.
        if lab == 0:
            continue
        color = LABEL_COLORS.get(lab, "#eeeeee")
        name = LABEL_NAMES.get(lab, f"label_{lab}")
        ax_plot.axvspan(
            start, end - 1, color=color, alpha=float(args.shade_alpha), linewidth=0.0
        )
        if lab not in legend_labels_added:
            legend_labels_added.add(lab)
            legend_patches.append(Patch(facecolor=color, edgecolor="none", label=name))

    # Draw bars at every label transition.
    for idx in range(1, total_t):
        if int(labels[idx]) != int(labels[idx - 1]):
            ax_plot.axvline(
                idx,
                color="black",
                linestyle="-",
                linewidth=float(args.transition_bar_width),
                alpha=0.85,
                zorder=5,
            )

    raw_line = None
    smooth_line = None
    if args.line_mode in ("raw", "both"):
        (raw_line,) = ax_plot.plot(
            x, raw[:total_t], color="#0b4f9c", linewidth=1.5, label="arm_l2_raw"
        )
    if args.line_mode in ("smooth", "both") and smooth is not None:
        (smooth_line,) = ax_plot.plot(
            x,
            smooth[:total_t],
            color="#d62728",
            linewidth=1.5,
            linestyle="--",
            label="arm_l2_smoothed",
        )

    cursor = ax_plot.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_plot.set_title("Force Magnitude + Contact/Precontact Regions")
    ax_plot.set_xlabel("Frame")
    ax_plot.set_ylabel("qfrc_constraint_arm_l2")
    ax_plot.set_xlim(0, max(total_t - 1, 1))
    ax_plot.set_ylim(ymin, ymax)
    ax_plot.grid(True, alpha=0.25)

    handles = []
    if raw_line is not None:
        handles.append(raw_line)
    if smooth_line is not None:
        handles.append(smooth_line)
    handles.extend(legend_patches)
    ax_plot.legend(handles=handles, loc="upper right", fontsize=8)

    text = ax_plot.text(
        0.01,
        0.98,
        "",
        transform=ax_plot.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
    )

    writer = imageio.get_writer(output_video, fps=fps)
    try:
        rendered = 0
        for t in range(total_t):
            try:
                frame = reader.get_data(t)
            except Exception:
                break
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=2)

            ax_cam.clear()
            ax_cam.imshow(frame)
            ax_cam.axis("off")
            ax_cam.set_title(args.camera_key)

            cursor.set_xdata([t, t])
            lab = int(labels[t])
            phase_name = LABEL_NAMES.get(lab, str(lab))

            raw_v = raw[t] if np.isfinite(raw[t]) else np.nan
            smooth_v = (
                smooth[t] if (smooth is not None and np.isfinite(smooth[t])) else np.nan
            )
            lines = [
                f"episode: {args.episode:06d}",
                f"frame: {t}",
                f"phase: {phase_name} ({lab})",
                f"raw: {raw_v:.5f}" if np.isfinite(raw_v) else "raw: nan",
            ]
            if smooth is not None:
                lines.append(
                    f"smooth: {smooth_v:.5f}"
                    if np.isfinite(smooth_v)
                    else "smooth: nan"
                )
            text.set_text("\n".join(lines))

            fig.canvas.draw()
            out = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(out)
            rendered += 1
    finally:
        writer.close()
        reader.close()
        plt.close(fig)

    print(f"Saved video: {output_video}")
    print(f"Frames rendered: {rendered}")
    print(f"Parquet: {parquet_path}")
    print(f"Video source: {video_path}")


if __name__ == "__main__":
    main()
