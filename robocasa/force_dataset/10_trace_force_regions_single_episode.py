#!/usr/bin/env python3
"""
Render a single video with:
1) rollout camera replay (left panel)
2) force-region graph with cursor (right panel)

Region rules:
- free motion: unshaded
- precontact: green shading
- contact: red shading
- single dotted black boundary on region transitions
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("Missing dependency: matplotlib.") from exc

try:
    import imageio.v2 as imageio
except Exception as exc:
    raise SystemExit(
        "Missing dependency: imageio. Install imageio imageio-ffmpeg."
    ) from exc


def find_episode_parquet(dataset_root: Path, episode: int) -> Path:
    matches = list(dataset_root.glob(f"data/*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode_{episode:06d} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple parquet files for episode {episode}: {matches}")
    return matches[0]


ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def get_env_metadata(dataset_dir: Path) -> dict:
    dataset_meta_path = dataset_dir / "extras" / "dataset_meta.json"
    with open(dataset_meta_path, "r") as f:
        dataset_meta = json.load(f)
    return dataset_meta["env_args"]


def get_episode_states(dataset_dir: Path, ep_num: int) -> np.ndarray:
    states_path = dataset_dir / "extras" / f"episode_{ep_num:06d}" / "states.npz"
    return np.load(states_path)["states"]


def get_episode_model_xml(dataset_dir: Path, ep_num: int) -> str:
    model_xml_path = dataset_dir / "extras" / f"episode_{ep_num:06d}" / "model.xml.gz"
    with gzip.open(model_xml_path, "rb") as f:
        return f.read().decode("utf-8")


def get_episode_meta(dataset_dir: Path, ep_num: int) -> dict:
    ep_meta_path = dataset_dir / "extras" / f"episode_{ep_num:06d}" / "ep_meta.json"
    with open(ep_meta_path, "r") as f:
        return json.load(f)


def get_modality_dict(dataset_dir: Path) -> dict:
    modality_path = dataset_dir / "meta" / "modality.json"
    with open(modality_path, "r") as f:
        return json.load(f)


def reorder_lerobot_action(action_lerobot: np.ndarray, dataset_dir: Path) -> np.ndarray:
    modality_dict = get_modality_dict(dataset_dir)
    action_info = modality_dict["action"]
    sorted_action_keys = sorted(
        action_info.keys(), key=lambda k: action_info[k]["start"]
    )
    reordered_action = np.zeros_like(action_lerobot)
    for key in sorted_action_keys:
        lerobot_start = action_info[key]["start"]
        lerobot_end = action_info[key]["end"]
        hdf5_start, hdf5_end = ACTION_KEY_ORDERING_HDF5[key]
        reordered_action[:, hdf5_start:hdf5_end] = action_lerobot[
            :, lerobot_start:lerobot_end
        ]
    return reordered_action


def get_episode_actions(dataset_dir: Path, ep_num: int) -> np.ndarray:
    data_file = find_episode_parquet(dataset_dir, ep_num)
    df = pd.read_parquet(data_file)
    raw_action = np.stack(df["action"].to_list())
    return reorder_lerobot_action(raw_action, dataset_dir)


def reset_to(env, state, robosuite_mod):
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
        robosuite_version_id = int(robosuite_mod.__version__.split(".")[1])
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


def make_env(dataset_root: Path, robosuite_mod):
    env_meta = get_env_metadata(dataset_root)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    return robosuite_mod.make(**env_kwargs)


def to_1d_contact(contact: np.ndarray) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float64)
    if c.ndim > 1:
        c = np.abs(c).sum(axis=-1)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return c.reshape(-1)


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
    for i in range(c.shape[0]):
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


def find_episode_video(
    dataset_root: Path, episode: int, camera_name: str
) -> Path | None:
    candidates = [camera_name]
    if not camera_name.startswith("observation.images."):
        candidates.append(f"observation.images.{camera_name}")
    for cam_dir in candidates:
        matches = sorted(
            dataset_root.glob(f"videos/*/{cam_dir}/episode_{episode:06d}.mp4")
        )
        if matches:
            return matches[0]
    return None


def collect_rollout_frames_from_dataset_video(
    dataset_root: Path,
    episode: int,
    camera_name: str,
    t_horizon: int,
) -> list[np.ndarray]:
    video_path = find_episode_video(dataset_root, episode, camera_name)
    if video_path is None:
        raise FileNotFoundError(
            f"No episode video found for episode={episode}, camera={camera_name}"
        )

    print(f"[rollout] source=dataset_video path={video_path}")
    print(f"[rollout] loading up to {t_horizon} frames from MP4...")
    frames: list[np.ndarray] = []
    reader = imageio.get_reader(str(video_path))
    try:
        for t, frame in enumerate(reader):
            if t >= t_horizon:
                break
            frames.append(np.asarray(frame, dtype=np.uint8))
            if (t + 1) % 25 == 0 or (t + 1) == t_horizon:
                print(f"[rollout] {t + 1}/{t_horizon}")
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"Loaded zero frames from video: {video_path}")
    return frames


def collect_rollout_frames_from_sim(
    dataset_root: Path,
    episode: int,
    camera_name: str,
    width: int,
    height: int,
    t_horizon: int,
) -> list[np.ndarray]:
    try:
        import robosuite as robosuite_mod
    except Exception as exc:
        raise RuntimeError(
            "Simulator replay requested but robosuite import failed."
        ) from exc

    env = make_env(dataset_root, robosuite_mod)
    try:
        states = get_episode_states(dataset_root, episode)
        actions = get_episode_actions(dataset_root, episode)
        if len(states) != len(actions):
            raise RuntimeError(
                f"states/actions mismatch for episode {episode}: {len(states)} vs {len(actions)}"
            )
        if t_horizon > len(states):
            raise RuntimeError(
                f"Requested horizon {t_horizon} > episode length {len(states)}."
            )

        initial_state = {
            "states": states[0],
            "model": get_episode_model_xml(dataset_root, episode),
            "ep_meta": json.dumps(get_episode_meta(dataset_root, episode)),
        }
        reset_to(env, initial_state, robosuite_mod)

        frames: list[np.ndarray] = []
        print(f"[rollout] source=sim_replay camera={camera_name}")
        print(f"[rollout] rendering {t_horizon} frames...")
        for t in range(t_horizon):
            reset_to(env, {"states": states[t]}, robosuite_mod)
            env.step(actions[t])
            frame = env.sim.render(height=height, width=width, camera_name=camera_name)[
                ::-1
            ]
            frames.append(np.asarray(frame, dtype=np.uint8))
            if (t + 1) % 25 == 0 or (t + 1) == t_horizon:
                print(f"[rollout] {t + 1}/{t_horizon}")

        return frames
    finally:
        env.close()


def collect_rollout_frames(
    dataset_root: Path,
    episode: int,
    camera_name: str,
    width: int,
    height: int,
    t_horizon: int,
    rollout_source: str,
) -> list[np.ndarray]:
    if rollout_source == "dataset_video":
        return collect_rollout_frames_from_dataset_video(
            dataset_root=dataset_root,
            episode=episode,
            camera_name=camera_name,
            t_horizon=t_horizon,
        )
    if rollout_source == "sim_replay":
        return collect_rollout_frames_from_sim(
            dataset_root=dataset_root,
            episode=episode,
            camera_name=camera_name,
            width=width,
            height=height,
            t_horizon=t_horizon,
        )
    # auto
    try:
        return collect_rollout_frames_from_dataset_video(
            dataset_root=dataset_root,
            episode=episode,
            camera_name=camera_name,
            t_horizon=t_horizon,
        )
    except Exception as exc:
        print(
            f"[rollout] dataset_video unavailable ({exc}); falling back to sim_replay"
        )
        return collect_rollout_frames_from_sim(
            dataset_root=dataset_root,
            episode=episode,
            camera_name=camera_name,
            width=width,
            height=height,
            t_horizon=t_horizon,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--contact-keys",
        type=str,
        default="observation.force.qfrc_constraint_arm_l2",
        help="Comma-separated parquet keys to aggregate",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=["episode", "none"],
        default="episode",
        help="Kept for compatibility; script enforces normalized hysteresis workflow",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--threshold-high", type=float, default=0.3)
    parser.add_argument("--threshold-low", type=float, default=0.1)
    parser.add_argument("--precontact-frames", type=int, default=30)
    parser.add_argument(
        "--camera-name",
        type=str,
        default="robot0_eye_in_hand",
        help="Ignored at runtime; script forces robot0_eye_in_hand",
    )
    parser.add_argument(
        "--rollout-source",
        type=str,
        choices=["auto", "dataset_video", "sim_replay"],
        default="auto",
        help="Where rollout frames come from",
    )
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Default: artifacts/force_regions_rollout_trace_epXXXXXX.mp4",
    )
    args = parser.parse_args()
    args.camera_name = "robot0_eye_in_hand"

    dataset_root = Path(args.dataset).resolve()
    print(f"[start] dataset={dataset_root}")
    print(f"[start] episode={args.episode}")
    parquet = find_episode_parquet(dataset_root, args.episode)
    print(f"[start] parquet={parquet}")
    df = pd.read_parquet(parquet)

    keys = [k.strip() for k in args.contact_keys.split(",") if k.strip()]
    raw = load_signal(df, keys)
    if args.normalize_mode != "episode":
        print("[cfg] normalize-mode overridden to 'episode' (always normalized)")
    norm = normalize_contact_1d(raw, mode="episode")
    smooth = ema_smooth(norm, alpha=args.ema_alpha)

    con_mask = hysteresis_contact_binary(
        smooth,
        threshold_high=float(args.threshold_high),
        threshold_low=float(args.threshold_low),
    )

    pre_mask = build_pre_mask(
        con_mask=con_mask, precontact_frames=args.precontact_frames
    )
    labels = build_labels(con_mask=con_mask, pre_mask=pre_mask)
    transitions = find_transitions(labels)

    t_horizon = len(smooth)
    if args.max_frames is not None:
        t_horizon = min(t_horizon, int(args.max_frames))
    print(f"[prep] horizon={t_horizon}")

    frames = collect_rollout_frames(
        dataset_root=dataset_root,
        episode=args.episode,
        camera_name=args.camera_name,
        width=int(args.width),
        height=int(args.height),
        t_horizon=t_horizon,
        rollout_source=args.rollout_source,
    )
    if len(frames) < t_horizon:
        print(
            f"[rollout] warning: got {len(frames)} frames, truncating signal timeline."
        )
        t_horizon = len(frames)
        labels = labels[:t_horizon]
        transitions = find_transitions(labels)

    x = np.arange(t_horizon)
    y = smooth[:t_horizon]
    finite = np.isfinite(y)
    if not finite.any():
        ymin, ymax = -1.0, 1.0
    else:
        ymin = float(np.min(y[finite]))
        ymax = float(np.max(y[finite]))
        if abs(ymax - ymin) < 1e-9:
            ymin -= 1.0
            ymax += 1.0

    fig, (ax_cam, ax_sig) = plt.subplots(
        1,
        2,
        figsize=(21.0, 4.8),
        dpi=130,
        gridspec_kw={"width_ratios": [1.0, 3.4]},
    )
    cam_artist = ax_cam.imshow(frames[0])
    ax_cam.set_title(f"Rollout Camera: {args.camera_name}")
    ax_cam.axis("off")

    ax_sig.plot(x, y, color="#1f77b4", linewidth=1.9, label="smoothed signal")

    for i in range(t_horizon):
        if labels[i] == 1:
            ax_sig.axvspan(i, i + 1, color="#2ca02c", alpha=0.35, linewidth=0)
        elif labels[i] == 2:
            ax_sig.axvspan(i, i + 1, color="#d62728", alpha=0.35, linewidth=0)

    for idx in transitions:
        if idx < t_horizon:
            ax_sig.axvline(
                idx, color="black", linestyle=":", linewidth=1.2, alpha=0.95, zorder=6
            )

    cursor = ax_sig.axvline(0, color="black", linestyle="--", linewidth=1.8)
    ax_sig.set_xlim(0, max(t_horizon - 1, 1))
    ax_sig.set_ylim(ymin, ymax)
    ax_sig.set_xlabel("frame")
    ax_sig.set_ylabel("signal")
    ax_sig.set_title("Force Regions")
    ax_sig.grid(alpha=0.22)
    ax_sig.legend(loc="upper right", fontsize=9)
    txt = ax_sig.text(
        0.01,
        0.98,
        "",
        transform=ax_sig.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )

    output = (
        Path(args.output_video)
        if args.output_video
        else Path("artifacts") / f"force_regions_rollout_trace_ep{args.episode:06d}.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] output={output.resolve()}")
    print(f"[write] encoding video at {args.fps} fps...")

    writer = imageio.get_writer(output, fps=float(args.fps))
    try:
        for t in range(t_horizon):
            cam_artist.set_data(frames[t])
            cursor.set_xdata([t, t])

            region = int(labels[t])
            if region == 0:
                region_name = "free"
            elif region == 1:
                region_name = "precontact"
            else:
                region_name = "contact"

            txt.set_text(
                "\n".join(
                    [
                        f"frame: {t}",
                        f"region: {region_name} ({region})",
                        f"signal: {float(y[t]):.5f}",
                    ]
                )
            )

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(frame)
            if (t + 1) % 25 == 0 or (t + 1) == t_horizon:
                print(f"[write] {t + 1}/{t_horizon}")
    finally:
        writer.close()
        plt.close(fig)

    print(f"Saved video: {output}")
    print(f"Parquet: {parquet}")


if __name__ == "__main__":
    main()
