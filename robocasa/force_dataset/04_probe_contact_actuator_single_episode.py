#!/usr/bin/env python3
"""
Contact + actuator probe dashboard for a single LeRobot episode.

This script does NOT modify datasets. It runs action replay and writes:
1) an MP4 dashboard video, and
2) a summary JSON artifact.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("matplotlib is required for dashboard rendering.") from e

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
        "robocasa is not importable. Set PYTHONPATH (e.g. PYTHONPATH=/home/alfredo/robocasa)."
    ) from e


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
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def select_arm_dofs(model, include_gripper=False):
    joint_names = [model.joint_id2name(i) for i in range(model.njnt)]
    dof_jntid = np.asarray(model.dof_jntid, dtype=np.int64)

    target_joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
    if include_gripper:
        for name in joint_names:
            if name.startswith("gripper0_"):
                target_joint_names.append(name)

    selected_joint_names = [n for n in target_joint_names if n in joint_names]
    selected_dof_indices = []
    for jname in selected_joint_names:
        jid = model.joint_name2id(jname)
        dof_idx = np.where(dof_jntid == jid)[0].tolist()
        selected_dof_indices.extend(dof_idx)

    selected_dof_indices = sorted(set(selected_dof_indices))
    return selected_joint_names, selected_dof_indices


def select_robot_geoms(model):
    prefixes = ("robot0_", "gripper0_", "mobilebase0_")
    geom_ids = []
    geom_names = []
    for i in range(model.ngeom):
        name = model.geom_id2name(i)
        if name is None:
            continue
        if name.startswith(prefixes):
            geom_ids.append(i)
            geom_names.append(name)
    return geom_ids, geom_names


def sample_contact_metrics(sim, robot_geom_ids_set):
    d = sim.data
    ncon = int(d.ncon)

    robot_count = 0
    robot_normal_sum = 0.0
    robot_normal_max = 0.0
    robot_force_l2_sum = 0.0

    all_normal_sum = 0.0
    all_normal_max = 0.0

    cf = np.zeros((6,), dtype=np.float64)
    for i in range(ncon):
        mujoco.mj_contactForce(sim.model._model, sim.data._data, i, cf)
        normal = abs(float(cf[0]))
        force_l2 = float(np.linalg.norm(cf[:3]))
        all_normal_sum += normal
        all_normal_max = max(all_normal_max, normal)

        c = d.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        robot_other = (g1 in robot_geom_ids_set) ^ (g2 in robot_geom_ids_set)
        if robot_other:
            robot_count += 1
            robot_normal_sum += normal
            robot_normal_max = max(robot_normal_max, normal)
            robot_force_l2_sum += force_l2

    return {
        "contact_total_count": ncon,
        "contact_robot_count": robot_count,
        "contact_robot_normal_sum": robot_normal_sum,
        "contact_robot_normal_max": robot_normal_max,
        "contact_robot_force_l2_sum": robot_force_l2_sum,
        "contact_all_normal_sum": all_normal_sum,
        "contact_all_normal_max": all_normal_max,
    }


def normalize_time_series(arr, mode: str, eps: float):
    x = np.asarray(arr, dtype=np.float64)
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True

    if mode == "none":
        out = x.copy()
    elif mode == "static_absmax":
        scale = np.max(np.abs(x), axis=0)
        denom = np.maximum(scale, float(eps))
        out = np.clip(x / denom[None, :], -1.0, 1.0)
    elif mode == "running_absmax":
        running = np.maximum.accumulate(np.abs(x), axis=0)
        denom = np.maximum(running, float(eps))
        out = np.clip(x / denom, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    if squeeze:
        out = out[:, 0]
    return out


def stats_1d(arr):
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p95": float(np.quantile(x, 0.95)),
    }


def collect_timeseries(
    env,
    dataset_root: Path,
    episode: int,
    camera_name: str,
    width: int,
    height: int,
    include_gripper: bool,
    max_steps: int | None,
):
    states = LU.get_episode_states(dataset_root, episode)
    actions = LU.get_episode_actions(dataset_root, episode, abs_actions=False)
    if len(states) != len(actions):
        raise RuntimeError(
            f"states/actions mismatch: {len(states)} vs {len(actions)} for episode {episode}"
        )

    t_horizon = len(states)
    if max_steps is not None:
        t_horizon = min(t_horizon, int(max_steps))
    if t_horizon < 2:
        raise RuntimeError("Need at least 2 timesteps for action replay visualization.")

    initial_state = {
        "states": states[0],
        "model": LU.get_episode_model_xml(dataset_root, episode),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset_root, episode)),
    }
    reset_to(env, initial_state)

    model = env.sim.model
    arm_joint_names, arm_dof_indices = select_arm_dofs(model, include_gripper)
    if len(arm_dof_indices) == 0:
        raise RuntimeError("No arm DoF indices found from robot0_joint1..7 selection.")

    robot_geom_ids, robot_geom_names = select_robot_geoms(model)
    robot_geom_ids_set = set(robot_geom_ids)

    qfrc_constraint_selected = []
    qfrc_actuator_selected = []
    contact_total_count = []
    contact_robot_count = []
    contact_robot_normal_sum = []
    contact_robot_normal_max = []
    contact_robot_force_l2_sum = []
    contact_all_normal_sum = []
    contact_all_normal_max = []
    divergence = np.full((t_horizon,), np.nan, dtype=np.float64)
    camera_frames = []

    for t in range(t_horizon):
        reset_to(env, {"states": states[t]})
        env.step(actions[t])  # restore-state -> play-action -> record-force

        d = env.sim.data
        qfrc_constraint = np.asarray(d.qfrc_constraint, dtype=np.float64)
        qfrc_actuator = np.asarray(d.qfrc_actuator, dtype=np.float64)
        contact = sample_contact_metrics(env.sim, robot_geom_ids_set=robot_geom_ids_set)

        frame = env.sim.render(height=height, width=width, camera_name=camera_name)[
            ::-1
        ]

        qfrc_constraint_selected.append(qfrc_constraint[arm_dof_indices])
        qfrc_actuator_selected.append(qfrc_actuator[arm_dof_indices])
        contact_total_count.append(contact["contact_total_count"])
        contact_robot_count.append(contact["contact_robot_count"])
        contact_robot_normal_sum.append(contact["contact_robot_normal_sum"])
        contact_robot_normal_max.append(contact["contact_robot_normal_max"])
        contact_robot_force_l2_sum.append(contact["contact_robot_force_l2_sum"])
        contact_all_normal_sum.append(contact["contact_all_normal_sum"])
        contact_all_normal_max.append(contact["contact_all_normal_max"])
        camera_frames.append(frame)

        if t < t_horizon - 1:
            s_next = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            divergence[t] = float(np.linalg.norm(s_next - states[t + 1]))

    return {
        "T": int(t_horizon),
        "camera_frames": camera_frames,
        "divergence": divergence,
        "arm_joint_names": arm_joint_names,
        "arm_dof_indices": arm_dof_indices,
        "robot_geom_names": robot_geom_names,
        "qfrc_constraint_selected": np.stack(qfrc_constraint_selected, axis=0),
        "qfrc_actuator_selected": np.stack(qfrc_actuator_selected, axis=0),
        "contact_total_count": np.asarray(contact_total_count, dtype=np.int64),
        "contact_robot_count": np.asarray(contact_robot_count, dtype=np.int64),
        "contact_robot_normal_sum": np.asarray(
            contact_robot_normal_sum, dtype=np.float64
        ),
        "contact_robot_normal_max": np.asarray(
            contact_robot_normal_max, dtype=np.float64
        ),
        "contact_robot_force_l2_sum": np.asarray(
            contact_robot_force_l2_sum, dtype=np.float64
        ),
        "contact_all_normal_sum": np.asarray(contact_all_normal_sum, dtype=np.float64),
        "contact_all_normal_max": np.asarray(contact_all_normal_max, dtype=np.float64),
    }


def make_dashboard_video(
    output_video: Path,
    data: dict,
    episode: int,
    fps: int,
    camera_name: str,
    div_thresh: float,
    normalize_mode: str,
    norm_eps: float,
):
    t_horizon = data["T"]
    x = np.arange(t_horizon)
    divergence = data["divergence"]
    camera_frames = data["camera_frames"]

    qfrc_constraint_mag = np.linalg.norm(data["qfrc_constraint_selected"], axis=1)
    qfrc_actuator_mag = np.linalg.norm(data["qfrc_actuator_selected"], axis=1)
    qfrc_constraint_mag_n = normalize_time_series(
        qfrc_constraint_mag, mode=normalize_mode, eps=norm_eps
    )
    qfrc_actuator_mag_n = normalize_time_series(
        qfrc_actuator_mag, mode=normalize_mode, eps=norm_eps
    )

    robot_contact_normal_sum = data["contact_robot_normal_sum"]
    robot_contact_normal_max = data["contact_robot_normal_max"]
    robot_contact_count = data["contact_robot_count"].astype(np.float64)
    robot_contact_normal_sum_n = normalize_time_series(
        robot_contact_normal_sum, mode=normalize_mode, eps=norm_eps
    )
    robot_contact_normal_max_n = normalize_time_series(
        robot_contact_normal_max, mode=normalize_mode, eps=norm_eps
    )
    robot_contact_count_n = normalize_time_series(
        robot_contact_count, mode=normalize_mode, eps=norm_eps
    )

    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.7])
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_cmag = fig.add_subplot(gs[1, 0])
    ax_ccount = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[2, :])

    cam_artist = ax_cam.imshow(camera_frames[0])
    ax_cam.set_title(f"Camera: {camera_name}")
    ax_cam.axis("off")

    ax_q.plot(
        x,
        qfrc_constraint_mag,
        linewidth=1.8,
        color="tab:blue",
        label="|qfrc_constraint|_arm",
    )
    ax_q.plot(
        x,
        qfrc_actuator_mag,
        linewidth=1.8,
        color="tab:orange",
        label="|qfrc_actuator|_arm",
    )
    if normalize_mode != "none":
        ax_q.plot(
            x,
            qfrc_constraint_mag_n,
            linewidth=1.2,
            linestyle="--",
            color="tab:cyan",
            label=f"constraint_norm({normalize_mode})",
        )
        ax_q.plot(
            x,
            qfrc_actuator_mag_n,
            linewidth=1.2,
            linestyle="--",
            color="tab:red",
            label=f"actuator_norm({normalize_mode})",
        )
    q_cursor = ax_q.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_q.set_title("Arm force magnitudes")
    ax_q.set_xlabel("Frame")
    ax_q.set_ylabel("Magnitude")
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="upper right", fontsize=8)

    ax_cmag.plot(
        x,
        robot_contact_normal_sum,
        linewidth=1.8,
        color="tab:green",
        label="contact_normal_sum(robot-other)",
    )
    ax_cmag.plot(
        x,
        robot_contact_normal_max,
        linewidth=1.8,
        color="tab:purple",
        label="contact_normal_max(robot-other)",
    )
    ax_cmag.plot(
        x,
        data["contact_robot_force_l2_sum"],
        linewidth=1.6,
        color="tab:red",
        label="contact_force_l2_sum(robot-other)",
    )
    if normalize_mode != "none":
        ax_cmag.plot(
            x,
            robot_contact_normal_sum_n,
            linewidth=1.1,
            linestyle="--",
            color="tab:olive",
            label=f"normal_sum_norm({normalize_mode})",
        )
        ax_cmag.plot(
            x,
            robot_contact_normal_max_n,
            linewidth=1.1,
            linestyle="--",
            color="tab:pink",
            label=f"normal_max_norm({normalize_mode})",
        )
    cmag_cursor = ax_cmag.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_cmag.set_title("Contact force magnitudes")
    ax_cmag.set_xlabel("Frame")
    ax_cmag.set_ylabel("Magnitude")
    ax_cmag.grid(True, alpha=0.25)
    ax_cmag.legend(loc="upper right", fontsize=7)

    ax_ccount.plot(
        x,
        robot_contact_count,
        linewidth=1.6,
        color="tab:brown",
        label="contact_count(robot-other)",
    )
    ax_ccount.plot(
        x,
        data["contact_total_count"].astype(np.float64),
        linewidth=1.4,
        color="tab:gray",
        label="contact_count(total)",
    )
    if normalize_mode != "none":
        ax_ccount.plot(
            x,
            robot_contact_count_n,
            linewidth=1.1,
            linestyle="--",
            color="tab:orange",
            label=f"count_norm({normalize_mode})",
        )
    ccount_cursor = ax_ccount.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_ccount.set_title("Contact counts")
    ax_ccount.set_xlabel("Frame")
    ax_ccount.set_ylabel("Count")
    ax_ccount.grid(True, alpha=0.25)
    ax_ccount.legend(loc="upper right", fontsize=7)

    ax_txt.axis("off")
    txt_artist = ax_txt.text(
        0.01, 0.99, "", va="top", ha="left", fontsize=10, family="monospace"
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_video, fps=fps) as writer:
        for t in range(t_horizon):
            cam_artist.set_data(camera_frames[t])
            q_cursor.set_xdata([t, t])
            cmag_cursor.set_xdata([t, t])
            ccount_cursor.set_xdata([t, t])

            div = divergence[t]
            is_valid = bool(np.isfinite(div) and (div <= div_thresh))
            valid_text = "VALID" if is_valid else "INVALID"
            div_text = f"{div:.6f}" if np.isfinite(div) else "nan"

            txt = "\n".join(
                [
                    f"episode: {episode:06d}",
                    "replay_mode: action",
                    f"frame: {t}/{t_horizon - 1}",
                    f"time_sec: {t / float(fps):.3f}",
                    f"div_l2: {div_text}",
                    f"valid(div<={div_thresh}): {valid_text}",
                    f"norm_mode: {normalize_mode}",
                    "",
                    f"arm_dofs: {len(data['arm_dof_indices'])}",
                    f"robot_geoms: {len(data['robot_geom_names'])}",
                    f"contact_total: {int(data['contact_total_count'][t])}",
                    f"contact_robot: {int(data['contact_robot_count'][t])}",
                    f"contact_normal_sum: {float(data['contact_robot_normal_sum'][t]):.5f}",
                    f"contact_normal_max: {float(data['contact_robot_normal_max'][t]):.5f}",
                    f"|qfrc_constraint|_arm: {float(qfrc_constraint_mag[t]):.5f}",
                    f"|qfrc_actuator|_arm: {float(qfrc_actuator_mag[t]):.5f}",
                ]
            )
            txt_artist.set_text(txt)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(frame)

    plt.close(fig)

    return {
        "force_metrics": {
            "qfrc_constraint_arm_l2": stats_1d(qfrc_constraint_mag),
            "qfrc_actuator_arm_l2": stats_1d(qfrc_actuator_mag),
        },
        "contact_metrics": {
            "contact_total_count": stats_1d(data["contact_total_count"]),
            "contact_robot_count": stats_1d(data["contact_robot_count"]),
            "contact_robot_normal_sum": stats_1d(data["contact_robot_normal_sum"]),
            "contact_robot_normal_max": stats_1d(data["contact_robot_normal_max"]),
            "contact_robot_force_l2_sum": stats_1d(data["contact_robot_force_l2_sum"]),
        },
    }


def write_summary(
    summary_path: Path,
    output_video: Path,
    data: dict,
    metrics: dict,
    episode: int,
    camera_name: str,
    fps: int,
    width: int,
    height: int,
    div_thresh: float,
    normalize_mode: str,
    norm_eps: float,
):
    div = data["divergence"]
    finite_div = div[np.isfinite(div)]

    valid_mask = np.zeros((data["T"],), dtype=bool)
    valid_mask[np.isfinite(div)] = div[np.isfinite(div)] <= div_thresh

    summary = {
        "episode": int(episode),
        "replay_mode": "action",
        "probe": "contact_actuator",
        "output_video": str(output_video),
        "camera_name": camera_name,
        "fps": int(fps),
        "camera_width": int(width),
        "camera_height": int(height),
        "normalization": {
            "mode": normalize_mode,
            "eps": float(norm_eps),
            "causal": normalize_mode in ("none", "running_absmax"),
        },
        "T": int(data["T"]),
        "selected_arm_joint_names": data["arm_joint_names"],
        "selected_arm_dof_indices": data["arm_dof_indices"],
        "robot_geom_count": len(data["robot_geom_names"]),
        "force_metrics": metrics["force_metrics"],
        "contact_metrics": metrics["contact_metrics"],
        "divergence_l2_stats": {
            "count": int(finite_div.size),
            "mean": float(np.mean(finite_div)) if finite_div.size else None,
            "max": float(np.max(finite_div)) if finite_div.size else None,
            "p95": float(np.quantile(finite_div, 0.95)) if finite_div.size else None,
            "valid_ratio": float(np.mean(valid_mask)) if valid_mask.size else None,
            "threshold": float(div_thresh),
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


def default_output_paths(episode: int):
    base = Path("artifacts") / f"contact_actuator_probe_ep{episode:06d}"
    return base.with_suffix(".mp4"), Path(str(base) + "_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to lerobot dataset root"
    )
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output MP4 path. Default: artifacts/contact_actuator_probe_epXXXXXX.mp4",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="robot0_agentview_right",
        help="Camera to render",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video fps")
    parser.add_argument("--width", type=int, default=256, help="Camera render width")
    parser.add_argument("--height", type=int, default=256, help="Camera render height")
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Include gripper joints in arm signal",
    )
    parser.add_argument(
        "--div-thresh",
        type=float,
        default=0.1,
        help="Divergence threshold for validity display",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Optional replay truncation length"
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=["none", "running_absmax", "static_absmax"],
        default="running_absmax",
        help="Force normalization mode for plotted signals",
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-6,
        help="Small epsilon to avoid divide-by-zero in normalization",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    default_video, default_summary = default_output_paths(args.episode)
    output_video = Path(args.output_video) if args.output_video else default_video
    summary_path = Path(str(output_video.with_suffix("")) + "_summary.json")
    if args.output_video is None:
        summary_path = default_summary

    env = make_env(dataset_root)
    try:
        data = collect_timeseries(
            env=env,
            dataset_root=dataset_root,
            episode=args.episode,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            include_gripper=args.include_gripper,
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    metrics = make_dashboard_video(
        output_video=output_video,
        data=data,
        episode=args.episode,
        fps=args.fps,
        camera_name=args.camera_name,
        div_thresh=args.div_thresh,
        normalize_mode=args.normalize_mode,
        norm_eps=args.norm_eps,
    )

    write_summary(
        summary_path=summary_path,
        output_video=output_video,
        data=data,
        metrics=metrics,
        episode=args.episode,
        camera_name=args.camera_name,
        fps=args.fps,
        width=args.width,
        height=args.height,
        div_thresh=args.div_thresh,
        normalize_mode=args.normalize_mode,
        norm_eps=args.norm_eps,
    )

    print(f"Saved video: {output_video}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
