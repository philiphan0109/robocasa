#!/usr/bin/env python3
"""
Jacobian-based contact-to-joint external torque probe for one LeRobot episode.

This script does NOT modify datasets. It replays one episode and writes:
1) MP4 probe dashboard
2) summary JSON
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


def _contact_frame_force_world(c, cf, normal_only: bool):
    # MuJoCo contact frame basis vectors are stored in c.frame as:
    # n = frame[0:3], t1 = frame[3:6], t2 = frame[6:9], all in world coordinates.
    n = np.asarray(c.frame[0:3], dtype=np.float64)
    t1 = np.asarray(c.frame[3:6], dtype=np.float64)
    t2 = np.asarray(c.frame[6:9], dtype=np.float64)

    if normal_only:
        f_world = n * float(cf[0])
        t_world = np.zeros((3,), dtype=np.float64)
    else:
        f_world = n * float(cf[0]) + t1 * float(cf[1]) + t2 * float(cf[2])
        t_world = n * float(cf[3]) + t1 * float(cf[4]) + t2 * float(cf[5])

    return f_world, t_world


def sample_contact_joint_torque(
    sim, robot_geom_ids_set, arm_dof_indices, normal_only=True
):
    d = sim.data
    m = sim.model
    nv = int(m.nv)
    tau_full = np.zeros((nv,), dtype=np.float64)

    ncon_total = int(d.ncon)
    ncon_robot = 0
    normal_sum_robot = 0.0
    normal_max_robot = 0.0

    cf = np.zeros((6,), dtype=np.float64)
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)

    for i in range(ncon_total):
        c = d.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        robot_other = (g1 in robot_geom_ids_set) ^ (g2 in robot_geom_ids_set)
        if not robot_other:
            continue

        ncon_robot += 1
        mujoco.mj_contactForce(m._model, d._data, i, cf)

        normal = abs(float(cf[0]))
        normal_sum_robot += normal
        normal_max_robot = max(normal_max_robot, normal)

        f_world, t_world = _contact_frame_force_world(
            c=c, cf=cf, normal_only=normal_only
        )

        robot_geom = g1 if g1 in robot_geom_ids_set else g2
        body_id = int(m.geom_bodyid[robot_geom])
        point = np.asarray(c.pos, dtype=np.float64)

        jacp.fill(0.0)
        jacr.fill(0.0)
        mujoco.mj_jac(m._model, d._data, jacp, jacr, point, body_id)
        tau_full += jacp.T @ f_world + jacr.T @ t_world

    tau_arm = tau_full[np.asarray(arm_dof_indices, dtype=np.int64)]
    return {
        "tau_contact_arm": tau_arm,
        "ncon_total": ncon_total,
        "ncon_robot": ncon_robot,
        "normal_sum_robot": normal_sum_robot,
        "normal_max_robot": normal_max_robot,
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
    normal_only: bool,
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

    tau_contact_arm = []
    ncon_total = []
    ncon_robot = []
    normal_sum_robot = []
    normal_max_robot = []
    divergence = np.full((t_horizon,), np.nan, dtype=np.float64)
    camera_frames = []

    for t in range(t_horizon):
        reset_to(env, {"states": states[t]})
        env.step(actions[t])

        contact = sample_contact_joint_torque(
            sim=env.sim,
            robot_geom_ids_set=robot_geom_ids_set,
            arm_dof_indices=arm_dof_indices,
            normal_only=normal_only,
        )

        frame = env.sim.render(height=height, width=width, camera_name=camera_name)[
            ::-1
        ]
        tau_contact_arm.append(contact["tau_contact_arm"])
        ncon_total.append(contact["ncon_total"])
        ncon_robot.append(contact["ncon_robot"])
        normal_sum_robot.append(contact["normal_sum_robot"])
        normal_max_robot.append(contact["normal_max_robot"])
        camera_frames.append(frame)

        if t < t_horizon - 1:
            s_next = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            divergence[t] = float(np.linalg.norm(s_next - states[t + 1]))

    tau_contact_arm = np.stack(tau_contact_arm, axis=0)
    ncon_total = np.asarray(ncon_total, dtype=np.int64)
    ncon_robot = np.asarray(ncon_robot, dtype=np.int64)
    normal_sum_robot = np.asarray(normal_sum_robot, dtype=np.float64)
    normal_max_robot = np.asarray(normal_max_robot, dtype=np.float64)

    return {
        "T": int(t_horizon),
        "camera_frames": camera_frames,
        "divergence": divergence,
        "arm_joint_names": arm_joint_names,
        "arm_dof_indices": arm_dof_indices,
        "robot_geom_names": robot_geom_names,
        "tau_contact_arm": tau_contact_arm,
        "ncon_total": ncon_total,
        "ncon_robot": ncon_robot,
        "normal_sum_robot": normal_sum_robot,
        "normal_max_robot": normal_max_robot,
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

    tau = data["tau_contact_arm"]  # [T, ndof]
    tau_normed = normalize_time_series(tau, mode=normalize_mode, eps=norm_eps)
    tau_plot = tau_normed if normalize_mode != "none" else tau
    tau_mag = np.linalg.norm(tau, axis=1)
    tau_maxabs = np.max(np.abs(tau), axis=1)

    divergence = data["divergence"]
    camera_frames = data["camera_frames"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    ax_cam, ax_tau, ax_mag, ax_txt = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    cam_artist = ax_cam.imshow(camera_frames[0])
    ax_cam.set_title(f"Camera: {camera_name}")
    ax_cam.axis("off")

    for j in range(tau_plot.shape[1]):
        ax_tau.plot(
            x,
            tau_plot[:, j],
            linewidth=1.0,
            label=f"dof{data['arm_dof_indices'][j]}",
        )
    tau_cursor = ax_tau.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ttl = "Estimated contact->joint external torque per DoF"
    if normalize_mode != "none":
        ttl += f" [{normalize_mode}]"
    ax_tau.set_title(ttl)
    ax_tau.set_xlabel("Frame")
    ax_tau.set_ylabel("Torque (or normalized)")
    ax_tau.grid(True, alpha=0.25)
    ax_tau.legend(loc="upper right", fontsize=7, ncol=2)

    ax_mag.plot(
        x, tau_mag, linewidth=1.8, color="tab:blue", label="||tau_contact_arm||_2"
    )
    ax_mag.plot(
        x, tau_maxabs, linewidth=1.6, color="tab:orange", label="max|tau_contact_arm|"
    )
    ax_mag.plot(
        x,
        data["normal_sum_robot"],
        linewidth=1.3,
        color="tab:green",
        label="contact_normal_sum(robot-other)",
    )
    mag_cursor = ax_mag.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_mag.set_title("External torque magnitude + contact context")
    ax_mag.set_xlabel("Frame")
    ax_mag.set_ylabel("Magnitude")
    ax_mag.grid(True, alpha=0.25)
    ax_mag.legend(loc="upper right", fontsize=8)

    ax_txt.axis("off")
    txt_artist = ax_txt.text(
        0.01, 0.99, "", va="top", ha="left", fontsize=10, family="monospace"
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_video, fps=fps) as writer:
        for t in range(t_horizon):
            cam_artist.set_data(camera_frames[t])
            tau_cursor.set_xdata([t, t])
            mag_cursor.set_xdata([t, t])

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
                    f"ncon_total: {int(data['ncon_total'][t])}",
                    f"ncon_robot: {int(data['ncon_robot'][t])}",
                    f"contact_normal_sum: {float(data['normal_sum_robot'][t]):.5f}",
                    f"contact_normal_max: {float(data['normal_max_robot'][t]):.5f}",
                    f"||tau_contact_arm||_2: {float(tau_mag[t]):.5f}",
                    f"max|tau_contact_arm|: {float(tau_maxabs[t]):.5f}",
                ]
            )
            txt_artist.set_text(txt)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(frame)

    plt.close(fig)

    return {
        "tau_contact_arm_l2": stats_1d(tau_mag),
        "tau_contact_arm_maxabs": stats_1d(tau_maxabs),
        "ncon_total": stats_1d(data["ncon_total"]),
        "ncon_robot": stats_1d(data["ncon_robot"]),
        "normal_sum_robot": stats_1d(data["normal_sum_robot"]),
        "normal_max_robot": stats_1d(data["normal_max_robot"]),
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
    contact_wrench: str,
):
    div = data["divergence"]
    finite_div = div[np.isfinite(div)]

    valid_mask = np.zeros((data["T"],), dtype=bool)
    valid_mask[np.isfinite(div)] = div[np.isfinite(div)] <= div_thresh

    summary = {
        "episode": int(episode),
        "replay_mode": "action",
        "probe": "contact_jacobian",
        "contact_wrench_mode": contact_wrench,
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
        "metrics": metrics,
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
    base = Path("artifacts") / f"contact_jacobian_probe_ep{episode:06d}"
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
        help="Output MP4 path. Default: artifacts/contact_jacobian_probe_epXXXXXX.mp4",
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
        help="Normalization mode for plotted DoF torques",
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-6,
        help="Small epsilon to avoid divide-by-zero in normalization",
    )
    parser.add_argument(
        "--contact-wrench",
        type=str,
        choices=["normal", "full"],
        default="normal",
        help="normal = normal component only; full = normal + friction + contact torque",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    default_video, default_summary = default_output_paths(args.episode)
    output_video = Path(args.output_video) if args.output_video else default_video
    summary_path = Path(str(output_video.with_suffix("")) + "_summary.json")
    if args.output_video is None:
        summary_path = default_summary

    normal_only = args.contact_wrench == "normal"
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
            normal_only=normal_only,
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
        contact_wrench=args.contact_wrench,
    )

    print(f"Saved video: {output_video}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
