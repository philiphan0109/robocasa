#!/usr/bin/env python3
"""
Visualization-only force dashboard generator for a single LeRobot episode.

This script does NOT edit or copy datasets. It runs action replay and writes:
1) an MP4 dashboard video, and
2) a summary JSON artifact.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
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


def select_robot_bodies(model):
    prefixes = ("robot0_", "gripper0_", "mobilebase0_")
    body_ids = []
    body_names = []
    for i in range(model.nbody):
        name = model.body_id2name(i)
        if name is None:
            continue
        if name.startswith(prefixes):
            body_ids.append(i)
            body_names.append(name)
    return body_ids, body_names


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

    robot_body_ids, robot_body_names = select_robot_bodies(model)
    if len(robot_body_ids) == 0:
        raise RuntimeError("No robot bodies found for cfrc selection.")

    qfrc_selected = []
    cfrc_robot_force_norms = []
    ncon = []
    divergence = np.full((t_horizon,), np.nan, dtype=np.float64)
    camera_frames = []

    for t in range(t_horizon):
        reset_to(env, {"states": states[t]})
        env.step(actions[t])  # restore-state -> play-action -> record-force

        d = env.sim.data
        qfrc = np.asarray(d.qfrc_constraint, dtype=np.float64)
        cfrc_ext = np.asarray(d.cfrc_ext, dtype=np.float64)
        cfrc_trans = cfrc_ext[robot_body_ids, 0:3]
        cfrc_norm = np.linalg.norm(cfrc_trans, axis=1)

        frame = env.sim.render(height=height, width=width, camera_name=camera_name)[
            ::-1
        ]

        qfrc_selected.append(qfrc[arm_dof_indices])
        cfrc_robot_force_norms.append(cfrc_norm)
        ncon.append(int(d.ncon))
        camera_frames.append(frame)

        if t < t_horizon - 1:
            s_next = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            divergence[t] = float(np.linalg.norm(s_next - states[t + 1]))

    qfrc_selected = np.stack(qfrc_selected, axis=0)
    cfrc_robot_force_norms = np.stack(cfrc_robot_force_norms, axis=0)
    ncon = np.asarray(ncon, dtype=np.int64)

    total_body_norm = np.sum(cfrc_robot_force_norms, axis=0)

    return {
        "T": int(t_horizon),
        "qfrc_selected": qfrc_selected,
        "cfrc_robot_force_norms": cfrc_robot_force_norms,
        "ncon": ncon,
        "divergence": divergence,
        "camera_frames": camera_frames,
        "arm_joint_names": arm_joint_names,
        "arm_dof_indices": arm_dof_indices,
        "robot_body_ids": robot_body_ids,
        "robot_body_names": robot_body_names,
        "total_body_norm": total_body_norm,
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "dof_joint_ids": np.asarray(model.dof_jntid, dtype=np.int64).tolist(),
    }


def make_dashboard_video(
    output_video: Path,
    data: dict,
    episode: int,
    fps: int,
    camera_name: str,
    topk_cfrc: int,
    div_thresh: float,
):
    t_horizon = data["T"]
    qfrc = data["qfrc_selected"]
    cfrc_norm = data["cfrc_robot_force_norms"]
    ncon = data["ncon"]
    divergence = data["divergence"]
    camera_frames = data["camera_frames"]

    robot_body_names = data["robot_body_names"]
    robot_body_ids = data["robot_body_ids"]

    rank_idx = np.argsort(-data["total_body_norm"])
    k = max(1, min(int(topk_cfrc), len(rank_idx)))
    topk_local = rank_idx[:k]
    cfrc_topk = cfrc_norm[:, topk_local]
    topk_body_names = [robot_body_names[i] for i in topk_local]
    topk_body_ids = [robot_body_ids[i] for i in topk_local]

    x = np.arange(t_horizon)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    ax_cam, ax_q, ax_c, ax_txt = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    cam_artist = ax_cam.imshow(camera_frames[0])
    ax_cam.set_title(f"Camera: {camera_name}")
    ax_cam.axis("off")

    for j in range(qfrc.shape[1]):
        ax_q.plot(
            x, qfrc[:, j], linewidth=1.0, label=f"dof{data['arm_dof_indices'][j]}"
        )
    q_cursor = ax_q.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_q.set_title("qfrc_constraint (selected arm DoFs)")
    ax_q.set_xlabel("Frame")
    ax_q.set_ylabel("Torque / generalized force")
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="upper right", fontsize=7, ncol=2)

    for j in range(cfrc_topk.shape[1]):
        ax_c.plot(x, cfrc_topk[:, j], linewidth=1.0, label=topk_body_names[j])
    c_cursor = ax_c.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_c.set_title("cfrc_ext translational norm (top-k robot bodies)")
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("||force_xyz||")
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(loc="upper right", fontsize=6)

    ax_txt.axis("off")
    txt_artist = ax_txt.text(
        0.01, 0.99, "", va="top", ha="left", fontsize=10, family="monospace"
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_video, fps=fps) as writer:
        for t in range(t_horizon):
            cam_artist.set_data(camera_frames[t])
            q_cursor.set_xdata([t, t])
            c_cursor.set_xdata([t, t])

            div = divergence[t]
            is_valid = bool(np.isfinite(div) and (div <= div_thresh))
            valid_text = "VALID" if is_valid else "INVALID"
            div_text = f"{div:.6f}" if np.isfinite(div) else "nan"

            txt = "\n".join(
                [
                    f"episode: {episode:06d}",
                    f"replay_mode: action",
                    f"frame: {t}/{t_horizon - 1}",
                    f"time_sec: {t / float(fps):.3f}",
                    f"ncon: {int(ncon[t])}",
                    f"div_l2: {div_text}",
                    f"valid(div<={div_thresh}): {valid_text}",
                    "",
                    f"arm_joints: {len(data['arm_joint_names'])}",
                    f"arm_dofs: {len(data['arm_dof_indices'])}",
                    f"robot_bodies_total: {len(robot_body_ids)}",
                    f"cfrc_topk: {k}",
                ]
            )
            txt_artist.set_text(txt)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(frame)

    plt.close(fig)

    return {
        "topk_body_names": topk_body_names,
        "topk_body_ids": topk_body_ids,
        "topk_local_indices": topk_local.tolist(),
    }


def write_summary(
    summary_path: Path,
    output_video: Path,
    data: dict,
    topk_info: dict,
    episode: int,
    camera_name: str,
    fps: int,
    width: int,
    height: int,
    div_thresh: float,
):
    div = data["divergence"]
    finite_div = div[np.isfinite(div)]
    ncon = data["ncon"]

    valid_mask = np.zeros((data["T"],), dtype=bool)
    valid_mask[np.isfinite(div)] = div[np.isfinite(div)] <= div_thresh

    summary = {
        "episode": int(episode),
        "replay_mode": "action",
        "output_video": str(output_video),
        "camera_name": camera_name,
        "fps": int(fps),
        "camera_width": int(width),
        "camera_height": int(height),
        "T": int(data["T"]),
        "nv": int(data["nv"]),
        "nbody": int(data["nbody"]),
        "selected_arm_joint_names": data["arm_joint_names"],
        "selected_arm_dof_indices": data["arm_dof_indices"],
        "selected_cfrc_topk_body_names": topk_info["topk_body_names"],
        "selected_cfrc_topk_body_ids": topk_info["topk_body_ids"],
        "dof_joint_ids": data["dof_joint_ids"],
        "ncon_stats": {
            "mean": float(np.mean(ncon)),
            "std": float(np.std(ncon)),
            "min": int(np.min(ncon)),
            "max": int(np.max(ncon)),
        },
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
    base = Path("artifacts") / f"force_dashboard_ep{episode:06d}"
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
        help="Output MP4 path. Default: artifacts/force_dashboard_epXXXXXX.mp4",
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
        help="Include gripper joints in qfrc plot",
    )
    parser.add_argument(
        "--topk-cfrc",
        type=int,
        default=8,
        help="Top-k robot bodies by total cfrc norm",
    )
    parser.add_argument(
        "--div-thresh",
        type=float,
        default=0.1,
        help="Divergence threshold for validity display",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional replay truncation length",
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

    topk_info = make_dashboard_video(
        output_video=output_video,
        data=data,
        episode=args.episode,
        fps=args.fps,
        camera_name=args.camera_name,
        topk_cfrc=args.topk_cfrc,
        div_thresh=args.div_thresh,
    )

    write_summary(
        summary_path=summary_path,
        output_video=output_video,
        data=data,
        topk_info=topk_info,
        episode=args.episode,
        camera_name=args.camera_name,
        fps=args.fps,
        width=args.width,
        height=args.height,
        div_thresh=args.div_thresh,
    )

    print(f"Saved video: {output_video}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
