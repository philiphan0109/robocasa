import argparse
import json
import re
from pathlib import Path

import imageio.v3 as iio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from matplotlib.animation import FuncAnimation

import robocasa.utils.lerobot_utils as LU
from robocasa.utils.env_utils import create_env


# ---------------------------
# Utils
# ---------------------------
def normalize_to_minus1_1(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    xrng = xmax - xmin
    if xrng <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - xmin) / xrng - 1.0).astype(np.float32)


def signal_variation_ratio(x: np.ndarray, threshold: float = 1e-6) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    d = np.abs(np.diff(x))
    return float(np.mean(d > threshold))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def print_signal_stats(name: str, x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    print(
        f"{name}: min={x.min():.6g}, max={x.max():.6g}, "
        f"range={(x.max() - x.min()):.6g}, mean={x.mean():.6g}, std={x.std():.6g}, "
        f"var_ratio(>|dx|>1e-6)={signal_variation_ratio(x):.3f}"
    )


def low_dynamic_range_warning(name: str, x: np.ndarray, eps: float = 1e-12):
    x = np.asarray(x, dtype=np.float64)
    xrng = float(np.max(x) - np.min(x))
    if xrng <= eps:
        print(
            f"[WARN] {name} has low dynamic range (range={xrng:.6g}); normalized curve will be flat."
        )


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def find_episode_dir(dataset_root: Path, episode_idx: int) -> Path:
    ep_dir = dataset_root / "extras" / f"episode_{episode_idx:06d}"
    if ep_dir.exists():
        return ep_dir
    ep_dir2 = dataset_root / "extras" / f"episode_{episode_idx}"
    if ep_dir2.exists():
        return ep_dir2
    matches = sorted((dataset_root / "extras").glob(f"episode_*{episode_idx}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Episode extras dir not found for episode {episode_idx}: tried {ep_dir} and {ep_dir2}"
    )


def load_states_npz(ep_dir: Path) -> np.ndarray:
    states_path = ep_dir / "states.npz"
    if not states_path.exists():
        raise FileNotFoundError(f"states.npz not found: {states_path}")
    data = np.load(states_path, allow_pickle=False)
    if "states" not in data:
        raise KeyError(
            f"'states' key not found in {states_path}. Keys: {list(data.keys())}"
        )
    return data["states"]


def pick_video_path(dataset_root: Path, episode_idx: int) -> Path:
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        raise FileNotFoundError(f"videos/ not found under dataset: {videos_root}")

    candidates = list(videos_root.rglob(f"episode_{episode_idx:06d}.mp4"))
    if not candidates:
        candidates = list(videos_root.rglob(f"episode_{episode_idx}.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"No episode video found for ep={episode_idx} under {videos_root}"
        )

    agent = [p for p in candidates if "agentview" in p.as_posix().lower()]
    if agent:
        left = [p for p in agent if "left" in p.as_posix().lower()]
        if left:
            return sorted(left)[0]
        right = [p for p in agent if "right" in p.as_posix().lower()]
        if right:
            return sorted(right)[0]
        return sorted(agent)[0]

    robotview = [p for p in candidates if "robotview" in p.as_posix().lower()]
    if robotview:
        return sorted(robotview)[0]

    eye = [p for p in candidates if "eye_in_hand" in p.as_posix().lower()]
    if eye:
        return sorted(eye)[0]

    return sorted(candidates)[0]


def read_video_frames(video_path: Path, max_frames: int | None = None) -> np.ndarray:
    frames = []
    for i, frame in enumerate(iio.imiter(video_path.as_posix())):
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame)
        if max_frames is not None and (i + 1) >= max_frames:
            break
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")
    return np.stack(frames, axis=0)


def reset_to_robosuite(env, flat_state: np.ndarray, do_reset: bool = False):
    if do_reset:
        env.reset()
    env.sim.set_state_from_flattened(flat_state)
    env.sim.forward()


def _joint_names(sim) -> list[str]:
    names = []
    try:
        names = list(sim.model.joint_names)
    except Exception:
        pass
    out = []
    for n in names:
        out.append(n.decode("utf-8") if isinstance(n, bytes) else str(n))
    return out


def _joint_dof_indices(sim, jid: int) -> list[int]:
    adr = int(sim.model.jnt_dofadr[jid])
    njnt = int(sim.model.njnt)
    nv = int(sim.model.nv)
    if jid < njnt - 1:
        next_adr = int(sim.model.jnt_dofadr[jid + 1])
        num = max(0, next_adr - adr)
    else:
        num = max(0, nv - adr)
    return list(range(adr, adr + num))


def infer_arm_dof_indices_from_model(sim) -> np.ndarray:
    joint_names = _joint_names(sim)

    panda = []
    for jn in joint_names:
        m = re.fullmatch(r"robot0_joint([1-7])", jn)
        if m:
            panda.append((int(m.group(1)), jn))
    if panda:
        panda = [jn for _, jn in sorted(panda, key=lambda x: x[0])]
        dofs = []
        for jn in panda:
            jid = sim.model.joint_name2id(jn)
            dofs.extend(_joint_dof_indices(sim, jid))
        return np.array(dofs, dtype=int)

    bad_substrings = ["finger", "gripper", "right_finger", "left_finger", "hand"]
    candidates = []
    for jn in joint_names:
        if not jn.startswith("robot0_"):
            continue
        if any(b in jn.lower() for b in bad_substrings):
            continue
        candidates.append(jn)

    def sort_key(name: str):
        m = re.search(r"(\d+)$", name)
        return (0, int(m.group(1))) if m else (1, name)

    candidates = sorted(set(candidates), key=sort_key)
    if not candidates:
        raise RuntimeError(
            "Could not infer arm joints from model joint names. "
            "Print sim.model.joint_names and adjust the filter."
        )

    candidates = candidates[:7]
    dofs = []
    for jn in candidates:
        jid = sim.model.joint_name2id(jn)
        dofs.extend(_joint_dof_indices(sim, jid))
    return np.array(dofs, dtype=int)


def get_selected_body_ids(sim) -> tuple[np.ndarray, list[str], int | None]:
    body_ids = []
    body_names = []
    eef_body_id = None

    # Prefer body linked to robot0 eef site.
    eef_site_id = None
    try:
        eef_site_id = int(sim.model.site_name2id("robot0_right_ft_frame"))
    except Exception:
        try:
            eef_site_id = int(sim.model.site_name2id("robot0_right_hand"))
        except Exception:
            eef_site_id = None

    if eef_site_id is not None:
        eef_body_id = int(sim.model.site_bodyid[eef_site_id])

    try:
        all_body_names = list(sim.model.body_names)
    except Exception:
        all_body_names = []

    for i, n in enumerate(all_body_names):
        name = n.decode("utf-8") if isinstance(n, bytes) else str(n)
        if name.startswith("robot0_") and any(
            k in name.lower() for k in ["gripper", "hand", "finger", "right_link"]
        ):
            body_ids.append(i)
            body_names.append(name)

    if eef_body_id is not None and eef_body_id not in body_ids:
        body_ids.insert(0, eef_body_id)
        bname = all_body_names[eef_body_id]
        body_names.insert(
            0, bname.decode("utf-8") if isinstance(bname, bytes) else str(bname)
        )

    if not body_ids:
        # Last fallback: include all robot0 bodies.
        for i, n in enumerate(all_body_names):
            name = n.decode("utf-8") if isinstance(n, bytes) else str(n)
            if name.startswith("robot0_"):
                body_ids.append(i)
                body_names.append(name)

    return np.array(body_ids, dtype=int), body_names, eef_body_id


def parse_saved_state_to_qpos_qvel(saved_row: np.ndarray, sim):
    saved = np.asarray(saved_row, dtype=np.float64).reshape(-1)
    nq = int(sim.model.nq)
    nv = int(sim.model.nv)
    min_len = 1 + nq + nv
    if saved.shape[0] < min_len:
        raise ValueError(
            f"Saved state length {saved.shape[0]} is shorter than required {min_len} (1+nq+nv)."
        )

    t = saved[0:1]
    qpos = saved[1 : 1 + nq]
    qvel = saved[1 + nq : 1 + nq + nv]
    comparable = np.concatenate([t, qpos, qvel], axis=0)
    return comparable


def print_state_shape_report(saved_states: np.ndarray, sim):
    saved_state_len = int(saved_states.shape[1])
    sim_state_len = int(sim.get_state().flatten().shape[0])
    comparable_len = int(1 + sim.model.nq + sim.model.nv)

    print("\nState format diagnostics:")
    print(f"  saved_state_len:      {saved_state_len}")
    print(f"  sim_state_len:        {sim_state_len}")
    print(f"  comparable_state_len: {comparable_len}")

    if saved_state_len != comparable_len:
        print(
            "[WARN] saved state rows include extra fields or model mismatch. "
            "Only [time, qpos, qvel] comparable slice will be used for drift."
        )
    if sim_state_len != comparable_len:
        print(
            "[WARN] sim.get_state().flatten() length differs from (1+nq+nv). "
            "Drift comparison may be invalid."
        )


def apply_episode_xml_to_env(env, model_xml: str, ep_meta: dict):
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)

    env.reset()

    robosuite_version_minor = int(robosuite.__version__.split(".")[1])
    if robosuite_version_minor <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        xml = postprocess_model_xml(model_xml)
    else:
        xml = env.edit_model_xml(model_xml)

    env.reset_from_xml_string(xml)
    env.sim.reset()
    if hasattr(env, "update_state"):
        env.update_state()


def build_env_from_extras(
    dataset_root: Path, episode_idx: int, use_episode_xml: bool = True
):
    extras_root = dataset_root / "extras"
    dataset_meta_path = extras_root / "dataset_meta.json"
    if not dataset_meta_path.exists():
        raise FileNotFoundError(f"dataset_meta.json not found: {dataset_meta_path}")
    dataset_meta = load_json(dataset_meta_path)

    ep_dir = find_episode_dir(dataset_root, episode_idx)
    ep_meta_path = ep_dir / "ep_meta.json"
    if not ep_meta_path.exists():
        raise FileNotFoundError(f"ep_meta.json not found: {ep_meta_path}")
    ep_meta = load_json(ep_meta_path)

    env_args = dataset_meta.get("env_args", {})
    if isinstance(env_args, str):
        env_args = json.loads(env_args)
    if not isinstance(env_args, dict):
        env_args = {}

    env_kwargs = env_args.get("env_kwargs", {})
    if not isinstance(env_kwargs, dict):
        env_kwargs = {}

    env_name = (
        env_args.get("env_name")
        or dataset_meta.get("env_name")
        or dataset_meta.get("task")
        or dataset_meta.get("task_name")
    )
    env_field = dataset_meta.get("env", None)
    if env_name is None:
        if isinstance(env_field, dict):
            env_name = env_field.get("env_name") or env_field.get("name")
        elif isinstance(env_field, str):
            env_name = env_field
    if env_name is None:
        env_name = dataset_root.parent.parent.name

    env_name = env_name.replace("robocasa/", "")

    robots = env_kwargs.get("robots") or dataset_meta.get("robots") or "PandaOmron"

    split = env_kwargs.get("obj_instance_split", None)
    if split not in ["pretrain", "target", "all", None]:
        split = "pretrain"

    layout_id = ep_meta.get("layout_id", None)
    style_id = ep_meta.get("style_id", None)

    print(
        f"Creating {env_name} with split={split} robots={robots} via create_env (render_onscreen=False)"
    )
    env = create_env(
        env_name=env_name,
        robots=robots,
        split=split,
        layout_ids=layout_id,
        style_ids=style_id,
        seed=0,
        render_onscreen=False,
        render_camera="robot0_robotview",
        camera_names=["robot0_robotview", "robot0_eye_in_hand"],
        camera_widths=128,
        camera_heights=128,
    )

    if hasattr(env, "render_camera"):
        env.render_camera = ["robot0_robotview"]

    xml_loaded = False
    if use_episode_xml:
        try:
            model_xml = LU.get_episode_model_xml(dataset_root, episode_idx)
            apply_episode_xml_to_env(env, model_xml=model_xml, ep_meta=ep_meta)
            xml_loaded = True
            print("Using exact episode model XML from extras/model.xml.gz")
        except Exception as e:
            print(
                f"[WARN] Failed to load/apply episode XML. Falling back to metadata env reset. Error: {e}"
            )

    if not xml_loaded:
        env.reset()

    gym_id = f"robocasa/{env_name}"
    return env, gym_id, robots, split, layout_id, style_id, xml_loaded


def collect_force_from_state_replay(env, states, arm_dof_indices, selected_body_ids):
    sim = env.sim
    t_len = len(states)

    qfrc_arm_norm = np.zeros((t_len,), dtype=np.float32)
    qfrc_arm_per_joint = np.zeros((t_len, len(arm_dof_indices)), dtype=np.float32)

    cfrc_ext_norm_global = np.zeros((t_len,), dtype=np.float32)
    cfrc_ext_selected = np.zeros((t_len, len(selected_body_ids), 6), dtype=np.float32)

    reset_to_robosuite(env, states[0], do_reset=True)
    for t in range(t_len):
        reset_to_robosuite(env, states[t], do_reset=False)
        qfrc = sim.data.qfrc_constraint
        cfrc = sim.data.cfrc_ext

        q_arm = qfrc[arm_dof_indices]
        qfrc_arm_per_joint[t] = q_arm.astype(np.float32)
        qfrc_arm_norm[t] = float(np.linalg.norm(q_arm))

        cfrc_ext_norm_global[t] = float(np.linalg.norm(cfrc))
        cfrc_ext_selected[t] = cfrc[selected_body_ids].astype(np.float32)

    return {
        "qfrc_arm_norm": qfrc_arm_norm,
        "qfrc_arm_per_joint": qfrc_arm_per_joint,
        "cfrc_ext_norm_global": cfrc_ext_norm_global,
        "cfrc_ext_selected": cfrc_ext_selected,
    }


def collect_force_from_action_rollout(
    env,
    states,
    actions,
    arm_dof_indices,
    selected_body_ids,
):
    sim = env.sim
    t_len = min(len(actions), len(states) - 1)

    qfrc_arm_norm = np.zeros((t_len,), dtype=np.float32)
    qfrc_arm_per_joint = np.zeros((t_len, len(arm_dof_indices)), dtype=np.float32)

    cfrc_ext_norm_global = np.zeros((t_len,), dtype=np.float32)
    cfrc_ext_selected = np.zeros((t_len, len(selected_body_ids), 6), dtype=np.float32)

    state_drift = np.zeros((t_len,), dtype=np.float32)

    reset_to_robosuite(env, states[0], do_reset=True)
    for t in range(t_len):
        env.step(actions[t])

        qfrc = sim.data.qfrc_constraint
        cfrc = sim.data.cfrc_ext

        q_arm = qfrc[arm_dof_indices]
        qfrc_arm_per_joint[t] = q_arm.astype(np.float32)
        qfrc_arm_norm[t] = float(np.linalg.norm(q_arm))

        cfrc_ext_norm_global[t] = float(np.linalg.norm(cfrc))
        cfrc_ext_selected[t] = cfrc[selected_body_ids].astype(np.float32)

        sim_state = np.array(sim.get_state().flatten(), dtype=np.float64)
        saved_cmp = parse_saved_state_to_qpos_qvel(states[t + 1], sim)

        n_cmp = min(len(sim_state), len(saved_cmp))
        state_drift[t] = float(np.linalg.norm(sim_state[:n_cmp] - saved_cmp[:n_cmp]))

    return {
        "qfrc_arm_norm": qfrc_arm_norm,
        "qfrc_arm_per_joint": qfrc_arm_per_joint,
        "cfrc_ext_norm_global": cfrc_ext_norm_global,
        "cfrc_ext_selected": cfrc_ext_selected,
        "state_drift_comparable": state_drift,
    }


def validate_mp4_writer(out_path: Path):
    if out_path.suffix.lower() != ".mp4":
        return
    writers = animation.writers.list()
    if "ffmpeg" not in writers:
        raise RuntimeError(
            "MP4 output requested but matplotlib ffmpeg writer is unavailable. "
            "Install ffmpeg in the active env and verify `matplotlib.animation.writers.list()` includes 'ffmpeg'."
        )


def render_sync_video(
    out_path: Path,
    frames: np.ndarray,
    x: np.ndarray,
    q_line: np.ndarray,
    c_line: np.ndarray,
    fps: int,
    title: str,
    q_label: str,
    c_label: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validate_mp4_writer(out_path)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], wspace=0.25, hspace=0.35)

    ax_vid = fig.add_subplot(gs[:, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    im = ax_vid.imshow(frames[0])
    ax_vid.set_title("Rollout")
    ax_vid.axis("off")

    ax_q.plot(x, q_line, linewidth=1.5)
    ax_q.set_title(q_label)
    ax_q.set_xlim(0, len(x) - 1)
    ax_q.set_ylim(-1.05, 1.05)
    vq = ax_q.axvline(0, linewidth=1.5)
    dq = ax_q.plot([0], [q_line[0]], marker="o")[0]

    ax_c.plot(x, c_line, linewidth=1.5)
    ax_c.set_title(c_label)
    ax_c.set_xlim(0, len(x) - 1)
    ax_c.set_ylim(-1.05, 1.05)
    vc = ax_c.axvline(0, linewidth=1.5)
    dc = ax_c.plot([0], [c_line[0]], marker="o")[0]

    fig.suptitle(title, y=0.98, fontsize=10)

    def update(t):
        im.set_data(frames[t])
        vq.set_xdata([t, t])
        vc.set_xdata([t, t])
        dq.set_data([t], [q_line[t]])
        dc.set_data([t], [c_line[t]])
        return (im, vq, vc, dq, dc)

    anim = FuncAnimation(fig, update, frames=len(x), interval=1000 / fps, blit=True)
    print(f"Saving synced visualization -> {out_path}")
    if out_path.suffix.lower() == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(out_path.as_posix(), writer=writer, dpi=150)
    else:
        anim.save(out_path.as_posix(), fps=fps, dpi=150)
    plt.close(fig)


def plot_debug_signals(mode, state_data, action_data=None, plot_raw=False):
    q_state_n = normalize_to_minus1_1(state_data["qfrc_arm_norm"])
    c_state_n = normalize_to_minus1_1(state_data["cfrc_ext_norm_global"])

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    x_state = np.arange(len(q_state_n))
    ax[0].plot(x_state, q_state_n, linewidth=1.5, label="state replay")
    ax[1].plot(x_state, c_state_n, linewidth=1.5, label="state replay")

    if mode in ["actions", "both"] and action_data is not None:
        q_action_n = normalize_to_minus1_1(action_data["qfrc_arm_norm"])
        c_action_n = normalize_to_minus1_1(action_data["cfrc_ext_norm_global"])
        x_action = np.arange(len(q_action_n))
        ax[0].plot(
            x_action, q_action_n, linewidth=1.2, alpha=0.8, label="action rollout"
        )
        ax[1].plot(
            x_action, c_action_n, linewidth=1.2, alpha=0.8, label="action rollout"
        )

    ax[0].set_title("qfrc_constraint_arm norm (normalized)")
    ax[1].set_title("cfrc_ext norm global (normalized)")
    ax[0].set_ylim(-1.05, 1.05)
    ax[1].set_ylim(-1.05, 1.05)
    ax[1].set_xlabel("timestep")
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    plt.tight_layout()
    plt.show()

    if plot_raw:
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        x_state = np.arange(len(state_data["qfrc_arm_norm"]))
        ax[0].plot(
            x_state, state_data["qfrc_arm_norm"], linewidth=1.5, label="state replay"
        )
        ax[1].plot(
            x_state,
            state_data["cfrc_ext_norm_global"],
            linewidth=1.5,
            label="state replay",
        )

        if mode in ["actions", "both"] and action_data is not None:
            x_action = np.arange(len(action_data["qfrc_arm_norm"]))
            ax[0].plot(
                x_action,
                action_data["qfrc_arm_norm"],
                linewidth=1.2,
                alpha=0.8,
                label="action rollout",
            )
            ax[1].plot(
                x_action,
                action_data["cfrc_ext_norm_global"],
                linewidth=1.2,
                alpha=0.8,
                label="action rollout",
            )

        ax[0].set_title("qfrc_constraint_arm norm (raw)")
        ax[1].set_title("cfrc_ext norm global (raw)")
        ax[1].set_xlabel("timestep")
        ax[0].legend(loc="best")
        ax[1].legend(loc="best")
        plt.tight_layout()
        plt.show()


def summarize_comparison(state_data, action_data):
    q_corr = safe_corr(state_data["qfrc_arm_norm"], action_data["qfrc_arm_norm"])
    c_corr = safe_corr(
        state_data["cfrc_ext_norm_global"], action_data["cfrc_ext_norm_global"]
    )

    drift = action_data["state_drift_comparable"]
    print("\nMode comparison summary:")
    print(f"  state_drift mean: {float(np.mean(drift)):.6g}")
    print(f"  state_drift max:  {float(np.max(drift)):.6g}")
    print(f"  corr(qfrc_arm_norm state vs action): {q_corr:.6g}")
    print(f"  corr(cfrc_ext_norm_global state vs action): {c_corr:.6g}")
    print(
        f"  nontrivial variation qfrc(state): {signal_variation_ratio(state_data['qfrc_arm_norm']):.3f}"
    )
    print(
        f"  nontrivial variation cfrc(state): {signal_variation_ratio(state_data['cfrc_ext_norm_global']):.3f}"
    )
    print(
        f"  nontrivial variation qfrc(action): {signal_variation_ratio(action_data['qfrc_arm_norm']):.3f}"
    )
    print(
        f"  nontrivial variation cfrc(action): {signal_variation_ratio(action_data['cfrc_ext_norm_global']):.3f}"
    )


def parse_bool_flag(s: str) -> bool:
    s = str(s).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {s}")


def main(args):
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset path does not exist: {dataset_root}")
    if not (dataset_root / "extras").exists():
        raise FileNotFoundError(
            f"--dataset must point to lerobot dataset root containing extras/: {dataset_root}"
        )

    ep_idx = int(args.episode)
    ep_dir = find_episode_dir(dataset_root, ep_idx)

    states = load_states_npz(ep_dir)
    t_states = len(states)
    t = min(t_states, args.max_steps) if args.max_steps is not None else t_states
    states = states[:t]

    video_path = pick_video_path(dataset_root, ep_idx)
    frames = read_video_frames(video_path, max_frames=t)
    t_vid = frames.shape[0]

    t = min(t, t_vid)
    states = states[:t]
    frames = frames[:t]

    print(f"Dataset: {dataset_root}")
    print(f"Episode: {ep_idx}")
    print(f"states: {t_states} -> using {t}")
    print(f"video:  {video_path} (frames read: {t_vid} -> using {t})")

    env, gym_id, robots, split, layout_id, style_id, xml_loaded = build_env_from_extras(
        dataset_root, ep_idx, use_episode_xml=parse_bool_flag(args.use_episode_xml)
    )
    sim = env.sim

    print_state_shape_report(states, sim)

    arm_dof_indices = infer_arm_dof_indices_from_model(sim)
    print(f"Arm DOF indices (len={len(arm_dof_indices)}): {arm_dof_indices}")

    selected_body_ids, selected_body_names, eef_body_id = get_selected_body_ids(sim)
    print(
        f"Selected bodies for cfrc_ext diagnostics ({len(selected_body_ids)}): {selected_body_names}"
    )

    state_data = collect_force_from_state_replay(
        env, states, arm_dof_indices, selected_body_ids
    )
    print_signal_stats("qfrc_arm_norm(state_replay)", state_data["qfrc_arm_norm"])
    print_signal_stats(
        "cfrc_ext_norm_global(state_replay)", state_data["cfrc_ext_norm_global"]
    )
    low_dynamic_range_warning(
        "qfrc_arm_norm(state_replay)", state_data["qfrc_arm_norm"]
    )
    low_dynamic_range_warning(
        "cfrc_ext_norm_global(state_replay)", state_data["cfrc_ext_norm_global"]
    )

    action_data = None
    if args.mode in ["actions", "both"]:
        actions = LU.get_episode_actions(dataset_root, ep_idx)
        t_action = min(len(actions), len(states) - 1)
        if t_action <= 0:
            raise RuntimeError(
                "Not enough states/actions to run action rollout comparison"
            )
        action_data = collect_force_from_action_rollout(
            env,
            states[: t_action + 1],
            actions[:t_action],
            arm_dof_indices,
            selected_body_ids,
        )
        print_signal_stats(
            "qfrc_arm_norm(action_rollout)", action_data["qfrc_arm_norm"]
        )
        print_signal_stats(
            "cfrc_ext_norm_global(action_rollout)",
            action_data["cfrc_ext_norm_global"],
        )
        print_signal_stats(
            "state_drift_comparable(action_rollout)",
            action_data["state_drift_comparable"],
        )
        low_dynamic_range_warning(
            "qfrc_arm_norm(action_rollout)", action_data["qfrc_arm_norm"]
        )
        low_dynamic_range_warning(
            "cfrc_ext_norm_global(action_rollout)", action_data["cfrc_ext_norm_global"]
        )

    if args.mode == "both" and action_data is not None:
        summarize_comparison(state_data, action_data)

    if args.plot:
        plot_debug_signals(
            mode=args.mode,
            state_data=state_data,
            action_data=action_data,
            plot_raw=args.plot_raw,
        )

    if args.mode == "actions" and action_data is not None:
        q_for_viz = normalize_to_minus1_1(action_data["qfrc_arm_norm"])
        c_for_viz = normalize_to_minus1_1(action_data["cfrc_ext_norm_global"])
        q_label = "qfrc_constraint_arm (action rollout, normalized)"
        c_label = "cfrc_ext_global (action rollout, normalized)"
    else:
        q_for_viz = normalize_to_minus1_1(state_data["qfrc_arm_norm"])
        c_for_viz = normalize_to_minus1_1(state_data["cfrc_ext_norm_global"])
        q_label = "qfrc_constraint_arm (state replay, normalized)"
        c_label = "cfrc_ext_global (state replay, normalized)"

    x = np.arange(min(len(q_for_viz), len(c_for_viz), len(frames)))
    q_for_viz = q_for_viz[: len(x)]
    c_for_viz = c_for_viz[: len(x)]
    frames = frames[: len(x)]

    out_path = Path(args.out)
    render_sync_video(
        out_path=out_path,
        frames=frames,
        x=x,
        q_line=q_for_viz,
        c_line=c_for_viz,
        fps=int(args.fps),
        title=(
            f"{gym_id} | ep={ep_idx} | robots={robots} | split={split} | "
            f"layout={layout_id} style={style_id} | mode={args.mode} | xml={xml_loaded}"
        ),
        q_label=q_label,
        c_label=c_label,
    )

    if args.save_debug_npz:
        dbg_path = out_path.with_suffix(".npz")
        payload = {
            "state_saved_len": np.array([states.shape[1]], dtype=np.int32),
            "state_sim_len": np.array([len(sim.get_state().flatten())], dtype=np.int32),
            "state_comparable_len": np.array(
                [1 + sim.model.nq + sim.model.nv], dtype=np.int32
            ),
            "qfrc_arm_norm_state": state_data["qfrc_arm_norm"],
            "qfrc_arm_per_joint_state": state_data["qfrc_arm_per_joint"],
            "cfrc_ext_norm_global_state": state_data["cfrc_ext_norm_global"],
            "cfrc_ext_selected_state": state_data["cfrc_ext_selected"],
            "selected_body_ids": selected_body_ids,
            "selected_body_names": np.array(selected_body_names, dtype="S"),
            "eef_body_id": np.array(
                [-1 if eef_body_id is None else eef_body_id], dtype=np.int32
            ),
            "cfrc_eef_state": (
                state_data["cfrc_ext_selected"][:, 0, :]
                if len(selected_body_ids) > 0
                else np.zeros((len(state_data["qfrc_arm_norm"]), 6), dtype=np.float32)
            ),
        }

        if action_data is not None:
            payload.update(
                {
                    "qfrc_arm_norm_action": action_data["qfrc_arm_norm"],
                    "qfrc_arm_per_joint_action": action_data["qfrc_arm_per_joint"],
                    "cfrc_ext_norm_global_action": action_data["cfrc_ext_norm_global"],
                    "cfrc_ext_selected_action": action_data["cfrc_ext_selected"],
                    "state_drift_comparable": action_data["state_drift_comparable"],
                    "cfrc_eef_action": (
                        action_data["cfrc_ext_selected"][:, 0, :]
                        if len(selected_body_ids) > 0
                        else np.zeros(
                            (len(action_data["qfrc_arm_norm"]), 6), dtype=np.float32
                        )
                    ),
                }
            )

        np.savez_compressed(dbg_path, **payload)
        print(f"Saved debug traces -> {dbg_path}")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="LeRobot dataset root (contains meta/, data/, videos/, extras/)",
    )
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", type=str, default="./vids/episode_force_sync.mp4")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot_raw", action="store_true")
    p.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["state", "actions", "both"],
        help="How to compute force traces.",
    )
    p.add_argument(
        "--use_episode_xml",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to load exact per-episode model.xml.gz before replay.",
    )
    p.add_argument(
        "--save_debug_npz",
        action="store_true",
        help="Save raw traces and diagnostics next to output video.",
    )
    args = p.parse_args()
    main(args)
