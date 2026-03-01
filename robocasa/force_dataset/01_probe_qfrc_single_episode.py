#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

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
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def sample_forces(env):
    d = env.sim.data
    return {
        "qfrc_constraint": np.array(d.qfrc_constraint, dtype=np.float64).copy(),
        "qfrc_actuator": np.array(d.qfrc_actuator, dtype=np.float64).copy(),
        "qfrc_bias": np.array(d.qfrc_bias, dtype=np.float64).copy(),
        "qfrc_passive": np.array(d.qfrc_passive, dtype=np.float64).copy(),
        "cfrc_ext": np.array(d.cfrc_ext, dtype=np.float64).copy(),  # [nbody, 6]
        "ncon": int(d.ncon),
    }


def summarize(name, arr):
    flat = arr.reshape(-1)
    return {
        "name": name,
        "shape": list(arr.shape),
        "nonzero_ratio": float(np.mean(np.abs(flat) > 1e-10)),
        "max_abs": float(np.max(np.abs(flat))),
        "mean_abs": float(np.mean(np.abs(flat))),
    }


def run_probe(dataset_root: Path, episode: int, replay_mode: str):
    env = make_env(dataset_root)
    try:
        states = LU.get_episode_states(dataset_root, episode)
        actions = LU.get_episode_actions(dataset_root, episode, abs_actions=False)

        if len(states) != len(actions):
            raise RuntimeError(
                f"states/actions length mismatch: {len(states)} vs {len(actions)}"
            )

        initial_state = {
            "states": states[0],
            "model": LU.get_episode_model_xml(dataset_root, episode),
            "ep_meta": json.dumps(LU.get_episode_meta(dataset_root, episode)),
        }
        reset_to(env, initial_state)

        result = {
            "episode": episode,
            "T": int(len(states)),
            "nv": int(env.sim.model.nv),
            "nbody": int(env.sim.model.nbody),
            "njnt": int(env.sim.model.njnt),
            "dof_joint_ids": env.sim.model.dof_jntid.astype(int).tolist(),
            "joint_names": [
                env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)
            ],
            "body_names": [
                env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)
            ],
            "state_replay": None,
            "action_replay": None,
        }

        if replay_mode in ("state", "both"):
            qfrc_constraint = []
            qfrc_actuator = []
            qfrc_bias = []
            qfrc_passive = []
            cfrc_ext = []
            ncon = []

            for t in range(len(states)):
                reset_to(env, {"states": states[t]})
                f = sample_forces(env)
                qfrc_constraint.append(f["qfrc_constraint"])
                qfrc_actuator.append(f["qfrc_actuator"])
                qfrc_bias.append(f["qfrc_bias"])
                qfrc_passive.append(f["qfrc_passive"])
                cfrc_ext.append(f["cfrc_ext"])
                ncon.append(f["ncon"])

            qfrc_constraint = np.stack(qfrc_constraint, axis=0)
            qfrc_actuator = np.stack(qfrc_actuator, axis=0)
            qfrc_bias = np.stack(qfrc_bias, axis=0)
            qfrc_passive = np.stack(qfrc_passive, axis=0)
            cfrc_ext = np.stack(cfrc_ext, axis=0)
            ncon = np.asarray(ncon, dtype=np.int64)

            result["state_replay"] = {
                "summary": [
                    summarize("qfrc_constraint", qfrc_constraint),
                    summarize("qfrc_actuator", qfrc_actuator),
                    summarize("qfrc_bias", qfrc_bias),
                    summarize("qfrc_passive", qfrc_passive),
                    summarize("cfrc_ext", cfrc_ext),
                    summarize("ncon", ncon.astype(np.float64)),
                ]
            }

        if replay_mode in ("action", "both"):
            qfrc_constraint = []
            qfrc_actuator = []
            qfrc_bias = []
            qfrc_passive = []
            cfrc_ext = []
            ncon = []
            divergence = np.full((len(states),), np.nan, dtype=np.float64)

            for t in range(len(states)):
                reset_to(env, {"states": states[t]})
                env.step(actions[t])  # restore-state -> play-action -> record-force
                f = sample_forces(env)

                qfrc_constraint.append(f["qfrc_constraint"])
                qfrc_actuator.append(f["qfrc_actuator"])
                qfrc_bias.append(f["qfrc_bias"])
                qfrc_passive.append(f["qfrc_passive"])
                cfrc_ext.append(f["cfrc_ext"])
                ncon.append(f["ncon"])

                if t < len(states) - 1:
                    s_next = np.array(env.sim.get_state().flatten(), dtype=np.float64)
                    divergence[t] = float(np.linalg.norm(s_next - states[t + 1]))

            qfrc_constraint = np.stack(qfrc_constraint, axis=0)
            qfrc_actuator = np.stack(qfrc_actuator, axis=0)
            qfrc_bias = np.stack(qfrc_bias, axis=0)
            qfrc_passive = np.stack(qfrc_passive, axis=0)
            cfrc_ext = np.stack(cfrc_ext, axis=0)
            ncon = np.asarray(ncon, dtype=np.int64)

            valid_div = divergence[np.isfinite(divergence)]
            result["action_replay"] = {
                "summary": [
                    summarize("qfrc_constraint", qfrc_constraint),
                    summarize("qfrc_actuator", qfrc_actuator),
                    summarize("qfrc_bias", qfrc_bias),
                    summarize("qfrc_passive", qfrc_passive),
                    summarize("cfrc_ext", cfrc_ext),
                    summarize("ncon", ncon.astype(np.float64)),
                ],
                "divergence_l2": {
                    "count": int(valid_div.size),
                    "mean": float(valid_div.mean()) if valid_div.size else None,
                    "max": float(valid_div.max()) if valid_div.size else None,
                    "p95": float(np.quantile(valid_div, 0.95))
                    if valid_div.size
                    else None,
                },
            }

        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to lerobot dataset root"
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="both",
        choices=["state", "action", "both"],
        help="Probe mode",
    )
    parser.add_argument(
        "--out-json", type=str, default=None, help="Optional output report json"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    report = run_probe(dataset_root, args.episode, args.replay_mode)

    print(json.dumps(report, indent=2))
    if args.out_json is not None:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

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
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def sample_forces(env):
    d = env.sim.data
    return {
        "qfrc_constraint": np.array(d.qfrc_constraint, dtype=np.float64).copy(),
        "qfrc_actuator": np.array(d.qfrc_actuator, dtype=np.float64).copy(),
        "qfrc_bias": np.array(d.qfrc_bias, dtype=np.float64).copy(),
        "qfrc_passive": np.array(d.qfrc_passive, dtype=np.float64).copy(),
        "cfrc_ext": np.array(d.cfrc_ext, dtype=np.float64).copy(),  # [nbody, 6]
        "ncon": int(d.ncon),
    }


def summarize(name, arr):
    flat = arr.reshape(-1)
    return {
        "name": name,
        "shape": list(arr.shape),
        "nonzero_ratio": float(np.mean(np.abs(flat) > 1e-10)),
        "max_abs": float(np.max(np.abs(flat))),
        "mean_abs": float(np.mean(np.abs(flat))),
    }


def run_probe(dataset_root: Path, episode: int, replay_mode: str):
    env = make_env(dataset_root)
    try:
        states = LU.get_episode_states(dataset_root, episode)
        actions = LU.get_episode_actions(dataset_root, episode, abs_actions=False)

        if len(states) != len(actions):
            raise RuntimeError(
                f"states/actions length mismatch: {len(states)} vs {len(actions)}"
            )

        initial_state = {
            "states": states[0],
            "model": LU.get_episode_model_xml(dataset_root, episode),
            "ep_meta": json.dumps(LU.get_episode_meta(dataset_root, episode)),
        }
        reset_to(env, initial_state)

        result = {
            "episode": episode,
            "T": int(len(states)),
            "nv": int(env.sim.model.nv),
            "nbody": int(env.sim.model.nbody),
            "njnt": int(env.sim.model.njnt),
            "dof_joint_ids": env.sim.model.dof_jntid.astype(int).tolist(),
            "joint_names": [
                env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)
            ],
            "body_names": [
                env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)
            ],
            "state_replay": None,
            "action_replay": None,
        }

        if replay_mode in ("state", "both"):
            qfrc_constraint = []
            qfrc_actuator = []
            qfrc_bias = []
            qfrc_passive = []
            cfrc_ext = []
            ncon = []

            for t in range(len(states)):
                reset_to(env, {"states": states[t]})
                f = sample_forces(env)
                qfrc_constraint.append(f["qfrc_constraint"])
                qfrc_actuator.append(f["qfrc_actuator"])
                qfrc_bias.append(f["qfrc_bias"])
                qfrc_passive.append(f["qfrc_passive"])
                cfrc_ext.append(f["cfrc_ext"])
                ncon.append(f["ncon"])

            qfrc_constraint = np.stack(qfrc_constraint, axis=0)
            qfrc_actuator = np.stack(qfrc_actuator, axis=0)
            qfrc_bias = np.stack(qfrc_bias, axis=0)
            qfrc_passive = np.stack(qfrc_passive, axis=0)
            cfrc_ext = np.stack(cfrc_ext, axis=0)
            ncon = np.asarray(ncon, dtype=np.int64)

            result["state_replay"] = {
                "summary": [
                    summarize("qfrc_constraint", qfrc_constraint),
                    summarize("qfrc_actuator", qfrc_actuator),
                    summarize("qfrc_bias", qfrc_bias),
                    summarize("qfrc_passive", qfrc_passive),
                    summarize("cfrc_ext", cfrc_ext),
                    summarize("ncon", ncon.astype(np.float64)),
                ]
            }

        if replay_mode in ("action", "both"):
            qfrc_constraint = []
            qfrc_actuator = []
            qfrc_bias = []
            qfrc_passive = []
            cfrc_ext = []
            ncon = []
            divergence = np.full((len(states),), np.nan, dtype=np.float64)

            for t in range(len(states)):
                reset_to(env, {"states": states[t]})
                env.step(actions[t])  # restore-state -> play-action -> record-force
                f = sample_forces(env)

                qfrc_constraint.append(f["qfrc_constraint"])
                qfrc_actuator.append(f["qfrc_actuator"])
                qfrc_bias.append(f["qfrc_bias"])
                qfrc_passive.append(f["qfrc_passive"])
                cfrc_ext.append(f["cfrc_ext"])
                ncon.append(f["ncon"])

                if t < len(states) - 1:
                    s_next = np.array(env.sim.get_state().flatten(), dtype=np.float64)
                    divergence[t] = float(np.linalg.norm(s_next - states[t + 1]))

            qfrc_constraint = np.stack(qfrc_constraint, axis=0)
            qfrc_actuator = np.stack(qfrc_actuator, axis=0)
            qfrc_bias = np.stack(qfrc_bias, axis=0)
            qfrc_passive = np.stack(qfrc_passive, axis=0)
            cfrc_ext = np.stack(cfrc_ext, axis=0)
            ncon = np.asarray(ncon, dtype=np.int64)

            valid_div = divergence[np.isfinite(divergence)]
            result["action_replay"] = {
                "summary": [
                    summarize("qfrc_constraint", qfrc_constraint),
                    summarize("qfrc_actuator", qfrc_actuator),
                    summarize("qfrc_bias", qfrc_bias),
                    summarize("qfrc_passive", qfrc_passive),
                    summarize("cfrc_ext", cfrc_ext),
                    summarize("ncon", ncon.astype(np.float64)),
                ],
                "divergence_l2": {
                    "count": int(valid_div.size),
                    "mean": float(valid_div.mean()) if valid_div.size else None,
                    "max": float(valid_div.max()) if valid_div.size else None,
                    "p95": float(np.quantile(valid_div, 0.95))
                    if valid_div.size
                    else None,
                },
            }

        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to lerobot dataset root"
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="both",
        choices=["state", "action", "both"],
        help="Probe mode",
    )
    parser.add_argument(
        "--out-json", type=str, default=None, help="Optional output report json"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    report = run_probe(dataset_root, args.episode, args.replay_mode)

    print(json.dumps(report, indent=2))
    if args.out_json is not None:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()
