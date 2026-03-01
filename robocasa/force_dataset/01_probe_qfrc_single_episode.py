import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
import robosuite

import robocasa
import robocasa.utils.lerobot_utils as LU


def reset_to(env, state, do_reset=False):
    if do_reset:
        env.reset()
    env.unwrapped.sim.set_state_from_flattened(state["states"])
    env.unwrapped.sim.forward()
    return env.unwrapped._get_observations()


def main(args):
    dataset_path = Path(args.dataset)
    assert dataset_path.exists()

    ep_idx = args.episode
    print(f"\n=== Probing Episode {ep_idx} ===")

    # ----------------------------
    # Load dataset + episode metadata
    # ----------------------------
    print("\n=== Loading episode metadata ===")
    env_meta = LU.get_env_metadata(dataset_path)
    env_name = env_meta["env_name"]

    ep_meta = LU.get_episode_meta(dataset_path, ep_idx)

    # ----------------------------
    # Reconstruct identical environment
    # ----------------------------
    print("Reconstructing environment with episode metadata...")

    env = gym.make(
        f"robocasa/{env_name}",
        robots=env_meta["env_kwargs"]["robots"],
        layout_ids=ep_meta["layout_id"],
        style_ids=ep_meta["style_id"],
        split=env_meta["env_kwargs"]["obj_instance_split"],
        seed=0,
    )

    # ----------------------------
    # Load recorded MuJoCo states
    # ----------------------------
    states = LU.get_episode_states(dataset_path, ep_idx)
    print("States shape:", states.shape)

    # Reset to first state
    reset_to(env, {"states": states[0]}, do_reset=True)
    sim = env.unwrapped.sim

    print("\nMuJoCo dimensions:")
    print("  nq:", sim.model.nq)
    print("  nv:", sim.model.nv)
    print("  nbody:", sim.model.nbody)

    # ----------------------------
    # Arm DOF indices (exclude gripper)
    # ----------------------------
    robot = env.unwrapped.robots[0]
    arm_dof_indices = np.array(robot.arm_joint_indexes, dtype=int)

    print("\nArm DOF indices:", arm_dof_indices)
    print("Number of arm DOFs:", len(arm_dof_indices))

    # ----------------------------
    # Probe forces
    # ----------------------------
    print("\n=== Extracting force signals for first 10 timesteps ===")

    for t in range(min(10, len(states))):
        reset_to(env, {"states": states[t]}, do_reset=False)
        sim.forward()

        qfrc = sim.data.qfrc_constraint.copy()
        cfrc_ext = sim.data.cfrc_ext.copy()

        arm_qfrc = qfrc[arm_dof_indices]

        print(f"\nTimestep {t}")
        print("  qfrc_constraint shape:", qfrc.shape)
        print("  arm_qfrc shape:", arm_qfrc.shape)
        print("  cfrc_ext shape:", cfrc_ext.shape)

        # 1) prove the state is actually changing
        print("  qpos[0:12]:", sim.data.qpos[:12].copy())
        print("  qvel[0:12]:", sim.data.qvel[:12].copy())

        # 2) inspect the actual arm vector, not just its norm
        print("  arm_qfrc:", arm_qfrc)

        print("  ||qfrc||:", np.linalg.norm(qfrc))
        print("  ||arm_qfrc||:", np.linalg.norm(arm_qfrc))
        print("  ||cfrc_ext||:", np.linalg.norm(cfrc_ext))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    main(args)
