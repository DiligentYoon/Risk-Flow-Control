# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P1.2 verification: the re-parented ``G1BaseEnv`` copy.

Checks that re-declaring the parent as :class:`ConstraintsEnv` produced a working environment:
the linearisation is the intended single chain, the scene builds, actions reach the articulation,
and the snapshot protocol added by :class:`ConstraintsEnv` survives on the real G1 code path.

A minimal task subclass fills in the methods ``G1BaseEnv`` leaves abstract. Its constraint state is
``[episode_length_buf, root_height]``: the first element makes the pre-/post-reset distinction
unambiguous, since ``Env._reset_idx`` zeroes that buffer.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_g1_base_env.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the re-parented G1BaseEnv.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=40, help="Number of steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from lib.env.env import Env

from envs.constraints_env import ConstraintsEnv
from envs.G1.base.G1_base_env import G1BaseEnv
from envs.G1.base.G1_base_env_cfg import G1BaseEnvCfg

NUM_JOINTS = 29
CONSTRAINT_DIM = 2


@configclass
class SmokeEnvCfg(G1BaseEnvCfg):
    """Smallest configuration that makes G1BaseEnv instantiable."""

    num_agents = 1
    episode_length_s = 0.2  # 10 steps at decimation 4 / dt 0.005, so time-outs happen early
    action_scale_factor = 0.5
    action_space = NUM_JOINTS
    observation_space = NUM_JOINTS
    state_space = NUM_JOINTS
    constraint_state_space = CONSTRAINT_DIM


class SmokeEnv(G1BaseEnv):
    """Task subclass that implements only what G1BaseEnv leaves abstract."""

    cfg: SmokeEnvCfg

    def _compute_intermediate_values(self, env_ids=None):
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        self.joint_pos[i] = self._robot.data.joint_pos[i]
        self.root_height[i] = self._robot.data.root_pos_w[i, 2]

    def _get_observations(self):
        return self.joint_pos - self._robot.data.default_joint_pos

    def _get_states(self):
        return self.joint_pos - self._robot.data.default_joint_pos

    def _get_constraint_states(self):
        return torch.stack([self.episode_length_buf.float(), self.root_height], dim=-1)

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    def _setup_scene(self):
        super()._setup_scene()
        # G1BaseEnv only spawns the robot and its contact sensor. Cloning the environments and
        # adding the ground plane are left to the task environment -- without them the articulation
        # view matches a single prim while the scene claims num_envs, and the first indexed write
        # after construction trips an out-of-bounds assert.
        self.scene.clone_environments(copy_from_source=False)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # allocate the cached values this task reads; done here because _setup_scene runs before
        # Env.__init__ finishes and before the first _compute_intermediate_values call
        self.joint_pos = torch.zeros(self.scene.num_envs, NUM_JOINTS, device=self.sim.device)
        self.root_height = torch.zeros(self.scene.num_envs, device=self.sim.device)


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []

    # -- check 1: the inheritance chain is the intended single chain
    mro = list(SmokeEnv.__mro__)
    emit("[P1.2] MRO: " + " -> ".join(c.__name__ for c in mro[:5]))
    expected_head = [SmokeEnv, G1BaseEnv, ConstraintsEnv, Env]
    if mro[: len(expected_head)] != expected_head:
        failures.append(f"MRO head is {[c.__name__ for c in mro[:4]]}, expected "
                        f"{[c.__name__ for c in expected_head]}")

    # -- check 2: the scene builds and reset returns the constraint states
    cfg = SmokeEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = SmokeEnv(cfg)

    obs, states, constraint_states, extras = env.reset()
    emit(f"[P1.2] reset: obs={tuple(obs.shape)} states={tuple(states.shape)} "
         f"constraint_states={tuple(constraint_states.shape)}")
    if tuple(constraint_states.shape) != (args_cli.num_envs, CONSTRAINT_DIM):
        failures.append(f"reset constraint_states shape {tuple(constraint_states.shape)}")
    if "final_constraint_states" not in extras or "final_observations" not in extras:
        failures.append("reset extras is missing the final_* channels")

    # -- check 3: stepping works and the snapshot protocol holds on the real code path
    action = torch.zeros(args_cli.num_envs, NUM_JOINTS, device=env.device)
    num_timeouts = 0
    for step in range(args_cli.num_steps):
        obs, states, constraint_states, reward, terminated, truncated, extras = env.step(action)
        final_cs = extras["final_constraint_states"]

        if torch.isnan(obs).any() or torch.isnan(constraint_states).any():
            failures.append(f"step {step}: NaN in the returned tensors")
        if final_cs is None or tuple(final_cs.shape) != (args_cli.num_envs, CONSTRAINT_DIM):
            failures.append(f"step {step}: bad final_constraint_states")
            continue

        done = terminated | truncated
        for i in range(args_cli.num_envs):
            # element 0 of the constraint state is episode_length_buf, zeroed by Env._reset_idx
            if done[i]:
                num_timeouts += 1
                if final_cs[i, 0].item() == 0.0:
                    failures.append(f"step {step} env {i}: snapshot was taken after the reset")
                if constraint_states[i, 0].item() != 0.0:
                    failures.append(f"step {step} env {i}: returned state is not the post-reset one")
            elif final_cs[i, 0].item() != constraint_states[i, 0].item():
                failures.append(f"step {step} env {i}: snapshot differs on a non-terminal step")

    # -- check that actions actually reach the articulation without clipping
    probe = torch.full((args_cli.num_envs, NUM_JOINTS), 0.3, device=env.device)
    env._pre_physics_step(probe)
    if not torch.allclose(env.processed_actions, probe * cfg.action_scale_factor):
        failures.append("processed_actions != actions * action_scale_factor")

    emit(f"[P1.2] steps run: {args_cli.num_steps}, terminal transitions seen: {num_timeouts}")
    if num_timeouts == 0:
        failures.append("no time-out was exercised; raise --num_steps")

    if failures:
        emit(f"[P1.2] FAILED ({len(failures)})")
        for f in failures[:20]:
            emit(f"        - {f}")
    else:
        emit("[P1.2] PASSED: single-chain MRO, scene builds, snapshots hold on the G1 path.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_2_result.txt"), "w") as fh:
        fh.write(chr(10).join(report) + chr(10))

    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
