# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P1.4 verification: :class:`G1RiskEnv`.

Checks the four things the task environment has to get right before any learning code touches it:
the inheritance chain and the constraint hook, the episode length that doubles as the recovery
deadline H, the single-agent action path, and the absence of any gait-scheduler leftovers.

The dataset reset event is disabled here: the risk-bucket files are produced in P0.3, and this
check is about the environment mechanics, not about the initial-state distribution.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_g1_risk_env.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify G1RiskEnv.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=330, help="Number of steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from lib.env.env import Env

import envs  # noqa: F401  -- triggers the gym registrations
from envs.constraints_env import ConstraintsEnv
from envs.G1.base.G1_base_env import G1BaseEnv
from envs.G1.risk.G1_risk_env import G1RiskEnv
from envs.G1.risk.G1_risk_env_cfg import G1RiskEnvCfg

NUM_JOINTS = 29
CONSTRAINT_DIM = 67


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []

    # -- check 1a: single inheritance chain
    mro = list(G1RiskEnv.__mro__)
    emit("[P1.4] MRO: " + " -> ".join(c.__name__ for c in mro[:4]))
    if mro[:4] != [G1RiskEnv, G1BaseEnv, ConstraintsEnv, Env]:
        failures.append(f"MRO head is {[c.__name__ for c in mro[:4]]}")

    # -- the task is registered and reachable by id
    if "G1-risk" not in gym.registry:
        failures.append("G1-risk is not registered")

    cfg = G1RiskEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    # P0.3 produces the risk-bucket dataset; until then start from the default pose
    cfg.events.reset_state_from_dataset = None

    env = G1RiskEnv(cfg)

    # -- check 1b: the constraint hook actually runs, once per step and once per reset
    calls = {"n": 0}
    original_hook = env._get_constraint_states

    def counting_hook():
        calls["n"] += 1
        return original_hook()

    env._get_constraint_states = counting_hook

    obs, states, constraint_states, extras = env.reset()
    emit(f"[P1.4] reset: obs={tuple(obs.shape)} states={states} "
         f"constraint_states={tuple(constraint_states.shape)}")
    if tuple(constraint_states.shape) != (args_cli.num_envs, CONSTRAINT_DIM):
        failures.append(f"constraint_states shape {tuple(constraint_states.shape)}, expected "
                        f"({args_cli.num_envs}, {CONSTRAINT_DIM})")
    if states is not None:
        failures.append("single-agent _get_states should be None; the critic reads constraint states")
    if tuple(obs.shape) != (args_cli.num_envs, cfg.observation_space):
        failures.append(f"observation shape {tuple(obs.shape)} disagrees with cfg.observation_space "
                        f"({cfg.observation_space})")

    # -- check 3: single-agent action path
    if not isinstance(env.prev_actions, torch.Tensor):
        failures.append("prev_actions is not a tensor; the multi-agent branch was taken")
    elif tuple(env.prev_actions.shape) != (args_cli.num_envs, NUM_JOINTS):
        failures.append(f"prev_actions shape {tuple(env.prev_actions.shape)}")
    if list(env._joint_dof_ids) != list(range(NUM_JOINTS)):
        failures.append(f"_joint_dof_ids is not the native 0..{NUM_JOINTS - 1} order")

    probe = torch.full((args_cli.num_envs, NUM_JOINTS), 0.3, device=env.device)
    env._pre_physics_step(probe)
    if not torch.allclose(env.processed_actions, probe * cfg.action_scale_factor):
        failures.append("processed_actions != actions * action_scale_factor (clipping crept in?)")

    # -- check 4: no gait-scheduler leftovers
    for attr in ("phase", "phase_sin", "phase_cos", "support_foot_pos", "contact_schedule"):
        if hasattr(env, attr):
            failures.append(f"gait-scheduler attribute '{attr}' is still present")
    if getattr(env.cfg, "commands", None) is not None:
        failures.append("cfg.commands is set; the intervention env has no velocity command")

    # -- check 2: the episode length is the recovery deadline H
    expected_period = env.max_episode_length - 1
    emit(f"[P1.4] max_episode_length={env.max_episode_length} (episode_length_s="
         f"{cfg.episode_length_s}, step_dt={env.step_dt:.4f})")

    action = torch.zeros(args_cli.num_envs, NUM_JOINTS, device=env.device)
    last_truncate = [-1] * args_cli.num_envs
    periods = []
    num_terminated = 0

    # Phase A keeps the real termination so the fall path is exercised; phase B suppresses it,
    # because a robot standing in the default pose under a zero action just collapses and would
    # never reach the deadline that this check is about.
    def dones_without_death():
        env._compute_intermediate_values()
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    for step in range(args_cli.num_steps):
        if step == args_cli.num_steps // 3:
            env._get_dones = dones_without_death

        obs, states, constraint_states, reward, terminated, truncated, extras = env.step(action)

        if torch.any(terminated & truncated):
            failures.append(f"step {step}: terminated and truncated are both set")
        if torch.isnan(constraint_states).any():
            failures.append(f"step {step}: NaN in constraint_states")
        if extras["final_constraint_states"] is None:
            failures.append(f"step {step}: final_constraint_states missing")

        num_terminated += int(terminated.sum().item())
        for i in range(args_cli.num_envs):
            if truncated[i]:
                if last_truncate[i] >= 0:
                    periods.append(step - last_truncate[i])
                last_truncate[i] = step

    if calls["n"] == 0:
        failures.append("_get_constraint_states was never called")

    unique_periods = sorted(set(periods))
    emit(f"[P1.4] truncation periods observed: {unique_periods} (expected [{expected_period}])")
    emit(f"[P1.4] terminations (phase A): {num_terminated}, constraint-hook calls: {calls['n']}")
    if num_terminated == 0:
        failures.append("the fall termination never fired; is the contact wiring right?")
    if not periods:
        failures.append("no full episode was observed; raise --num_steps")
    elif unique_periods != [expected_period]:
        failures.append(f"truncation period {unique_periods}, expected [{expected_period}]")

    if failures:
        emit(f"[P1.4] FAILED ({len(failures)})")
        for f in failures[:20]:
            emit(f"        - {f}")
    else:
        emit("[P1.4] PASSED: single chain, H-length episodes, single-agent path, no gait leftovers.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_4_result.txt"), "w") as fh:
        fh.write(chr(10).join(report) + chr(10))

    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
