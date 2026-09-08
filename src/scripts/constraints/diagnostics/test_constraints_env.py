# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P1.1 verification: the pre-reset snapshot protocol of :class:`ConstraintsEnv`.

What is under test is the *ordering* inside :meth:`ConstraintsEnv.step`, not the simulator. So the
environment is the counter-driven stub in :mod:`stub_env`. On a terminal step the snapshot must hold
the counter as it was before the reset, while the returned tensors must already hold the counter of
the new episode.

The Isaac Sim app still has to be launched because ``lib.env.env`` imports ``omni`` at module level,
but no scene is created, so the check runs in seconds.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_constraints_env.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the ConstraintsEnv snapshot protocol.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of stub environments.")
parser.add_argument("--episode_length", type=int, default=3, help="Steps before a time-out.")
parser.add_argument("--num_steps", type=int, default=12, help="Number of steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch



from stub_env import StubConstraintsEnv


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    env = StubConstraintsEnv(args_cli.num_envs, args_cli.episode_length)
    action = torch.zeros(args_cli.num_envs, 1)

    failures = []
    checked_terminal = 0
    checked_non_terminal = 0

    for step in range(args_cli.num_steps):
        obs, states, constraint_states, reward, terminated, truncated, extras = env.step(action)

        final_obs = extras["final_observations"]
        final_cs = extras["final_constraint_states"]
        done = terminated | truncated

        # -- check 3: the snapshots are always published, whatever happened this step
        if final_obs is None or final_cs is None:
            failures.append(f"step {step}: snapshot missing")
            continue
        if final_obs.shape != obs.shape or final_cs.shape != constraint_states.shape:
            failures.append(f"step {step}: snapshot shape {tuple(final_cs.shape)} != {tuple(constraint_states.shape)}")
        if torch.isnan(final_obs).any() or torch.isnan(final_cs).any():
            failures.append(f"step {step}: snapshot contains NaN")

        for i in range(args_cli.num_envs):
            if done[i]:
                # -- check 2: pre-reset value in the snapshot, post-reset value in the return
                checked_terminal += 1
                if final_cs[i].item() != float(args_cli.episode_length):
                    failures.append(
                        f"step {step} env {i}: final_constraint_states={final_cs[i].item()},"
                        f" expected {float(args_cli.episode_length)} (pre-reset counter)"
                    )
                if constraint_states[i].item() != 0.0:
                    failures.append(
                        f"step {step} env {i}: constraint_states={constraint_states[i].item()},"
                        f" expected 0.0 (post-reset counter)"
                    )
                if final_obs[i].item() != final_cs[i].item():
                    failures.append(f"step {step} env {i}: final_observations and final_constraint_states disagree")
            else:
                # -- check 1: nothing was reset, so the snapshot equals the returned value
                checked_non_terminal += 1
                if final_cs[i].item() != constraint_states[i].item():
                    failures.append(
                        f"step {step} env {i}: final_constraint_states={final_cs[i].item()}"
                        f" != constraint_states={constraint_states[i].item()} on a non-terminal step"
                    )
                if final_obs[i].item() != obs[i].item():
                    failures.append(f"step {step} env {i}: final_observations != observations on a non-terminal step")

    emit("")
    emit(f"[P1.1] non-terminal transitions checked : {checked_non_terminal}")
    emit(f"[P1.1] terminal transitions checked     : {checked_terminal}")
    if checked_terminal == 0:
        failures.append("no terminal step was exercised; raise --num_steps")

    if failures:
        emit(f"[P1.1] FAILED ({len(failures)})")
        for f in failures[:20]:
            emit(f"        - {f}")
    else:
        emit("[P1.1] PASSED: snapshots hold the pre-reset state and are always populated.")

    # kit swallows stdout in some launch modes, so the verdict is also written to disk
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_1_result.txt"), "w") as fh:
        fh.write(chr(10).join(report) + chr(10))
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
