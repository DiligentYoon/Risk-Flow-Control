# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P1.3 verification: :class:`ConstraintsWrapper` relays the 7-tuple unchanged.

The same stub environment as P1.1 is run twice, unwrapped and wrapped, from an identical initial
state. Every tensor the wrapper passes through must be bit-exact, and the constraint-state space
must be queryable from the wrapper without reaching into the unwrapped environment.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_constraints_wrapper.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify ConstraintsWrapper.")
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

from wrappers.constraints_wrapper import ConstraintsWrapper

from stub_env import StubConstraintsEnv


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []

    raw = StubConstraintsEnv(args_cli.num_envs, args_cli.episode_length)
    wrapped = ConstraintsWrapper(StubConstraintsEnv(args_cli.num_envs, args_cli.episode_length))

    # -- check 1: the constraint-state space is reachable from the wrapper
    space = wrapped.constraint_state_space
    emit(f"[P1.3] constraint_state_space: {space}")
    if space is None or space.shape != (raw.cfg.constraint_state_space,):
        failures.append(f"constraint_state_space is {space}")

    # -- check 2: reset relays the 4-tuple
    r_obs, r_states, r_cs, r_info = raw.reset()
    w_obs, w_states, w_cs, w_info = wrapped.reset()
    for name, a, b in (("obs", r_obs, w_obs), ("states", r_states, w_states), ("constraint_states", r_cs, w_cs)):
        if not torch.equal(a, b):
            failures.append(f"reset {name} differs through the wrapper")
    for key in ("final_observations", "final_constraint_states"):
        if key not in w_info:
            failures.append(f"reset info is missing {key}")

    # -- check 3: every step is relayed bit-exact
    action = torch.zeros(args_cli.num_envs, 1)
    checked = 0
    for step in range(args_cli.num_steps):
        r_obs, r_states, r_cs, r_rew, r_term, r_trunc, r_info = raw.step(action)
        w_obs, w_states, w_cs, w_rew, w_term, w_trunc, w_info = wrapped.step(action)

        pairs = [
            ("obs", r_obs, w_obs),
            ("states", r_states, w_states),
            ("constraint_states", r_cs, w_cs),
            ("final_observations", r_info["final_observations"], w_info["final_observations"]),
            ("final_constraint_states", r_info["final_constraint_states"], w_info["final_constraint_states"]),
        ]
        for name, a, b in pairs:
            if not torch.equal(a, b):
                failures.append(f"step {step}: {name} differs through the wrapper")

        # the wrapper reshapes the scalar channels to (E, 1), matching IsaacLabWrapper
        for name, a, b in (("reward", r_rew, w_rew), ("terminated", r_term, w_term), ("truncated", r_trunc, w_trunc)):
            if tuple(b.shape) != (args_cli.num_envs, 1):
                failures.append(f"step {step}: {name} shape {tuple(b.shape)}, expected (E, 1)")
            if not torch.equal(a.reshape(-1, 1), b):
                failures.append(f"step {step}: {name} value differs through the wrapper")
        checked += 1

    emit(f"[P1.3] steps compared: {checked}")

    if failures:
        emit(f"[P1.3] FAILED ({len(failures)})")
        for f in failures[:20]:
            emit(f"        - {f}")
    else:
        emit("[P1.3] PASSED: the wrapper relays every channel bit-exact and exposes the space.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_3_result.txt"), "w") as fh:
        fh.write(chr(10).join(report) + chr(10))

    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
