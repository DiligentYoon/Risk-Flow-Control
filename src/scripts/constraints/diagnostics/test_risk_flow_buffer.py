# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P3.2 verification: :class:`RiskFlowBuffer`.

Checks that the storage is trustworthy before any learning depends on it: no NaN survives
initialization, what goes in comes back bit-exact, the circular overwrite drops exactly the oldest
rows, the flat sampling index maps to the storage cell it claims to, and the production-sized
buffer fits in the GPU budget it has to share with the simulator and the models.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_risk_flow_buffer.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify RiskFlowBuffer.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import numpy as np
import torch

from buffer.risk_flow_buffer import RiskFlowBuffer

CONSTRAINT_DIM = 67
OBS_DIM = 96
NUM_ACTIONS = 29

ROWS = 32
NUM_ENVS = 8

# production sizing from research.md 5
PROD_ROWS = 512
PROD_NUM_ENVS = 2048
GPU_BUDGET_FRACTION = 0.25


def _box(dim):
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _make_buffer(rows, num_envs, device):
    buffer = RiskFlowBuffer(buffer_size=rows, num_envs=num_envs, device=device)
    buffer.init_buffer(_box(OBS_DIM), _box(CONSTRAINT_DIM), _box(NUM_ACTIONS))
    return buffer


def _sample(step, num_envs, device):
    """A transition whose every entry encodes the step it was written at."""
    base = torch.arange(num_envs, device=device, dtype=torch.float32).unsqueeze(-1)
    tag = base + 1000.0 * step
    return dict(
        observations=tag.expand(num_envs, OBS_DIM).contiguous(),
        constraint_states=(tag + 0.5).expand(num_envs, CONSTRAINT_DIM).contiguous(),
        actions=(tag + 0.25).expand(num_envs, NUM_ACTIONS).contiguous(),
        final_observations=(tag + 0.75).expand(num_envs, OBS_DIM).contiguous(),
        final_constraint_states=(tag + 0.125).expand(num_envs, CONSTRAINT_DIM).contiguous(),
        terminated=torch.full((num_envs,), step % 2 == 0, dtype=torch.bool, device=device),
        truncated=torch.full((num_envs,), step % 3 == 0, dtype=torch.bool, device=device),
    )


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit(f"[P3.2] device={device}")

    buffer = _make_buffer(ROWS, NUM_ENVS, device)

    # -- check 1: initialization leaves no NaN behind
    names = tuple(buffer.tensors.keys())
    emit(f"[P3.2] tensors: {list(names)}")
    if len(names) != 7:
        failures.append(f"expected 7 tensors, found {len(names)}")
    nan_tensors = [
        name for name, tensor in buffer.tensors.items()
        if torch.is_floating_point(tensor) and torch.isnan(tensor).any()
    ]
    if nan_tensors:
        failures.append(f"NaN present right after init_buffer: {nan_tensors}")

    # -- check 2: round-trip, bit-exact
    written = [_sample(step, NUM_ENVS, device) for step in range(ROWS)]
    for sample in written:
        buffer.add_samples(**sample)

    if len(buffer) != ROWS * NUM_ENVS:
        failures.append(f"len(buffer)={len(buffer)}, expected {ROWS * NUM_ENVS}")
    if not buffer.filled:
        failures.append("the buffer did not report itself as filled after exactly `rows` inserts")

    mismatched = []
    for name in names:
        stored = buffer.get_tensor_by_name(name)
        expected = torch.stack([sample[name] if sample[name].ndim == 2
                                else sample[name].unsqueeze(-1) for sample in written])
        if not torch.equal(stored, expected):
            mismatched.append(name)
    emit(f"[P3.2] round-trip mismatched tensors: {mismatched}")
    if mismatched:
        failures.append(f"round-trip is not bit-exact for {mismatched}")

    # -- check 4: the flat sampling index maps where it claims to
    #    the values encode (step, env), so a wrong mapping is visible rather than merely unequal
    flat = buffer.get_tensor_by_name("observations", keepdim=False)
    keep = buffer.get_tensor_by_name("observations")
    bad_indexes = []
    for index in (0, 1, NUM_ENVS - 1, NUM_ENVS, NUM_ENVS + 3, len(buffer) - 1):
        row, env = index // NUM_ENVS, index % NUM_ENVS
        if not torch.equal(flat[index], keep[row, env]):
            bad_indexes.append(index)
        if flat[index, 0].item() != 1000.0 * row + env:
            bad_indexes.append(index)
    emit(f"[P3.2] flat index i <-> (i // num_envs, i % num_envs): "
         f"{'consistent' if not bad_indexes else bad_indexes}")
    if bad_indexes:
        failures.append(f"flat index does not match (row, env) at {sorted(set(bad_indexes))}")

    # -- check 3: circular overwrite drops exactly the oldest rows
    overflow = 10
    for step in range(ROWS, ROWS + overflow):
        buffer.add_samples(**_sample(step, NUM_ENVS, device))

    stored = buffer.get_tensor_by_name("observations")
    survived, lost, misplaced = [], [], []
    for step in range(ROWS + overflow):
        row = step % ROWS
        # the newest write to `row` wins; every step older than that must be gone
        newest = max(candidate for candidate in range(ROWS + overflow) if candidate % ROWS == row)
        expected = 1000.0 * step + 0.0
        present = stored[row, 0, 0].item() == expected
        if step == newest:
            survived.append(step) if present else misplaced.append(step)
        else:
            lost.append(step) if not present else misplaced.append(step)

    emit(f"[P3.2] after {ROWS + overflow} inserts into {ROWS} rows: "
         f"{len(survived)} newest kept, {len(lost)} oldest dropped, {len(misplaced)} misplaced")
    if misplaced:
        failures.append(f"circular overwrite kept/dropped the wrong steps: {misplaced[:5]}")
    if sorted(lost) != list(range(overflow)):
        failures.append(f"the dropped steps are {sorted(lost)[:12]}, expected 0..{overflow - 1}")
    if len(buffer) != ROWS * NUM_ENVS:
        failures.append(f"len(buffer)={len(buffer)} after wrap-around, expected {ROWS * NUM_ENVS}")

    # sampling has to keep working across the wrap
    batch = buffer.sample_batch(("observations", "final_constraint_states", "terminated"), batch_size=128)
    shapes = [tuple(tensor.shape) for tensor in batch]
    emit(f"[P3.2] sample_batch shapes: {shapes}")
    if shapes != [(128, OBS_DIM), (128, CONSTRAINT_DIM), (128, 1)]:
        failures.append(f"sample_batch returned {shapes}")
    if any(torch.isnan(tensor).any() for tensor in batch if torch.is_floating_point(tensor)):
        failures.append("sample_batch returned NaN")

    # -- check 5: the production-sized buffer against the GPU budget
    per_transition = 2 * OBS_DIM + 2 * CONSTRAINT_DIM + NUM_ACTIONS
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
    production = _make_buffer(PROD_ROWS, PROD_NUM_ENVS, device)
    measured = production.memory_bytes
    line = (f"[P3.2] production buffer ({PROD_ROWS} x {PROD_NUM_ENVS} x {per_transition} float32 "
            f"+ 2 bool): {measured / 2 ** 30:.3f} GiB")
    if device.startswith("cuda"):
        allocated = torch.cuda.memory_allocated() - before
        total = torch.cuda.get_device_properties(0).total_memory
        line += (f", allocated {allocated / 2 ** 30:.3f} GiB of {total / 2 ** 30:.1f} GiB "
                 f"({100.0 * allocated / total:.1f}%)")
        if allocated > GPU_BUDGET_FRACTION * total:
            failures.append(
                f"the replay buffer takes {100.0 * allocated / total:.1f}% of the GPU; the simulator "
                f"and the models share it -- lower `rows` or share the observation tensor")
    emit(line)
    del production
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if failures:
        emit(f"[P3.2] FAILED ({len(failures)})")
        for failure in failures[:20]:
            emit(f"        - {failure}")
    else:
        emit("[P3.2] PASSED: NaN-free init, bit-exact round-trip, correct wrap-around and indexing.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p3_2_result.txt"), "w") as handle:
        handle.write(chr(10).join(report) + chr(10))

    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
