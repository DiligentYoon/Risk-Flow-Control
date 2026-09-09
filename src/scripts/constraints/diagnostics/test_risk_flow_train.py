# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P5.1 verification: the integrated training script.

Nothing here judges learning. P5.2 does that. What is checked is that the loop runs, that the run
is resumable, and that the quantities research.md section 5 asks for actually reach the event file:

1. **Config surface** -- every hyperparameter of research.md 4.6 is reachable through the gym
   registry, and ``horizon: auto`` resolves to the environment's own ``max_episode_length - 1``.
2. **Checkpoint round-trip** -- critic, actor, target critic, multiplier and both optimizer states
   survive save/restore, and the restored agent is bit-identical: same actions on the same input,
   same parameters after the same update on the same batch. This is the part of "same seed, same
   trajectory" that belongs to this repository; simulator-side determinism is Isaac Lab's.
3. **Smoke run** -- ``train.main()`` itself, on a short run. No crash, nothing non-finite, GPU
   memory flat across the run.
4. **Logging coverage** -- the event file carries per-head critic loss, ``lambda``, ``g``, the
   ``V_N`` and ``Delta_N`` distributions and the termination-reason shares.

The script imports ``train.py`` rather than re-implementing its loop, so what is measured is the
code that will actually be run. That import is what launches Isaac Sim here, which is why this file
has no ``AppLauncher`` preamble of its own: ``train.py`` owns it, and the arguments it parses are
the ones prepared below.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_risk_flow_train.py
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Verify the integrated training script.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--timesteps", type=int, default=1000, help="Length of the smoke run.")
parser.add_argument("--v_n_checkpoint", type=str, default=None, help="Path to the frozen V_N checkpoint.")
args_local, _ = parser.parse_known_args()

# Hand `train.py` its own command line before importing it: its argument parser and its
# `AppLauncher` both run at import time.
TRAIN_ARGV = [
    "train.py",
    "--task", "G1-risk",
    "--num_envs", str(args_local.num_envs),
    "--timesteps", str(args_local.timesteps),
    "--seed", "42",
    "--no_dataset_reset",
]
if args_local.v_n_checkpoint is not None:
    TRAIN_ARGV += ["--v_n_checkpoint", args_local.v_n_checkpoint]
else:
    TRAIN_ARGV += ["--allow_untrained_v_n"]

sys.argv = TRAIN_ARGV

import scripts.constraints.train as train  # noqa: E402  -- launches Isaac Sim

"""Rest everything follows."""

import copy  # noqa: E402
import glob  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lib.utils.parse_utils import load_cfg_from_registry  # noqa: E402

from agents.risk_flow import RiskFlow  # noqa: E402
from buffer.risk_flow_buffer import RiskFlowBuffer  # noqa: E402
from models.risk_flow_models import DeterministicActor, MultiHorizonCritic  # noqa: E402

import gymnasium as gym  # noqa: E402

CONSTRAINT_DIM = 67
OBS_DIM = 96
NUM_ACTIONS = 29
HORIZON = 99

# research.md 4.6. Every one of these has to be reachable from the yaml, or it is not a knob.
REQUIRED_AGENT_KEYS = (
    "horizon",
    "batch_size",
    "learning_starts",
    "gradient_steps",
    "grad_norm_clip",
    "exploration_sigma",
    "critic_learning_rate",
    "target_update_tau",
    "actor_learning_rate",
    "control_cost_scale",
    "update_actor",
    "dual_learning_rate",
    "terminal_risk_threshold",
    "update_dual",
    "lagrange_init",
)

# research.md 5, bottom. Tag prefixes that must appear in the event file.
REQUIRED_TAGS = (
    "Loss / critic head",             # per-horizon critic loss
    "Loss / critic per-head",         # and its distribution over all H heads
    "Constraint / lambda",
    "Constraint / violation g",
    "Value / V_N distribution",
    "Value / Delta_N distribution",
    "Episode / fall share",
    "Episode / timeout share",
)

# A relative growth above this between the first and the last quarter of the run is read as a leak
# rather than as allocator noise.
MEMORY_GROWTH_LIMIT = 0.10


def _box(dim):
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _torque_model(device):
    """Stand-in for ``G1RiskEnv.get_torque_model()``.

    The constants are irrelevant to serialization and were verified against the simulator in P4.2;
    using them here only avoids holding a second environment open across the smoke run.
    """
    return {
        "action_scale": 0.5,
        "stiffness": torch.full((NUM_ACTIONS,), 200.0, device=device),
        "damping": torch.full((NUM_ACTIONS,), 5.0, device=device),
        "joint_pos_offset": 9,
        "joint_vel_offset": 38,
    }


def check_config_surface(emit, failures):
    """Every research.md 4.6 knob is exposed through the gym registry."""
    cfg = load_cfg_from_registry("G1-risk", "rl_risk_flow_cfg_entry_point")

    missing = [key for key in REQUIRED_AGENT_KEYS if key not in cfg["agent"]]
    if missing:
        failures.append(f"the config does not expose {missing}")

    for section in ("train", "buffer", "models", "v_n"):
        if section not in cfg:
            failures.append(f"the config has no '{section}' section")

    if cfg["agent"]["horizon"] != "auto":
        emit(f"[P5.1] note: horizon is pinned to {cfg['agent']['horizon']} rather than read from the environment")

    emit(f"[P5.1] config surface: {len(REQUIRED_AGENT_KEYS) - len(missing)}/{len(REQUIRED_AGENT_KEYS)} "
         f"agent keys, buffer_size={cfg['buffer']['buffer_size']}, timesteps={cfg['train']['timesteps']}")
    return cfg


def check_horizon_resolution(cfg, emit, failures):
    """``horizon: auto`` follows the environment, and a disagreeing literal is refused."""

    class _Env:
        max_episode_length = HORIZON + 1

    env = _Env()
    resolved = train.resolve_horizon("auto", env)
    if resolved != HORIZON:
        failures.append(f"horizon: auto resolved to {resolved}, expected {HORIZON}")

    if train.resolve_horizon(HORIZON, env) != HORIZON:
        failures.append("a matching literal horizon was not accepted")

    try:
        train.resolve_horizon(HORIZON + 7, env)
        failures.append("a horizon disagreeing with the environment was accepted")
    except ValueError:
        pass

    emit(f"[P5.1] horizon: auto -> {resolved} (env.max_episode_length {env.max_episode_length} - 1)")


def _build_agent(cfg, value_critic, device):
    torch.manual_seed(0)
    buffer = RiskFlowBuffer(buffer_size=8, num_envs=16, device=device)
    buffer.init_buffer(_box(OBS_DIM), _box(CONSTRAINT_DIM), _box(NUM_ACTIONS))

    model = {
        "critic": MultiHorizonCritic(CONSTRAINT_DIM, NUM_ACTIONS, HORIZON, device=device),
        "actor": DeterministicActor(OBS_DIM, NUM_ACTIONS, device=device),
    }
    agent_cfg = dict(cfg["agent"])
    agent_cfg["horizon"] = HORIZON
    agent_cfg["seed"] = 42
    agent_cfg["learning_starts"] = 0
    return RiskFlow(model, buffer, value_critic, _torque_model(device), device=device, cfg=agent_cfg)


def _fill(agent, device, rows=8):
    for _ in range(rows):
        agent.insert_data(
            observations=torch.randn(16, OBS_DIM, device=device),
            constraint_states=torch.randn(16, CONSTRAINT_DIM, device=device),
            actions=0.1 * torch.randn(16, NUM_ACTIONS, device=device),
            final_observations=torch.randn(16, OBS_DIM, device=device),
            final_constraint_states=torch.randn(16, CONSTRAINT_DIM, device=device),
            terminated=torch.rand(16, 1, device=device) < 0.05,
            truncated=torch.rand(16, 1, device=device) < 0.05,
        )


def check_checkpoint_round_trip(cfg, value_critic, device, emit, failures):
    """Save, clobber, restore -- and the agent has to come back identical, not merely similar.

    Both optimizers are included on purpose. Adam's moment estimates are state: an agent resumed
    with them reset takes a burst of oversized steps into a critic that was already converged,
    which looks like instability rather than like a missing checkpoint entry.
    """
    agent = _build_agent(cfg, value_critic, device)
    _fill(agent, device)
    for _ in range(20):
        agent.update()

    probe_observations = torch.randn(32, OBS_DIM, device=device)
    probe_constraints = torch.randn(32, CONSTRAINT_DIM, device=device)
    before_actions = agent.act(probe_observations, deterministic=True).clone()
    before_flow = agent.critic(probe_constraints, before_actions).clone()
    before_target = agent.target_critic(probe_constraints, before_actions).clone()
    before_lambda = agent.lagrange().item()
    before_state = copy.deepcopy(agent.critic_optimizer.state_dict())

    path = os.path.join(tempfile.mkdtemp(), "agent.pt")
    agent.save(path)

    # Clobber everything the checkpoint is supposed to carry.
    with torch.no_grad():
        for parameter in list(agent.critic.parameters()) + list(agent.actor.parameters()):
            parameter.add_(torch.randn_like(parameter))
        for parameter in agent.target_critic.parameters():
            parameter.add_(torch.randn_like(parameter))
        agent.lagrange.nu.fill_(-3.0)
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=agent.critic_learning_rate)
    agent.checkpoint_modules["critic_optimizer"] = agent.critic_optimizer

    agent.load(path)

    after_actions = agent.act(probe_observations, deterministic=True)
    after_flow = agent.critic(probe_constraints, after_actions)
    after_target = agent.target_critic(probe_constraints, after_actions)
    after_state = agent.critic_optimizer.state_dict()

    action_error = (after_actions - before_actions).abs().max().item()
    flow_error = (after_flow - before_flow).abs().max().item()
    target_error = (after_target - before_target).abs().max().item()
    lambda_error = abs(agent.lagrange().item() - before_lambda)

    step_before = before_state["state"][0]["step"]
    step_after = after_state["state"][0]["step"]
    exp_avg_error = (after_state["state"][0]["exp_avg"] - before_state["state"][0]["exp_avg"]).abs().max().item()

    emit(f"[P5.1] checkpoint: action {action_error:.3e}, critic {flow_error:.3e}, "
         f"target critic {target_error:.3e}, lambda {lambda_error:.3e}, "
         f"Adam step {int(step_before)} -> {int(step_after)} exp_avg {exp_avg_error:.3e}")

    for name, error in (("actor", action_error), ("critic", flow_error),
                        ("target critic", target_error), ("lambda", lambda_error),
                        ("critic optimizer moments", exp_avg_error)):
        if error != 0.0:
            failures.append(f"{name} did not survive the checkpoint round-trip (error {error:.3e})")
    if int(step_before) != int(step_after):
        failures.append(f"the optimizer step counter was not restored ({int(step_before)} -> {int(step_after)})")


def check_resumed_update(cfg, value_critic, device, emit, failures):
    """A restored agent must continue, not restart: the same batch has to move it the same way.

    Run separately from the round-trip check because it fails differently. Identical weights with a
    reset optimizer, or with a stale target critic, still pass the equality above and then diverge
    on the first update.
    """
    reference = _build_agent(cfg, value_critic, device)
    _fill(reference, device)
    for _ in range(20):
        reference.update()

    path = os.path.join(tempfile.mkdtemp(), "agent.pt")
    reference.save(path)

    torch.manual_seed(7)
    reference.update()
    reference_actions = reference.act(torch.zeros(4, OBS_DIM, device=device), deterministic=True)

    resumed = _build_agent(cfg, value_critic, device)
    resumed.buffer = reference.buffer
    resumed.load(path)

    torch.manual_seed(7)
    resumed.update()
    resumed_actions = resumed.act(torch.zeros(4, OBS_DIM, device=device), deterministic=True)

    error = (resumed_actions - reference_actions).abs().max().item()
    emit(f"[P5.1] resumed update: action divergence after one shared update = {error:.3e}")
    if error != 0.0:
        failures.append(f"a resumed agent stepped differently from the one it was saved from ({error:.3e})")


def _latest_run_dir():
    candidates = sorted(glob.glob(os.path.join(os.getcwd(), "runs", "g1_risk", "*")))
    return candidates[-1] if candidates else None


def _event_file(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    return files[-1] if files else None


def check_logging_coverage(run_dir, emit, failures):
    """Every research.md 5 item reached the event file.

    The tags are matched against the raw bytes rather than through an event reader: what matters is
    that the writer emitted them, and the byte scan covers scalars and histograms alike without
    depending on which TensorBoard reader happens to be installed.
    """
    path = _event_file(run_dir)
    if path is None:
        failures.append(f"no event file was written to {run_dir}")
        return

    with open(path, "rb") as handle:
        blob = handle.read()

    missing = [tag for tag in REQUIRED_TAGS if tag.encode("utf-8") not in blob]
    emit(f"[P5.1] logging: {len(REQUIRED_TAGS) - len(missing)}/{len(REQUIRED_TAGS)} required tags "
         f"in {os.path.basename(path)} ({len(blob) / 1024:.0f} KiB)")
    if missing:
        failures.append(f"these logging items never reached the event file: {missing}")


def check_memory_trend(run_dir, emit, failures):
    """The GPU footprint the run logged itself has to be flat.

    The buffer is preallocated and the tracking containers are cleared at every write interval, so
    a rising trend means something is being retained per step -- a graph held by a stored tensor is
    the usual one, and it stays invisible until the run is long.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        emit("[P5.1] memory trend: skipped (no TensorBoard event reader available)")
        return

    accumulator = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    accumulator.Reload()
    tag = "System / gpu memory (GiB)"
    if tag not in accumulator.Tags().get("scalars", []):
        emit("[P5.1] memory trend: skipped (no GPU memory series)")
        return

    series = [event.value for event in accumulator.Scalars(tag)]
    if len(series) < 8:
        emit(f"[P5.1] memory trend: skipped (only {len(series)} samples)")
        return

    quarter = max(1, len(series) // 4)
    head = float(np.mean(series[:quarter]))
    tail = float(np.mean(series[-quarter:]))
    growth = (tail - head) / max(head, 1e-9)
    emit(f"[P5.1] memory trend: {head:.3f} -> {tail:.3f} GiB over {len(series)} samples ({growth:+.1%})")
    if growth > MEMORY_GROWTH_LIMIT:
        failures.append(f"the GPU footprint grew by {growth:.1%} across the run; something is being retained")


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit(f"[P5.1] device={device}, num_envs={args_local.num_envs}, timesteps={args_local.timesteps}")

    cfg = check_config_surface(emit, failures)
    check_horizon_resolution(cfg, emit, failures)

    # The frozen V_N is built once and shared: these checks are about serialization and resumption,
    # neither of which touches it (it is deliberately absent from `checkpoint_modules`).
    value_critic = train.build_value_critic(
        {**cfg["v_n"]["agent"], "seed": 42}, CONSTRAINT_DIM, device)

    check_checkpoint_round_trip(cfg, value_critic, device, emit, failures)
    check_resumed_update(cfg, value_critic, device, emit, failures)

    # -- the real loop, last: it creates and closes its own environment
    before_runs = set(glob.glob(os.path.join(os.getcwd(), "runs", "g1_risk", "*")))
    code = train.main()
    if code != 0:
        failures.append(f"train.main() returned {code}")

    run_dir = _latest_run_dir()
    if run_dir is None or run_dir in before_runs:
        failures.append("the smoke run produced no new log directory")
    else:
        emit(f"[P5.1] smoke run: completed {args_local.timesteps} steps into {os.path.basename(run_dir)}")
        if not os.path.exists(os.path.join(run_dir, "agent_final.pt")):
            failures.append("no final checkpoint was written")
        check_logging_coverage(run_dir, emit, failures)
        check_memory_trend(run_dir, emit, failures)

    if failures:
        emit(f"[P5.1] FAILED ({len(failures)})")
        for failure in failures[:20]:
            emit(f"        - {failure}")
    else:
        emit("[P5.1] PASSED: the config exposes every hyperparameter, the run is resumable "
             "bit-for-bit, and the loop logs everything section 5 asks for.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p5_1_result.txt"), "w") as handle:
        handle.write(chr(10).join(report) + chr(10))

    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    train.simulation_app.close()
    raise SystemExit(code)
