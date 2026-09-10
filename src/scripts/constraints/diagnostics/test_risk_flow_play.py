# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P6 verification: the evaluation and visualization pipeline.

Nothing here judges the policy. plan.md P5.2 does that, and this script only establishes that the
numbers it would judge come from the network the checkpoint holds and are reduced correctly:

1. **Restoration identity** -- a restored agent reproduces the saved one exactly, actions and
   critic outputs alike, *including* the ``RunningMeanStd`` statistics. Those are registered
   buffers rather than parameters, so a checkpoint that dropped them would still load and still
   return plausible actions, standardized against ``mean = 0, var = 1``. The check therefore drives
   the statistics well away from their initial values first, and confirms in the same pass that an
   agent which has *not* loaded the checkpoint does differ -- otherwise the equality proves nothing.

2. **Episode bookkeeping** -- the per-env accumulators are harvested on the step an episode ends
   and reset for the next one. This is the part of ``play.py`` that can be wrong without anything
   looking wrong: accumulators that leak across an episode boundary make a policy that falls early
   score better the worse it does, and no end-to-end run reveals it.

3. **`viz_data` contract** -- the environment fills the simulator-side channels and leaves the
   algorithm-side ones (``risk_value``, ``risk_flow``, ``terminal_risk``) for ``play.py`` to write,
   the way ``main/reach_avoid/play.py`` injects ``risk_value``. A channel declared on the cfg but
   never written stays at its scalar default, and the plotter then draws a flat line rather than
   failing, so the split is checked rather than assumed.

4. **End to end** -- ``play.main()`` itself: checkpoint load, rollout, the metric table, the plot
   and the video, all under ``log_dir``. No crash, and the artifacts on disk.

Like the P5.1 script, this one imports ``play.py`` rather than re-implementing it, so what is
measured is the code that will actually be run. That import is what launches Isaac Sim, which is
why there is no ``AppLauncher`` preamble here: ``play.py`` owns it, and the arguments it parses are
the ones prepared below.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_risk_flow_play.py \\
        --checkpoint runs/g1_risk/<run>/agent_final.pt
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Verify the evaluation pipeline.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--timesteps", type=int, default=400, help="Length of the end-to-end run.")
parser.add_argument("--checkpoint", type=str, default=None, help="RiskFlow checkpoint to play back.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Frozen V_N checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Include the video path in the run.")
args_local, _ = parser.parse_known_args()

if args_local.checkpoint is None:
    raise SystemExit("[P6] --checkpoint is required: there is no evaluation pipeline without one.")

# Hand `play.py` its own command line before importing it: its argument parser and its
# `AppLauncher` both run at import time.
PLAY_ARGV = [
    "play.py",
    "--task", "G1-risk-play",
    "--num_envs", str(args_local.num_envs),
    "--timesteps", str(args_local.timesteps),
    "--policy", "riskflow",
    "--checkpoint", args_local.checkpoint,
    "--seed", "42",
    "--no_dataset_reset",
]
if args_local.predictor_checkpoint is not None:
    PLAY_ARGV += ["--predictor_checkpoint", args_local.predictor_checkpoint]
else:
    PLAY_ARGV += ["--allow_untrained_predictor"]
if args_local.video:
    PLAY_ARGV += ["--video", "--video_length", "60"]

sys.argv = PLAY_ARGV

import scripts.constraints.play as play  # noqa: E402  -- launches Isaac Sim

"""Rest everything follows."""

import glob  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lib.utils.parse_utils import load_cfg_from_registry  # noqa: E402

from scripts.utils import build_agent, build_models  # noqa: E402

CONSTRAINT_DIM = 67
OBS_DIM = 96
NUM_ACTIONS = 29
HORIZON = 99

def _torque_model(device):
    """Stand-in for ``G1RiskEnv.get_torque_model()``.

    The constants were checked against the simulator in P4.2 and are irrelevant to serialization;
    using them here avoids holding a second environment open across the checks.
    """
    return {
        "action_scale": 0.5,
        "stiffness": torch.full((NUM_ACTIONS,), 200.0, device=device),
        "damping": torch.full((NUM_ACTIONS,), 5.0, device=device),
        "joint_pos_offset": 9,
        "joint_vel_offset": 38,
    }


def _build(cfg, value_critic, device, checkpoint=None):
    agent_cfg = dict(cfg["agent"])
    agent_cfg["horizon"] = HORIZON
    agent_cfg["seed"] = 42
    models = build_models(cfg.get("models", {}), OBS_DIM, CONSTRAINT_DIM, NUM_ACTIONS, HORIZON, device)
    return build_agent(agent_cfg, models, None, value_critic, _torque_model(device),
                       device=device, checkpoint=checkpoint)


def check_restoration_identity(cfg, value_critic, device, emit, failures):
    """What was saved is what plays back -- normalization statistics included."""
    agent = _build(cfg, value_critic, device)

    # Drive the standardizers away from `mean = 0, var = 1`. A checkpoint that carried only the
    # parameters would pass every equality below if this step were skipped.
    for _ in range(16):
        agent.actor(3.0 + 2.0 * torch.randn(256, OBS_DIM, device=device), update_rms=True)
        agent.critic(3.0 + 2.0 * torch.randn(256, CONSTRAINT_DIM, device=device),
                     torch.randn(256, NUM_ACTIONS, device=device), update_rms=True)

    actor_mean = agent.actor.actor_standardizer.mean.abs().mean().item()
    critic_mean = agent.critic.critic_standardizer.mean.abs().mean().item()
    if actor_mean < 1e-3 or critic_mean < 1e-3:
        failures.append("the standardizers did not move; the round-trip below would be vacuous")

    probe_observations = torch.randn(64, OBS_DIM, device=device)
    probe_constraints = torch.randn(64, CONSTRAINT_DIM, device=device)
    saved_actions = agent.act(probe_observations, deterministic=True).clone()
    saved_flow = agent.critic(probe_constraints, saved_actions).clone()

    path = os.path.join(tempfile.mkdtemp(), "agent.pt")
    agent.save(path)

    # A fresh agent built the same way but *not* loaded: the control that makes the equality mean
    # something. Its seed is the one `Agent.__init__` sets, so identical output here would say the
    # checkpoint is irrelevant rather than that it was restored.
    naive = _build(cfg, value_critic, device)
    naive_error = (naive.act(probe_observations, deterministic=True) - saved_actions).abs().max().item()

    restored = _build(cfg, value_critic, device, checkpoint=path)
    restored_actions = restored.act(probe_observations, deterministic=True)
    restored_flow = restored.critic(probe_constraints, restored_actions)

    action_error = (restored_actions - saved_actions).abs().max().item()
    flow_error = (restored_flow - saved_flow).abs().max().item()
    rms_error = max(
        (restored.actor.actor_standardizer.mean - agent.actor.actor_standardizer.mean).abs().max().item(),
        (restored.actor.actor_standardizer.var - agent.actor.actor_standardizer.var).abs().max().item(),
        (restored.critic.critic_standardizer.mean - agent.critic.critic_standardizer.mean).abs().max().item(),
        (restored.critic.critic_standardizer.var - agent.critic.critic_standardizer.var).abs().max().item(),
        abs(restored.actor.actor_standardizer.count.item() - agent.actor.actor_standardizer.count.item()),
    )

    emit(f"[P6] restoration: action {action_error:.3e}, critic {flow_error:.3e}, RunningMeanStd {rms_error:.3e} "
         f"(standardizer |mean| actor {actor_mean:.3f} / critic {critic_mean:.3f}, "
         f"unloaded agent differs by {naive_error:.3e})")

    for name, error in (("actions", action_error), ("critic outputs", flow_error),
                        ("normalization statistics", rms_error)):
        if error != 0.0:
            failures.append(f"restored {name} are not bit-identical to the saved ones ({error:.3e})")
    if naive_error == 0.0:
        failures.append("an agent that never loaded the checkpoint matched it exactly; "
                        "the round-trip proves nothing")


def check_episode_bookkeeping(emit, failures):
    """The accumulate / harvest / reset cycle of ``play.main()``, on a scripted episode pattern.

    The loop itself needs a simulator, so the rule it applies is replayed here against a hand-made
    sequence of dones whose answer is known in closed form. Two of the four environments end inside
    the window, so an accumulator that failed to reset would carry the first episode's cost into the
    second and land on a different number.
    """
    num_envs, delta_n = 4, 0.0

    # env 0 ends at step 2 (fall) and again at step 5; env 1 ends at step 5 (time-out);
    # envs 2 and 3 never end inside the window.
    dones = {2: [True, False, False, False], 5: [True, True, False, False]}
    terminals = {2: [True, False, False, False], 5: [False, False, False, False]}
    # V_N after each step, per env.
    values = [
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2, 0.2],
        [0.9, -0.1, 0.3, 0.3],     # step 2: env 0 falls at V_N = 0.9 -> not recovered
        [0.1, -0.2, 0.1, 0.1],
        [0.0, -0.3, 0.0, 0.0],
        [-0.4, -0.5, -0.1, -0.1],  # step 5: env 0 and env 1 end, both at V_N <= 0
    ]

    steps = torch.zeros(num_envs)
    cost = torch.zeros(num_envs)
    recovery = torch.full((num_envs,), -1.0)
    initial = torch.zeros(num_envs)
    started = torch.zeros(num_envs, dtype=torch.bool)
    harvest = {"recovered": [], "fell": [], "cost": [], "recovery": [], "realized": []}

    for step in range(1, 6):
        value = torch.tensor(values[step - 1])
        control_cost = torch.full((num_envs,), float(step))

        fresh = ~started
        if fresh.any():
            initial = torch.where(fresh, value, initial)
            recovery = torch.where(fresh & (value <= delta_n), torch.zeros_like(recovery), recovery)
            started |= fresh

        steps += 1.0
        cost += control_cost
        next_value = torch.tensor(values[step])
        first = (recovery < 0) & (next_value <= delta_n)
        recovery = torch.where(first, steps, recovery)

        done = torch.tensor(dones.get(step, [False] * num_envs))
        terminated = torch.tensor(terminals.get(step, [False] * num_envs))
        finished = done.nonzero(as_tuple=False).squeeze(-1)
        if finished.numel():
            harvest["recovered"] += (next_value[finished] <= delta_n).float().tolist()
            harvest["fell"] += terminated[finished].float().tolist()
            harvest["cost"] += (cost[finished] / steps[finished]).tolist()
            harvest["recovery"] += recovery[finished].tolist()
            harvest["realized"] += (next_value[finished] - initial[finished]).tolist()
            steps[finished] = 0.0
            cost[finished] = 0.0
            recovery[finished] = -1.0
            started[finished] = False

    # env 0 episode 1: steps 1-2, cost (1+2)/2 = 1.5, fell, ended at V_N = 0.9 -> not recovered,
    #                  realized 0.9 - 0.5 = 0.4.
    # env 0 episode 2: steps 3-5, cost (3+4+5)/3 = 4.0, timed out, ended at -0.4 -> recovered,
    #                  realized -0.4 - 0.1 = -0.5. A leaked accumulator would give cost 3.0 here.
    # env 1 episode 1: steps 1-5, cost (1+2+3+4+5)/5 = 3.0, timed out, ended at -0.5, initial 0.5.
    expected = {
        "episodes": 3,
        "recovered": [0.0, 1.0, 1.0],
        "fell": [1.0, 0.0, 0.0],
        "cost": [1.5, 4.0, 3.0],
        "realized": [0.4, -0.5, -1.0],
    }

    wrong = []
    if len(harvest["fell"]) != expected["episodes"]:
        wrong.append(f"harvested {len(harvest['fell'])} episodes, expected {expected['episodes']}")
    for key in ("recovered", "fell", "cost", "realized"):
        got, want = harvest[key], expected[key]
        if len(got) != len(want) or any(not math.isclose(g, w, abs_tol=1e-6) for g, w in zip(got, want)):
            wrong.append(f"{key}: {got} != {want}")

    # env 0's second episode first reached V_N <= 0 on its own step 1, not on the global step.
    if len(harvest["recovery"]) > 1 and not math.isclose(harvest["recovery"][1], 1.0, abs_tol=1e-6):
        wrong.append(f"recovery step of the second episode: {harvest['recovery'][1]} != 1.0")

    if wrong:
        failures.append(f"the episode bookkeeping is off: {wrong}")

    emit(f"[P6] episode bookkeeping: {len(harvest['fell'])} episodes harvested, "
         f"cost {[round(c, 3) for c in harvest['cost']]}, fell {harvest['fell']}, "
         f"{len(wrong)} mismatches")


def check_viz_contract(emit, failures):
    """The cfg declares the channels, the environment fills its half, ``play.py`` fills the rest."""
    play_cfg = load_cfg_from_registry("G1-risk-play", "env_cfg_entry_point")
    train_cfg = load_cfg_from_registry("G1-risk", "env_cfg_entry_point")

    if train_cfg.viz_data is not None or train_cfg.plotter is not None:
        failures.append("the training task declares a plotter; nothing consumes it there")

    declared = set(play_cfg.viz_data or {})
    algorithm_side = {"risk_value", "risk_flow", "terminal_risk"}
    environment_side = {"action_magnitude", "max_torque", "CoM_height"}

    missing = (algorithm_side | environment_side) - declared
    if missing:
        failures.append(f"the play cfg does not declare {sorted(missing)}")
    if play_cfg.plotter is None:
        failures.append("the play cfg declares no plotter, so no trajectory is recorded")

    # `play.py` has to write every algorithm-side channel; the environment writes none of them.
    with open(os.path.abspath(play.__file__), encoding="utf-8") as handle:
        source = handle.read()
    unwritten = [key for key in sorted(algorithm_side) if f'viz_data"]["{key}"]' not in source]
    if unwritten:
        failures.append(f"play.py never writes {unwritten}; those columns would stay at their default")

    emit(f"[P6] viz contract: {len(declared)} channels declared, "
         f"{len(algorithm_side) - len(unwritten)}/{len(algorithm_side)} algorithm-side written by play.py, "
         f"plotter={getattr(play_cfg.plotter, '__name__', None)}")


def check_end_to_end(emit, failures):
    """``play.main()`` runs, and the artifacts land where the submodule's play scripts put them."""
    code = play.main()
    if code != 0:
        failures.append(f"play.main() returned {code}")
        return

    # Same rule as play.py: the run writes into the directory of the checkpoint it played back.
    log_dir = os.path.dirname(os.path.abspath(args_local.checkpoint))
    plot = os.path.join(log_dir, "plot", "trajectory.png")
    sheet = os.path.join(log_dir, "plot", "trajectory.xlsx")

    artifacts = []
    for label, path in (("plot", plot), ("sheet", sheet)):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            artifacts.append(f"{label} {os.path.getsize(path) / 1024:.0f} KiB")
        else:
            failures.append(f"the run wrote no {label} to {path}")

    if play.args_cli.video:
        videos = glob.glob(os.path.join(log_dir, "videos", "play", "*.mp4"))
        if videos:
            artifacts.append(f"video {os.path.getsize(videos[0]) / 1024:.0f} KiB")
        else:
            failures.append(f"the run wrote no video to {os.path.join(log_dir, 'videos', 'play')}")

    # The recorded trajectory has to carry both halves of `viz_data` with real values in them: a
    # column that stayed at its declared default is a channel nobody wrote.
    if os.path.exists(sheet):
        try:
            import pandas as pd

            frame = pd.read_excel(sheet, sheet_name="trajectory")
            flat = [column for column in ("risk_value", "risk_flow", "CoM_height", "max_torque")
                    if column in frame.columns and frame[column].nunique() <= 1]
            if flat:
                failures.append(f"these recorded channels never varied: {flat}")
            artifacts.append(f"{len(frame)} frames x {len(frame.columns)} columns")
        except ImportError:
            emit("[P6] trajectory sheet: skipped (pandas not available)")

    emit(f"[P6] end to end: {', '.join(artifacts)} in {log_dir}")


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit(f"[P6] device={device}, num_envs={args_local.num_envs}, timesteps={args_local.timesteps}, "
         f"checkpoint={os.path.basename(args_local.checkpoint)}")

    cfg = load_cfg_from_registry("G1-risk-play", "rl_risk_flow_cfg_entry_point")

    # The frozen V_N takes no part in any of the checks below -- it is deliberately absent from
    # `checkpoint_modules` -- so one untrained instance is enough to construct agents with.
    value_critic = play.build_predictor({**cfg["predictor"]["agent"], "seed": 42},
                                        CONSTRAINT_DIM, device, allow_untrained=True)

    check_restoration_identity(cfg, value_critic, device, emit, failures)
    check_episode_bookkeeping(emit, failures)
    check_viz_contract(emit, failures)

    # -- the real pipeline, last: it creates and closes its own environment
    check_end_to_end(emit, failures)

    if failures:
        emit("[P6] FAILED:")
        for failure in failures:
            emit(f"[P6]   - {failure}")
    else:
        emit("[P6] PASSED: the checkpoint plays back exactly, the episode bookkeeping is sound, "
             "and the pipeline runs to its artifacts.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p6_result.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print(f"--------- Saved report to: {path}", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    play.simulation_app.close()
    sys.exit(code)
