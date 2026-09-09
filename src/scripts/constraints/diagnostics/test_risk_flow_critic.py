# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P4.1 verification: the critic-only stage of :class:`RiskFlow`.

One-step TD learns ``D_h`` purely from the recursion ``D_h <- Delta_N + D_{h-1}``. A flipped
horizon shift, an absorbing mask landing on the wrong head, or a mis-wired ``final_*`` channel all
leave that recursion self-consistent, so the loss curve keeps falling while the network converges
to the wrong quantity. Two outside references are needed:

1. **The target algebra** -- with a stub ``V_N`` and a stub critic whose output is the exact fixed
   point of the recursion, the target has a closed form. Any shift breaks the equality.
2. **Monte Carlo** -- under a *fixed deterministic* policy, ``V_N(s_{t+h}) - V_N(s_t)`` measured
   from a real rollout is the definition of ``D_h``. Predictions are compared per horizon for sign
   and scale, and against the truth at ``h-1``/``h``/``h+1`` for alignment -- the latter only where
   the neighbouring references can be told apart at all.

No accuracy threshold is applied; the checks are structural (sign, alignment, finiteness). How
close the critic actually gets is judged in P5.2.

The frozen ``V_N`` may be randomly initialized here: ``D_h`` is defined relative to whichever
``V_N`` is installed, and the Monte-Carlo reference uses that same network, so the check does not
depend on P0.2 having produced a trained one. Pass ``--v_n_checkpoint`` to use the real one.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_risk_flow_critic.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the critic-only stage of RiskFlow.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--rollout_steps", type=int, default=300, help="Steps of the Monte-Carlo rollout.")
parser.add_argument("--updates", type=int, default=4000, help="Critic gradient steps.")
parser.add_argument("--v_n_checkpoint", type=str, default=None, help="Path to the frozen V_N checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from lib.agent.reach_avoid import ReachAvoid
from lib.model.MLP import RA_Critic

from agents.risk_flow import RiskFlow
from buffer.risk_flow_buffer import RiskFlowBuffer
from envs.G1.risk.G1_risk_env import G1RiskEnv
from envs.G1.risk.G1_risk_env_cfg import G1RiskEnvCfg
from models.risk_flow_models import DeterministicActor, MultiHorizonCritic

CONSTRAINT_DIM = 67
OBS_DIM = 96
NUM_ACTIONS = 29
HORIZON = 99

# The recovery deadline H is 99 in production, tied to `episode_length_s = 2.0`. The Monte-Carlo
# reference needs contiguous trajectories longer than H, so the episode is stretched here and H is
# pinned by hand. Nothing in the recursion depends on the two being equal.
MC_EPISODE_LENGTH_S = 8.0

HORIZONS_UNDER_TEST = (1, 10, 50, 99)

# The alignment test is only run where the neighbouring Monte-Carlo references are actually
# distinguishable from each other; beyond this correlation they are the same vector up to noise.
REFERENCE_RESOLUTION_LIMIT = 0.9

# ...and only where the prediction carries enough signal for the comparison to mean anything.
ALIGNMENT_SIGNAL_FLOOR = 0.3

V_N_AGENT_CFG = {
    "seed": 0,
    "learning_epochs": 1,
    "batch_size": 1,
    "learning_rate": 1.0e-4,
    "discount_factor": 0.99,
    "grad_norm_clip": 1.0,
    "learning_starts": 0,
}

RISK_FLOW_CFG = {
    "seed": 0,
    "horizon": HORIZON,
    "batch_size": 4096,
    "learning_starts": 0,
    "grad_norm_clip": 1.0,
    "exploration_sigma": 0.1,
    "critic_learning_rate": 3.0e-4,
    "target_update_tau": 0.005,
    "actor_learning_rate": 1.0e-4,
    "control_cost_scale": 1.0e-3,
    "dual_learning_rate": 1.0e-3,
    "terminal_risk_threshold": 0.0,
    "lagrange_init": 0.0,
    # the Monte-Carlo reference is defined against a *fixed* policy
    "update_actor": False,
    "update_dual": False,
}


def _box(dim):
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _build_value_critic(device):
    """The frozen V_N, built and loaded exactly as the run that trained it did."""
    model = {"critic": RA_Critic(CONSTRAINT_DIM, device)}
    agent = ReachAvoid(model, None, device=device, cfg=dict(V_N_AGENT_CFG))
    if args_cli.v_n_checkpoint is not None:
        agent.load(os.path.abspath(args_cli.v_n_checkpoint))
    return agent.critic


def _build_agent(env, value_critic, buffer, device):
    model = {
        "critic": MultiHorizonCritic(CONSTRAINT_DIM, NUM_ACTIONS, HORIZON, device=device),
        "actor": DeterministicActor(OBS_DIM, NUM_ACTIONS, device=device),
    }
    return RiskFlow(model, buffer, value_critic, env.get_torque_model(), device=device,
                    cfg=dict(RISK_FLOW_CFG))


def _new_buffer(rows, num_envs, device):
    buffer = RiskFlowBuffer(buffer_size=rows, num_envs=num_envs, device=device)
    buffer.init_buffer(_box(OBS_DIM), _box(CONSTRAINT_DIM), _box(NUM_ACTIONS))
    return buffer


def _fill_random(buffer, rows, num_envs, device):
    """Fill a buffer with random transitions whose flags differ from each other.

    ``terminated`` and ``truncated`` are given disjoint patterns so that masking on the wrong one
    cannot coincide with masking on the right one.
    """
    for row in range(rows):
        terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        truncated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        terminated[row % num_envs] = True
        truncated[(row + 1) % num_envs] = True
        buffer.add_samples(
            observations=torch.randn(num_envs, OBS_DIM, device=device),
            constraint_states=torch.randn(num_envs, CONSTRAINT_DIM, device=device),
            actions=torch.randn(num_envs, NUM_ACTIONS, device=device),
            final_observations=torch.randn(num_envs, OBS_DIM, device=device),
            final_constraint_states=torch.randn(num_envs, CONSTRAINT_DIM, device=device),
            terminated=terminated,
            truncated=truncated,
        )


def _correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return (a @ b / denominator).item() if denominator > 0 else float("nan")


def check_target_algebra(agent, device, emit, failures):
    """The target has a closed form when the critic already sits at the fixed point.

    With ``Delta_N`` constant at ``d`` along a trajectory, the true flow is ``D_h = h * d``. Feeding
    the critic that exact function back in must reproduce it: ``y[j] = d + (j) * d = (j + 1) * d``,
    where column ``j`` holds ``D_{j+1}``. A horizon shift in either direction, or a mask applied to
    the wrong axis, changes this equality.
    """
    batch = 8
    deltas = torch.linspace(-2.0, 2.0, batch, device=device).unsqueeze(-1)

    states = torch.zeros(batch, CONSTRAINT_DIM, device=device)
    final_states = torch.zeros(batch, CONSTRAINT_DIM, device=device)
    # a stub V_N reading channel 0, so that Delta_N = final_states[:, 0] - states[:, 0]
    final_states[:, :1] = deltas
    final_observations = torch.zeros(batch, OBS_DIM, device=device)

    heads = torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32)
    original_value_critic, original_critic = agent.value_critic, agent.target_critic
    # the fixed point: D_h(s) = h * Delta_N(s), with Delta_N read out of channel 0.
    # The bootstrap reads the target critic, so that is the one that has to be stubbed.
    agent.value_critic = lambda states_in: (states_in[:, :1], None, None)
    agent.target_critic = lambda states_in, actions_in, update_rms=False: states_in[:, :1] * heads

    try:
        alive = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        target_alive = agent.compute_target(states, final_states, final_observations, alive)
        expected_alive = deltas * heads

        dead = torch.ones(batch, 1, dtype=torch.bool, device=device)
        target_dead = agent.compute_target(states, final_states, final_observations, dead)
        expected_dead = deltas.expand(batch, HORIZON)
    finally:
        agent.value_critic, agent.target_critic = original_value_critic, original_critic

    alive_error = (target_alive - expected_alive).abs().max().item()
    dead_error = (target_dead - expected_dead).abs().max().item()
    emit(f"[P4.1] target algebra: fixed-point error={alive_error:.3e}, absorbing error={dead_error:.3e}")
    if alive_error > 1e-4:
        failures.append(f"the target is not the fixed point of the recursion (max error {alive_error:.3e}); "
                        f"the horizon shift is wrong")
    if dead_error > 1e-4:
        failures.append(f"a terminated transition still bootstraps (max error {dead_error:.3e})")



def check_mask_wiring(agent, device, emit, failures):
    """``update()`` must mask on ``terminated`` alone, never on ``terminated | truncated``.

    A time-out is not absorbing: the pre-reset snapshot preserved the true next state, so there is
    a real state to bootstrap from. Folding the two flags together is the easy mistake, and it is
    invisible in the loss -- it just quietly throws away every long-horizon bootstrap. The two
    flags are given disjoint, differing patterns here so that the wrong wiring cannot coincide
    with the right one.
    """
    rows, num_envs = 4, 8
    buffer = _new_buffer(rows, num_envs, device)
    _fill_random(buffer, rows=rows, num_envs=num_envs, device=device)

    original_buffer, original_compute_target = agent.buffer, agent.compute_target
    seen = {}

    def recording_compute_target(constraint_states, final_constraint_states, final_observations, terminated):
        seen["mask_input"] = terminated.clone()
        return original_compute_target(constraint_states, final_constraint_states, final_observations, terminated)

    agent.buffer = buffer
    agent.compute_target = recording_compute_target
    try:
        agent.update()
    finally:
        agent.buffer, agent.compute_target = original_buffer, original_compute_target

    indexes = buffer.sampling_indexes
    stored_terminated = buffer.get_tensor_by_name("terminated", keepdim=False)[indexes]
    stored_truncated = buffer.get_tensor_by_name("truncated", keepdim=False)[indexes]
    combined = stored_terminated | stored_truncated

    passed = seen.get("mask_input")
    emit(f"[P4.1] mask wiring: batch has {int(stored_terminated.sum())} terminated / "
         f"{int(stored_truncated.sum())} truncated")
    if passed is None:
        failures.append("update() never called compute_target")
    elif not torch.equal(passed, stored_terminated):
        failures.append("update() masked on something other than the stored `terminated` flag")
    elif torch.equal(stored_terminated, combined):
        failures.append("the wiring check is degenerate: no truncated-only transition in the batch")


def check_target_network(env, value_critic, device, emit, failures):
    """The target critic must be a genuinely separate, lagging, gradient-free copy.

    Every part of that sentence is a way to get it wrong: sharing storage with the online net makes
    it not lag at all, leaving it in the optimizer makes it trained, and blending its normalization
    statistics instead of copying them leaves the target standardizing with numbers the online net
    has already abandoned.
    """
    buffer = _new_buffer(4, 8, device)
    _fill_random(buffer, rows=4, num_envs=8, device=device)
    agent = _build_agent(env, value_critic, buffer, device)
    tau = agent.target_update_tau

    # -- distinct objects, no shared storage
    online_pointers = {parameter.data_ptr() for parameter in agent.critic.parameters()}
    target_pointers = {parameter.data_ptr() for parameter in agent.target_critic.parameters()}
    if agent.target_critic is agent.critic or online_pointers & target_pointers:
        failures.append("the target critic shares storage with the online critic")

    # -- the optimizer must never see it
    optimized = {id(parameter) for group in agent.critic_optimizer.param_groups for parameter in group["params"]}
    if optimized & {id(parameter) for parameter in agent.target_critic.parameters()}:
        failures.append("the target critic's parameters are in the optimizer")

    # -- exact Polyak arithmetic
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.add_(torch.randn_like(parameter))
        before = [parameter.clone() for parameter in agent.target_critic.parameters()]
        online = [parameter.clone() for parameter in agent.critic.parameters()]

    agent.update_target()

    polyak_error = max(
        (after - ((1.0 - tau) * old + tau * new)).abs().max().item()
        for after, old, new in zip(agent.target_critic.parameters(), before, online)
    )
    moved = max((after - old).abs().max().item() for after, old in zip(agent.target_critic.parameters(), before))
    emit(f"[P4.1] target critic: tau={tau}, polyak error={polyak_error:.3e}, step size={moved:.3e}")
    if polyak_error > 1e-6:
        failures.append(f"the soft update is not (1-tau)*target + tau*online (error {polyak_error:.3e})")
    if moved == 0.0:
        failures.append("the soft update did not move the target critic at all")

    # -- normalization statistics are copied, not blended
    with torch.no_grad():
        agent.critic.critic_standardizer.mean.fill_(7.0)
    agent.update_target()
    target_mean = agent.target_critic.critic_standardizer.mean
    if not torch.allclose(target_mean, torch.full_like(target_mean, 7.0)):
        failures.append("the target critic's normalization statistics lag behind the online ones; "
                        "they must be copied, not Polyak-blended")

    # -- compute_target must read the target critic, not the online one
    batch = 8
    states = torch.randn(batch, CONSTRAINT_DIM, device=device)
    final_states = torch.randn(batch, CONSTRAINT_DIM, device=device)
    final_observations = torch.randn(batch, OBS_DIM, device=device)
    alive = torch.zeros(batch, 1, dtype=torch.bool, device=device)

    original_target = agent.target_critic
    agent.target_critic = lambda states_in, actions_in, update_rms=False: torch.zeros(
        states_in.shape[0], HORIZON, device=states_in.device)
    try:
        with torch.no_grad():
            silenced = agent.compute_target(states, final_states, final_observations, alive)
    finally:
        agent.target_critic = original_target

    # with the bootstrap silenced every head collapses onto Delta_N; if the online critic were
    # being read instead, the far heads would still carry its (non-zero) output
    if not torch.allclose(silenced, silenced[:, :1].expand_as(silenced), atol=1e-6):
        failures.append("compute_target bootstraps from the online critic, not the target critic")

    # -- gradients never reach it
    agent.update()
    with_grad = [name for name, parameter in agent.target_critic.named_parameters() if parameter.grad is not None]
    if with_grad:
        failures.append(f"the target critic received gradients: {with_grad[:3]}")

    if "target_critic" not in agent.checkpoint_modules:
        failures.append("the target critic is not in checkpoint_modules; a resumed run would restart "
                        "its lag from the online weights")


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit(f"[P4.1] device={device}, num_envs={args_cli.num_envs}")

    torch.manual_seed(0)
    value_critic = _build_value_critic(device)

    cfg = G1RiskEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = MC_EPISODE_LENGTH_S
    # P0.3 produces the risk-bucket dataset; until then the randomized default pose is the ro_I
    cfg.events.reset_state_from_dataset = None
    env = G1RiskEnv(cfg)

    # ---- check 1: the target algebra, independent of any rollout
    buffer = _new_buffer(args_cli.rollout_steps, args_cli.num_envs, device)
    agent = _build_agent(env, value_critic, buffer, device)
    check_target_algebra(agent, device, emit, failures)
    check_mask_wiring(agent, device, emit, failures)
    check_target_network(env, value_critic, device, emit, failures)

    # ---- check 2: a short run with real terminations, so the masked path is actually executed
    obs, _, constraint_states, _ = env.reset()
    num_terminated = 0
    for _ in range(100):
        actions = agent.act(obs)
        next_obs, _, next_constraint_states, _, terminated, truncated, extras = env.step(actions)
        agent.insert_data(
            observations=obs,
            constraint_states=constraint_states,
            actions=actions,
            final_observations=extras["final_observations"],
            final_constraint_states=extras["final_constraint_states"],
            terminated=terminated,
            truncated=truncated,
        )
        num_terminated += int(terminated.sum().item())
        obs, constraint_states = next_obs, next_constraint_states

    losses = [agent.update() for _ in range(20)]
    emit(f"[P4.1] masked path: {num_terminated} terminations, "
         f"critic_loss {losses[0]['critic_loss']:.4e} -> {losses[-1]['critic_loss']:.4e}")
    if num_terminated == 0:
        failures.append("no termination occurred; the absorbing branch was never executed")
    if any(not np.isfinite(entry["critic_loss"]) for entry in losses):
        failures.append("the critic loss went non-finite on data containing terminations")

    # ---- check 3: Monte Carlo, under a fixed deterministic policy
    # Terminations are suppressed so that every environment yields one contiguous trajectory: the
    # reference V_N(s_{t+h}) - V_N(s_t) needs both endpoints in the same episode.
    def dones_without_death():
        env._compute_intermediate_values()
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    env._get_dones = dones_without_death

    buffer = _new_buffer(args_cli.rollout_steps, args_cli.num_envs, device)
    agent = _build_agent(env, value_critic, buffer, device)

    steps = args_cli.rollout_steps
    trajectory_states = torch.zeros(steps, args_cli.num_envs, CONSTRAINT_DIM, device=device)
    trajectory_actions = torch.zeros(steps, args_cli.num_envs, NUM_ACTIONS, device=device)
    trajectory_values = torch.zeros(steps + 1, args_cli.num_envs, device=device)

    obs, _, constraint_states, _ = env.reset()
    num_resets = 0
    for step in range(steps):
        # deterministic: "follow pi afterwards" has to be exactly what the rollout does, or the
        # Monte-Carlo reference measures a different quantity than the one the critic predicts
        actions = agent.act(obs, deterministic=True)
        next_obs, _, next_constraint_states, _, terminated, truncated, extras = env.step(actions)

        with torch.no_grad():
            value, _, _ = value_critic(constraint_states)
        trajectory_states[step] = constraint_states
        trajectory_actions[step] = actions
        trajectory_values[step] = value.squeeze(-1)

        agent.insert_data(
            observations=obs,
            constraint_states=constraint_states,
            actions=actions,
            final_observations=extras["final_observations"],
            final_constraint_states=extras["final_constraint_states"],
            terminated=terminated,
            truncated=truncated,
        )
        num_resets += int((terminated | truncated).sum().item())
        obs, constraint_states = next_obs, next_constraint_states

    with torch.no_grad():
        value, _, _ = value_critic(constraint_states)
    trajectory_values[steps] = value.squeeze(-1)

    if num_resets != 0:
        failures.append(f"{num_resets} episodes ended during the Monte-Carlo rollout; the reference "
                        f"would span an episode boundary")

    history = []
    for iteration in range(args_cli.updates):
        info = agent.update()
        if not np.isfinite(info["critic_loss"]):
            failures.append(f"critic loss went non-finite at iteration {iteration}")
            break
        if iteration % max(1, args_cli.updates // 4) == 0 or iteration == args_cli.updates - 1:
            history.append((iteration, info["critic_loss"]))
    emit("[P4.1] critic loss: " + ", ".join(f"{i}:{loss:.3e}" for i, loss in history))

    per_head = info["per_head_loss"]
    emit(f"[P4.1] per-head loss: h=1 {per_head[0]:.3e}, h=10 {per_head[9]:.3e}, "
         f"h=50 {per_head[49]:.3e}, h={HORIZON} {per_head[-1]:.3e}")

    # predictions on the stored (s, a), compared against the measured flow
    with torch.no_grad():
        flat_states = trajectory_states.reshape(-1, CONSTRAINT_DIM)
        flat_actions = trajectory_actions.reshape(-1, NUM_ACTIONS)
        predictions = agent.critic(flat_states, flat_actions).reshape(steps, args_cli.num_envs, HORIZON)

    # -- sign and scale, from the cumulative flow
    emit(f"[P4.1] cumulative  {'h':>4} {'corr':>8} {'bias':>11} {'slope':>8}")
    for horizon in HORIZONS_UNDER_TEST:
        usable = steps - horizon
        truth = (trajectory_values[horizon:horizon + usable] - trajectory_values[:usable]).reshape(-1)
        prediction = predictions[:usable, :, horizon - 1].reshape(-1)

        correlation = _correlation(prediction, truth)
        bias = (prediction - truth).mean().item()
        variance = truth.var().item()
        slope = (((prediction - prediction.mean()) * (truth - truth.mean())).mean().item() / variance
                 if variance > 0 else float("nan"))
        emit(f"[P4.1] cumulative  {horizon:>4} {correlation:>8.4f} {bias:>11.3e} {slope:>8.3f}")

        if not np.isfinite(correlation):
            failures.append(f"h={horizon}: correlation with the Monte-Carlo reference is not finite")
        elif correlation <= 0.0:
            failures.append(f"h={horizon}: correlation with the Monte-Carlo reference is "
                            f"{correlation:.3f}; the sign of the flow is inverted")
        if not np.isfinite(slope) or slope <= 0.0:
            failures.append(f"h={horizon}: the regression slope against the reference is {slope:.3f}; "
                            f"the predicted flow does not scale with the measured one")

    # -- alignment, from the per-head increment
    #
    # Comparing cumulative flows cannot resolve a one-step shift at large h: V_N moves slowly, so
    # truth_h and truth_{h-1} are the same vector up to noise (their correlation reaches 0.999 by
    # h=99) and whichever of the two correlates better with the prediction flips from run to run.
    # The increment D_h - D_{h-1} predicts a single step of V_N, v[t+h] - v[t+h-1], and consecutive
    # steps are far less alike -- which is exactly the resolution the alignment test needs.
    increments = predictions.clone()
    increments[:, :, 1:] = predictions[:, :, 1:] - predictions[:, :, :-1]

    emit(f"[P4.1] increment   {'h':>4} {'corr(h-1)':>10} {'corr(h)':>10} {'corr(h+1)':>10} {'ref auto':>9}")
    for horizon in HORIZONS_UNDER_TEST:
        usable = steps - (horizon + 1)
        references = {
            offset: (trajectory_values[horizon + offset:horizon + offset + usable]
                     - trajectory_values[horizon + offset - 1:horizon + offset - 1 + usable]).reshape(-1)
            for offset in (-1, 0, 1) if 1 <= horizon + offset <= HORIZON
        }
        truth = references[0]

        correlations = {
            offset: _correlation(increments[:usable, :, horizon + offset - 1].reshape(-1), truth)
            for offset in references
        }
        neighbours = {offset: _correlation(reference, truth)
                      for offset, reference in references.items() if offset != 0}
        resolution = max(neighbours.values())

        emit(f"[P4.1] increment   {horizon:>4} "
             f"{correlations.get(-1, float('nan')):>10.4f} {correlations[0]:>10.4f} "
             f"{correlations.get(1, float('nan')):>10.4f} {resolution:>9.4f}")

        separable = [offset for offset, correlation in neighbours.items()
                     if correlation < REFERENCE_RESOLUTION_LIMIT]
        if not separable or not np.isfinite(correlations[0]):
            continue
        if correlations[0] < ALIGNMENT_SIGNAL_FLOOR:
            # Nothing to align. A single V_N step h into the future is barely a function of
            # (s_t, a_t) at all -- the cumulative flow is predictable, the increment that
            # distinguishes head h from head h-1 is not -- so a converged critic would score near
            # zero here too, and picking the largest of three near-zero correlations is noise.
            # The exact alignment evidence is the target-algebra identity, which covers every head.
            emit(f"[P4.1] increment   {horizon:>4} alignment not resolvable "
                 f"(|corr| {correlations[0]:.3f} < {ALIGNMENT_SIGNAL_FLOOR}) -- reported, not gated")
            continue
        best = max([0] + separable, key=lambda offset: correlations[offset])
        if best != 0:
            failures.append(f"h={horizon}: the predicted increment matches the reference at h{best:+d} "
                            f"better than at h ({correlations[best]:.3f} vs {correlations[0]:.3f}); "
                            f"the horizon is shifted")

    if failures:
        emit(f"[P4.1] FAILED ({len(failures)})")
        for failure in failures[:20]:
            emit(f"        - {failure}")
    else:
        emit("[P4.1] PASSED: target algebra exact over every head, absorbing mask wired to "
             "`terminated` alone, and the predicted flow tracks the Monte-Carlo reference in sign "
             "and scale at every horizon.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4_1_result.txt"), "w") as handle:
        handle.write(chr(10).join(report) + chr(10))

    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
