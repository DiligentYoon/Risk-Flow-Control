# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P4.3 verification: the primal-dual stage of :class:`RiskFlow`.

How far ``lambda`` climbs and whether the constraint improves anything are P5.2 questions. What is
checked here is the sign and the wiring, because both fail silently:

1. **Direction** -- a batch that violates the constraint must push ``lambda`` up and one with slack
   must push it down. A flipped ``loss_lam`` still trains smoothly, just with the constraint
   pressing the wrong way for a long time before anything looks wrong.
2. **Non-negativity** -- ``softplus(nu) >= 0`` for every ``nu``, with a finite gradient, so that no
   projection step is needed.
3. **Serialization** -- ``nu`` travels in the checkpoint; a resumed run must not restart from the
   initial multiplier.
4. **Effective weight** -- the coefficient the actor loss puts on ``D_H`` must be ``1/H + lambda``,
   with ``lambda`` detached on the policy side and ``g`` detached on the dual side. Swapping the
   two detaches couples the two optimizations through one graph.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_risk_flow_dual.py
"""

import argparse
import os
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the primal-dual stage of RiskFlow.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--rollout_steps", type=int, default=400, help="Interleaved environment steps.")
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
    "update_actor": True,
    "update_dual": True,
}

CRITIC_LOSS_BLOWUP_FACTOR = 100.0


def _box(dim):
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _build_value_critic(device):
    model = {"critic": RA_Critic(CONSTRAINT_DIM, device)}
    agent = ReachAvoid(model, None, device=device, cfg=dict(V_N_AGENT_CFG))
    if args_cli.v_n_checkpoint is not None:
        agent.load(os.path.abspath(args_cli.v_n_checkpoint))
    return agent.critic


def check_direction(agent, device, emit, failures):
    """A violated constraint raises the multiplier; slack lowers it."""
    batch = 512

    def drive(sign):
        before = agent.lagrange().item()
        for _ in range(50):
            agent._update_dual(sign * torch.rand(batch, device=device))
        return before, agent.lagrange().item()

    violated_before, violated_after = drive(+1.0)
    slack_before, slack_after = drive(-1.0)

    emit(f"[P4.3] direction: g>0 lambda {violated_before:.5f} -> {violated_after:.5f}, "
         f"g<0 lambda {slack_before:.5f} -> {slack_after:.5f}")
    if violated_after <= violated_before:
        failures.append(f"a violating batch (g > 0) did not raise lambda "
                        f"({violated_before:.5f} -> {violated_after:.5f}); loss_lam has the wrong sign")
    if slack_after >= slack_before:
        failures.append(f"a slack batch (g < 0) did not lower lambda "
                        f"({slack_before:.5f} -> {slack_after:.5f}); loss_lam has the wrong sign")


def check_non_negativity(agent, device, emit, failures):
    """softplus keeps lambda in [0, inf) without a projection step."""
    values = []
    for nu in (-1.0e3, -50.0, -1.0, 0.0, 5.0):
        with torch.no_grad():
            agent.lagrange.nu.fill_(nu)
        multiplier = agent.lagrange()
        gradient = torch.autograd.grad(multiplier, agent.lagrange.nu)[0]
        values.append((nu, multiplier.item(), gradient.item()))
        if multiplier.item() < 0.0:
            failures.append(f"lambda is negative at nu={nu}")
        if not np.isfinite(multiplier.item()) or not np.isfinite(gradient.item()):
            failures.append(f"lambda or its gradient is not finite at nu={nu}")

    emit("[P4.3] softplus: " + ", ".join(f"nu={nu:g} -> lambda={value:.4g} (d/dnu {gradient:.2g})"
                                         for nu, value, gradient in values))


def check_serialization(agent, emit, failures):
    """nu has to survive a checkpoint round-trip."""
    with torch.no_grad():
        agent.lagrange.nu.fill_(1.234)
    expected = agent.lagrange().item()

    state = agent.lagrange.state_dict()
    if "nu" not in state:
        failures.append(f"nu is not in the multiplier's state_dict: {list(state.keys())}")
    if "lagrange" not in agent.checkpoint_modules:
        failures.append("the multiplier is not in checkpoint_modules; a resumed run would restart "
                        "from the initial lambda")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "checkpoints", "agent.pt")
        agent.save(path)
        with torch.no_grad():
            agent.lagrange.nu.fill_(-9.0)
        agent.load(path)

    restored = agent.lagrange().item()
    emit(f"[P4.3] serialization: lambda {expected:.6f} -> saved -> clobbered -> {restored:.6f}")
    if abs(restored - expected) > 1e-6:
        failures.append(f"lambda did not survive the checkpoint round-trip "
                        f"({expected:.6f} vs {restored:.6f})")


def check_effective_weight(agent, device, emit, failures):
    """The actor loss must weight D_H by 1/H + lambda, and detach on exactly one side of each product.

    The critic is replaced by a leaf tensor, so the gradient the actor loss puts on each head can be
    read directly instead of inferred.
    """
    batch = 64
    with torch.no_grad():
        agent.lagrange.nu.fill_(1.0)
    multiplier = agent.lagrange().item()

    observations = torch.randn(batch, OBS_DIM, device=device)
    constraint_states = torch.randn(batch, CONSTRAINT_DIM, device=device)
    flow = torch.randn(batch, HORIZON, device=device, requires_grad=True)

    original_critic = agent.critic
    stub = lambda states_in, actions_in, update_rms=False: flow  # noqa: E731
    stub.frozen = original_critic.frozen
    agent.critic = stub
    agent.lagrange.zero_grad(set_to_none=True)
    try:
        agent._update_actor(observations, constraint_states)
    finally:
        agent.critic = original_critic

    gradient = flow.grad
    # the outer mean over the batch scales every entry by 1/batch
    interior = gradient[:, 0] * batch
    terminal = gradient[:, -1] * batch
    expected_interior = 1.0 / HORIZON
    expected_terminal = 1.0 / HORIZON + multiplier

    emit(f"[P4.3] effective weight (lambda={multiplier:.4f}): interior head {interior.mean().item():.6f} "
         f"(expect {expected_interior:.6f}), D_H {terminal.mean().item():.6f} "
         f"(expect {expected_terminal:.6f})")
    if abs(interior.mean().item() - expected_interior) > 1e-5:
        failures.append(f"an interior head is weighted {interior.mean().item():.6f}, expected 1/H = "
                        f"{expected_interior:.6f}")
    if abs(terminal.mean().item() - expected_terminal) > 1e-5:
        failures.append(f"D_H is weighted {terminal.mean().item():.6f}, expected 1/H + lambda = "
                        f"{expected_terminal:.6f}")

    # lambda is detached on the policy side: the actor's backward must not touch nu
    if agent.lagrange.nu.grad is not None:
        failures.append("the actor loss left a gradient on nu; lambda is not detached on the "
                        "policy side")

    # g is detached on the dual side: the dual's backward must not touch the policy
    agent.actor.zero_grad(set_to_none=True)
    agent._update_dual(torch.randn(batch, device=device))
    actor_touched = [name for name, parameter in agent.actor.named_parameters() if parameter.grad is not None]
    if actor_touched:
        failures.append(f"the dual update left gradients on the policy: {actor_touched[:3]}; "
                        f"g is not detached on the dual side")


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit(f"[P4.3] device={device}, num_envs={args_cli.num_envs}")

    torch.manual_seed(0)
    value_critic = _build_value_critic(device)

    cfg = G1RiskEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.events.reset_state_from_dataset = None
    env = G1RiskEnv(cfg)

    buffer = RiskFlowBuffer(buffer_size=args_cli.rollout_steps, num_envs=args_cli.num_envs, device=device)
    buffer.init_buffer(_box(OBS_DIM), _box(CONSTRAINT_DIM), _box(NUM_ACTIONS))

    model = {
        "critic": MultiHorizonCritic(CONSTRAINT_DIM, NUM_ACTIONS, HORIZON, device=device),
        "actor": DeterministicActor(OBS_DIM, NUM_ACTIONS, device=device),
    }
    agent = RiskFlow(model, buffer, value_critic, env.get_torque_model(), device=device,
                     cfg=dict(RISK_FLOW_CFG))

    emit(f"[P4.3] initial: nu={agent.lagrange.nu.item():.4f} -> lambda={agent.lagrange().item():.4f} "
         f"(1/H = {1.0 / HORIZON:.4f})")

    check_direction(agent, device, emit, failures)
    check_non_negativity(agent, device, emit, failures)
    check_serialization(agent, emit, failures)
    check_effective_weight(agent, device, emit, failures)

    # -- the dual switch must refuse a configuration it cannot act on
    try:
        RiskFlow(model, buffer, value_critic, env.get_torque_model(), device=device,
                 cfg={**RISK_FLOW_CFG, "update_actor": False, "update_dual": True})
        failures.append("update_dual without update_actor was accepted")
    except ValueError:
        pass

    # -- all three updates running together, at one gradient step per environment step
    model = {
        "critic": MultiHorizonCritic(CONSTRAINT_DIM, NUM_ACTIONS, HORIZON, device=device),
        "actor": DeterministicActor(OBS_DIM, NUM_ACTIONS, device=device),
    }
    buffer = RiskFlowBuffer(buffer_size=args_cli.rollout_steps, num_envs=args_cli.num_envs, device=device)
    buffer.init_buffer(_box(OBS_DIM), _box(CONSTRAINT_DIM), _box(NUM_ACTIONS))
    agent = RiskFlow(model, buffer, value_critic, env.get_torque_model(), device=device,
                     cfg=dict(RISK_FLOW_CFG))

    obs, _, constraint_states, _ = env.reset()
    history = []
    interval_violations = []
    for step in range(args_cli.rollout_steps):
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
        obs, constraint_states = next_obs, next_constraint_states

        info = agent.update()
        if info is None:
            continue
        if not all(np.isfinite(info[key]) for key in ("critic_loss", "actor_loss", "lambda")):
            failures.append(f"a quantity went non-finite at step {step}")
            break
        interval_violations.append(info["constraint_violation"])
        if step % max(1, args_cli.rollout_steps // 4) == 0 or step == args_cli.rollout_steps - 1:
            history.append((step, info["critic_loss"], info["actor_loss"], info["lambda"],
                            info["constraint_violation"], info["violation_ratio"], info["action_rms"],
                            float(np.mean(interval_violations))))
            interval_violations = []

    for step, critic_loss, actor_loss, multiplier, violation, ratio, action_rms, _ in history:
        emit(f"[P4.3]   step {step:>4}: critic {critic_loss:.3e}  actor {actor_loss:+.3e}  "
             f"lambda {multiplier:.4f}  g {violation:+.3e}  violating {ratio:.2f}  "
             f"|a|rms {action_rms:.3f}")

    if any(entry[3] < 0.0 for entry in history):
        failures.append("lambda went negative during training")
    critic_losses = [entry[1] for entry in history]
    if critic_losses and critic_losses[-1] > CRITIC_LOSS_BLOWUP_FACTOR * min(critic_losses):
        failures.append(f"the critic loss blew up once the dual was live: "
                        f"{min(critic_losses):.3e} -> {critic_losses[-1]:.3e}")

    # lambda has to track the constraint it is pricing, not drift on its own.
    #
    # The comparison uses g averaged over each interval, not sampled at its endpoints: lambda
    # integrates every step in between, and g crosses zero inside the first interval here, so an
    # endpoint sample says the opposite of what lambda actually integrated.
    if len(history) >= 2:
        multipliers = [entry[3] for entry in history]
        interval_means = [entry[7] for entry in history]
        moved_together = [(multipliers[i + 1] - multipliers[i]) * np.sign(interval_means[i + 1])
                          for i in range(len(history) - 1)]
        emit("[P4.3] lambda vs interval-mean g: " + ", ".join(
            f"g {interval_means[i + 1]:+.3e} -> dlambda {multipliers[i + 1] - multipliers[i]:+.4f}"
            for i in range(len(history) - 1)))
        if any(step_change < 0 for step_change in moved_together):
            failures.append("lambda moved against the sign of the violation averaged over the interval")

    if failures:
        emit(f"[P4.3] FAILED ({len(failures)})")
        for failure in failures[:20]:
            emit(f"        - {failure}")
    else:
        emit("[P4.3] PASSED: the dual step has the right sign, lambda stays non-negative and "
             "serializable, and D_H carries exactly 1/H + lambda.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4_3_result.txt"), "w") as handle:
        handle.write(chr(10).join(report) + chr(10))

    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
