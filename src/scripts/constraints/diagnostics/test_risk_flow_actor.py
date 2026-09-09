# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P4.2 verification: the actor stage of :class:`RiskFlow` with ``lambda = 0``.

Performance is judged in P5.2. What is checked here is that switching the actor on does not break
the critic and that the deterministic policy gradient actually reaches the policy:

1. the actor update runs NaN-free and every policy parameter receives a gradient;
2. the actor loss leaves no gradient on the critic's parameters -- a leak there would let the
   policy quietly train the critic towards the states it happens to prefer;
3. the analytic PD torque surrogate is the right function, is differentiable in the action, and
   does not collapse to a constant.

The surrogate is additionally compared against the torque the simulator actually applied. That is
reported rather than gated: the comparison is taken across a decimated step, so the joint state has
moved on by the time the simulator's value is read.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_risk_flow_actor.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the actor stage of RiskFlow.")
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
    # P4.2 is the lambda = 0 stage; the dual arrives in P4.3
    "update_dual": False,
}

# The critic loss may drift, but it must not run away from its own best value.
CRITIC_LOSS_BLOWUP_FACTOR = 100.0


def _box(dim):
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _build_value_critic(device):
    model = {"critic": RA_Critic(CONSTRAINT_DIM, device)}
    agent = ReachAvoid(model, None, device=device, cfg=dict(V_N_AGENT_CFG))
    if args_cli.v_n_checkpoint is not None:
        agent.load(os.path.abspath(args_cli.v_n_checkpoint))
    return agent.critic


def check_torque_surrogate(agent, device, emit, failures):
    """The surrogate has to be the right function of the action, not merely a differentiable one."""
    batch = 16
    states = torch.randn(batch, CONSTRAINT_DIM, device=device)
    actions = torch.randn(batch, NUM_ACTIONS, device=device, requires_grad=True)

    # -- the closed form, written out independently of the implementation
    joint_pos = states[:, agent.joint_pos_slice]
    joint_vel = states[:, agent.joint_vel_slice]
    expected = (agent.joint_stiffness * (agent.action_scale * actions - joint_pos)
                - agent.joint_damping * joint_vel)
    error = (agent.torque(states, actions) - expected).abs().max().item()

    # -- the slices must be the ones the environment publishes, not an assumed layout
    layout_ok = (agent.joint_pos_slice.stop - agent.joint_pos_slice.start == NUM_ACTIONS
                 and agent.joint_vel_slice.stop - agent.joint_vel_slice.start == NUM_ACTIONS
                 and agent.joint_vel_slice.stop == CONSTRAINT_DIM)
    emit(f"[P4.2] torque surrogate: closed-form error={error:.3e}, "
         f"q slice={agent.joint_pos_slice.start}:{agent.joint_pos_slice.stop}, "
         f"q_dot slice={agent.joint_vel_slice.start}:{agent.joint_vel_slice.stop}, "
         f"Kp[0]={agent.joint_stiffness[0, 0].item():.1f}, Kd[0]={agent.joint_damping[0, 0].item():.2f}")
    if error > 1e-4:
        failures.append(f"the torque surrogate disagrees with its closed form (error {error:.3e})")
    if not layout_ok:
        failures.append("the joint slices do not cover the constraint state's joint block")

    # -- differentiable in the action, and not folded into a constant
    cost = agent.control_cost(states, actions)
    cost.sum().backward()
    gradient = actions.grad
    emit(f"[P4.2] control cost: mean={cost.mean().item():.4e}, "
         f"|grad_a C_reg|_mean={gradient.abs().mean().item():.4e}")
    if not torch.isfinite(gradient).all():
        failures.append("grad_a C_reg is not finite")
    if gradient.abs().max().item() == 0.0:
        failures.append("grad_a C_reg is identically zero; the surrogate collapsed to a constant")
    if cost.shape != (batch,):
        failures.append(f"control_cost returned shape {tuple(cost.shape)}, expected ({batch},)")

    # -- the gain on the action must be the environment's scale factor, not 1
    scaled = agent.torque(states, actions.detach() * 2.0) - agent.torque(states, torch.zeros_like(actions))
    unscaled = agent.joint_stiffness * (2.0 * agent.action_scale * actions.detach())
    if (scaled - unscaled).abs().max().item() > 1e-4:
        failures.append("the action scale factor is not applied inside the torque surrogate")


def check_gradient_isolation(agent, device, emit, failures):
    """The actor loss must move the policy and leave the critic untouched."""
    observations = torch.randn(256, OBS_DIM, device=device)
    constraint_states = torch.randn(256, CONSTRAINT_DIM, device=device)

    agent.critic.zero_grad(set_to_none=True)
    agent.actor.zero_grad(set_to_none=True)
    info, violation = agent._update_actor(observations, constraint_states)

    critic_with_grad = [name for name, parameter in agent.critic.named_parameters() if parameter.grad is not None]
    target_with_grad = [name for name, parameter in agent.target_critic.named_parameters()
                        if parameter.grad is not None]
    actor_grads = [parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in agent.actor.parameters()]

    emit(f"[P4.2] actor update: loss={info['actor_loss']:.4e}, flow={info['flow_mean']:.4e}, "
         f"C_reg={info['control_cost']:.4e}, |a|_max={info['action_absmax']:.4f}")
    emit(f"[P4.2] gradient isolation: critic params with grad={len(critic_with_grad)}, "
         f"target critic={len(target_with_grad)}, actor params with finite grad="
         f"{sum(actor_grads)}/{len(actor_grads)}")

    if not np.isfinite(info["actor_loss"]):
        failures.append("the actor loss is not finite")
    if critic_with_grad:
        failures.append(f"the actor loss left gradients on the critic: {critic_with_grad[:3]}")
    if target_with_grad:
        failures.append(f"the actor loss left gradients on the target critic: {target_with_grad[:3]}")
    if not all(actor_grads):
        failures.append("some policy parameters received no gradient, or a non-finite one")
    if not all(parameter.requires_grad for parameter in agent.critic.parameters()):
        failures.append("the critic was left frozen after the actor update")


def check_descent(agent, device, emit, failures):
    """With the critic held still, the actor update must walk downhill on its own objective."""
    observations = torch.randn(1024, OBS_DIM, device=device)
    constraint_states = torch.randn(1024, CONSTRAINT_DIM, device=device)

    losses = [agent._update_actor(observations, constraint_states)[0]["actor_loss"] for _ in range(50)]
    emit(f"[P4.2] actor descent (critic held): {losses[0]:.4e} -> {losses[-1]:.4e}")
    if not all(np.isfinite(loss) for loss in losses):
        failures.append("the actor loss went non-finite during repeated updates")
    elif losses[-1] >= losses[0]:
        failures.append(f"the actor loss did not decrease on a fixed batch "
                        f"({losses[0]:.4e} -> {losses[-1]:.4e}); the gradient step has the wrong sign")


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit(f"[P4.2] device={device}, num_envs={args_cli.num_envs}")

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

    check_torque_surrogate(agent, device, emit, failures)
    check_gradient_isolation(agent, device, emit, failures)
    check_descent(agent, device, emit, failures)

    # collection and updates interleaved at one gradient step per environment step.
    obs, _, constraint_states, _ = env.reset()
    surrogate_samples, applied_samples = [], []
    history = []
    num_terminated = 0

    for step in range(args_cli.rollout_steps):
        actions = agent.act(obs)
        with torch.no_grad():
            surrogate_samples.append(agent.torque(constraint_states, actions))

        next_obs, _, next_constraint_states, _, terminated, truncated, extras = env.step(actions)
        applied_samples.append(env._robot.data.applied_torque[:, env._joint_dof_ids].clone())
        num_terminated += int(terminated.sum().item())

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
        if not (np.isfinite(info["critic_loss"]) and np.isfinite(info["actor_loss"])):
            failures.append(f"a loss went non-finite at step {step}")
            break
        if step % max(1, args_cli.rollout_steps // 4) == 0 or step == args_cli.rollout_steps - 1:
            history.append((step, info["critic_loss"], info["actor_loss"], info["flow_mean"],
                            info["control_cost"], info["action_rms"], info["action_absmax"]))

    emit(f"[P4.2] interleaved {args_cli.rollout_steps} steps, {num_terminated} terminations, "
         f"1 gradient step per environment step")
    for step, critic_loss, actor_loss, flow, control_cost, action_rms, action_absmax in history:
        emit(f"[P4.2]   step {step:>4}: critic {critic_loss:.3e}  actor {actor_loss:+.3e}  "
             f"D {flow:+.3e}  C_reg {control_cost:.3e}  |a|rms {action_rms:.3f}  "
             f"|a|max {action_absmax:.3f}")

    # -- the surrogate against the torque the simulator actually applied
    surrogate = torch.cat(surrogate_samples).reshape(-1)
    applied = torch.cat(applied_samples).reshape(-1)
    centered_surrogate = surrogate - surrogate.mean()
    centered_applied = applied - applied.mean()
    correlation = (centered_surrogate @ centered_applied
                   / (centered_surrogate.norm() * centered_applied.norm())).item()
    emit(f"[P4.2] surrogate vs simulator torque: corr={correlation:.4f}, "
         f"rms surrogate={surrogate.pow(2).mean().sqrt().item():.2f}, "
         f"rms applied={applied.pow(2).mean().sqrt().item():.2f} (reported, not gated)")

    # Where the action magnitude settles is set by `beta`, and how large a control cost is
    # acceptable is decided in P5.2 against the recovery metrics -- not here. What this stage has to
    # rule out is the failure that would make those metrics meaningless: an actor that keeps walking
    # away from the action distribution the critic was fitted on, dragging the critic with it. So
    # the gate is on divergence, not on magnitude.
    magnitudes = [entry[6] for entry in history]
    emit(f"[P4.2] |a|max trajectory: " + " -> ".join(f"{magnitude:.2f}" for magnitude in magnitudes))
    if not all(np.isfinite(magnitude) for magnitude in magnitudes):
        failures.append("the action magnitude went non-finite")
    elif len(magnitudes) >= 3:
        early_growth = magnitudes[1] - magnitudes[0]
        late_growth = magnitudes[-1] - magnitudes[-2]
        emit(f"[P4.2] action growth per interval: first {early_growth:+.3f}, last {late_growth:+.3f}")
        if late_growth > max(early_growth, 0.0):
            failures.append(f"the action magnitude is still accelerating ({early_growth:+.3f} -> "
                            f"{late_growth:+.3f} per interval); the control cost is not holding it")

    critic_losses = [entry[1] for entry in history]
    if critic_losses and critic_losses[-1] > CRITIC_LOSS_BLOWUP_FACTOR * min(critic_losses):
        failures.append(f"the critic loss blew up under the actor updates: "
                        f"{min(critic_losses):.3e} -> {critic_losses[-1]:.3e}")

    # what `beta` is actually buying, for P5.2 to tune against
    if history:
        _, _, _, flow, control_cost, _, _ = history[-1]
        weighted = RISK_FLOW_CFG["control_cost_scale"] * control_cost
        emit(f"[P4.2] actor objective balance at the end: D {flow:+.3e} vs beta*C_reg {weighted:+.3e} "
             f"(beta={RISK_FLOW_CFG['control_cost_scale']:.0e}) -- the equilibrium magnitude is "
             f"beta's to set, and is judged in P5.2")

    if not torch.isfinite(agent.actor(torch.randn(8, OBS_DIM, device=device))).all():
        failures.append("the policy produces non-finite actions after training")

    if failures:
        emit(f"[P4.2] FAILED ({len(failures)})")
        for failure in failures[:20]:
            emit(f"        - {failure}")
    else:
        emit("[P4.2] PASSED: the policy gradient reaches the actor, the critic stays isolated, and "
             "the torque surrogate is exact and differentiable.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4_2_result.txt"), "w") as handle:
        handle.write(chr(10).join(report) + chr(10))

    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
