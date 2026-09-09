# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P3.1 verification: the Risk-Flow networks.

Checks the four things the models have to get right before the agent is wired up: the output
shapes, the gradient path the deterministic policy gradient depends on, the initial output scale
that stands in for the absent squashing, and the checkpoint round-trip -- including the frozen
``V_N``, which is restored under the same RA_Critic/ReachAvoid structure it was trained with.

The absence of action clipping is verified by reading the code (``_pre_physics_step`` scales and
nothing else, and the action-space box is not narrow), not by a runtime test.

Run with:
    C:\\IsaacLab\\isaaclab.bat -p src/scripts/constraints/diagnostics/test_risk_flow_models.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the Risk-Flow networks.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import tempfile

import torch

from lib.agent.reach_avoid import ReachAvoid
from lib.model.MLP import RA_Critic

from models.risk_flow_models import DeterministicActor, MultiHorizonCritic

CONSTRAINT_DIM = 67
OBS_DIM = 96
NUM_ACTIONS = 29
HORIZON = 99
BATCH = 64

# only what ReachAvoid.__init__ reads; the frozen network is never updated through it
V_N_AGENT_CFG = {
    "seed": 0,
    "learning_epochs": 1,
    "batch_size": 1,
    "learning_rate": 1.0e-4,
    "discount_factor": 0.99,
    "grad_norm_clip": 1.0,
    "learning_starts": 0,
}


def main():
    report = []

    def emit(line):
        report.append(line)
        print(line, flush=True)

    failures = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit(f"[P3.1] device={device}")

    torch.manual_seed(0)
    critic = MultiHorizonCritic(CONSTRAINT_DIM, NUM_ACTIONS, HORIZON, device=device)
    actor = DeterministicActor(OBS_DIM, NUM_ACTIONS, device=device)

    states = torch.randn(BATCH, CONSTRAINT_DIM, device=device)
    observations = torch.randn(BATCH, OBS_DIM, device=device)

    # -- check 1: shapes, and the initial output scale
    actions = actor(observations)
    values = critic(states, actions)
    emit(f"[P3.1] pi(o) -> {tuple(actions.shape)}, D(s, a) -> {tuple(values.shape)}")
    if tuple(actions.shape) != (BATCH, NUM_ACTIONS):
        failures.append(f"actor output shape {tuple(actions.shape)}")
    if tuple(values.shape) != (BATCH, HORIZON):
        failures.append(f"critic output shape {tuple(values.shape)}, expected ({BATCH}, {HORIZON})")

    action_absmax = actions.abs().max().item()
    value_absmax = values.abs().max().item()
    emit(f"[P3.1] initial |a|_max={action_absmax:.4f}, |D|_max={value_absmax:.4f}")
    # There is no squashing, so the small output gain is the only thing keeping the first actions
    # inside a sane range; a loose bound is enough to catch a gain that was not applied.
    if action_absmax > 1.0:
        failures.append(f"initial |a|_max={action_absmax:.4f} is far from zero; output gain not applied?")
    if value_absmax > 1.0:
        failures.append(f"initial |D|_max={value_absmax:.4f} is far from zero; output gain not applied?")

    # -- check 2a: the deterministic policy gradient path, grad_a D
    probe_actions = actions.detach().clone().requires_grad_(True)
    critic(states, probe_actions).mean(dim=1).sum().backward()
    action_grad = probe_actions.grad
    if action_grad is None:
        failures.append("grad_a D is None; the critic broke the graph to the action")
    else:
        emit(f"[P3.1] |grad_a D|_mean={action_grad.abs().mean().item():.3e}")
        if not torch.isfinite(action_grad).all():
            failures.append("grad_a D contains non-finite values")
        if action_grad.abs().max().item() == 0.0:
            failures.append("grad_a D is identically zero")

    # -- check 2b: the freeze toggle really keeps gradients out of the critic parameters
    critic.zero_grad(set_to_none=True)
    actor.zero_grad(set_to_none=True)
    with critic.frozen():
        loss = critic(states, actor(observations)).mean()
        loss.backward()

    leaked = [name for name, parameter in critic.named_parameters() if parameter.grad is not None]
    actor_grads = [parameter.grad is not None for parameter in actor.parameters()]
    emit(f"[P3.1] frozen critic: params with grad={len(leaked)}, actor params with grad="
         f"{sum(actor_grads)}/{len(actor_grads)}")
    if leaked:
        failures.append(f"critic parameters received gradients while frozen: {leaked[:3]}")
    if not all(actor_grads):
        failures.append("the actor did not receive gradients through the frozen critic")

    # the toggle must restore what it found
    if not all(parameter.requires_grad for parameter in critic.parameters()):
        failures.append("critic.frozen() did not restore requires_grad on exit")

    # -- check 4a: the standardizer travels with the checkpoint
    for name, model in (("critic", critic), ("actor", actor)):
        keys = model.state_dict().keys()
        rms_keys = [key for key in keys if "standardizer" in key]
        if not {f"{name}_standardizer.mean", f"{name}_standardizer.var",
                f"{name}_standardizer.count"}.issubset(set(rms_keys)):
            failures.append(f"{name} state_dict is missing standardizer statistics: {rms_keys}")

    # feed a shifted batch so the statistics are no longer the defaults, then round-trip
    critic(states * 3.0 + 5.0, actions, update_rms=True)
    actor(observations * 3.0 + 5.0, update_rms=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "models.pt")
        torch.save({"critic": critic.state_dict(), "actor": actor.state_dict()}, path)

        restored_critic = MultiHorizonCritic(CONSTRAINT_DIM, NUM_ACTIONS, HORIZON, device=device)
        restored_actor = DeterministicActor(OBS_DIM, NUM_ACTIONS, device=device)
        payload = torch.load(path, map_location=device, weights_only=False)
        restored_critic.load_state_dict(payload["critic"])
        restored_actor.load_state_dict(payload["actor"])

        with torch.no_grad():
            same_actor = torch.equal(actor(observations), restored_actor(observations))
            same_critic = torch.equal(critic(states, actions), restored_critic(states, actions))
        emit(f"[P3.1] round-trip bit-exact: actor={same_actor}, critic={same_critic}")
        if not same_actor:
            failures.append("actor output changed across a save/load round-trip")
        if not same_critic:
            failures.append("critic output changed across a save/load round-trip")

        # -- check 4b: the frozen V_N is restored under its own training-time structure
        #    RA_Critic held by a ReachAvoid agent, exactly as main/reach_avoid/play.py builds it.
        #    Nothing about the network may differ between the run that trained it and this one.
        source_model = {"critic": RA_Critic(CONSTRAINT_DIM, device)}
        source_agent = ReachAvoid(source_model, None, device=device, cfg=dict(V_N_AGENT_CFG))
        # give the statistics a recognizable, non-default value
        source_agent.critic.critic_standardizer.mean.fill_(0.5)
        source_agent.critic.critic_standardizer.var.fill_(2.0)
        value_path = os.path.join(tmp, "checkpoints", "v_n.pt")
        source_agent.save(value_path)

        frozen_model = {"critic": RA_Critic(CONSTRAINT_DIM, device)}
        frozen_agent = ReachAvoid(frozen_model, None, device=device, cfg=dict(V_N_AGENT_CFG))
        frozen_agent.load(value_path)
        standardizer = frozen_agent.critic.critic_standardizer

        stats_before = (standardizer.mean.clone(), standardizer.var.clone(), standardizer.count.clone())
        if not torch.allclose(stats_before[0], torch.full_like(stats_before[0], 0.5)):
            failures.append("V_N standardizer statistics were not restored from the checkpoint")

        with torch.no_grad():
            reference, _, _ = frozen_agent.critic(states)
            # hammer it with off-distribution data, the way an intervention rollout would
            for _ in range(10):
                frozen_agent.critic(torch.randn(BATCH, CONSTRAINT_DIM, device=device) * 50.0 + 100.0)

        drifted = [
            name for name, before, after in (
                ("mean", stats_before[0], standardizer.mean),
                ("var", stats_before[1], standardizer.var),
                ("count", stats_before[2], standardizer.count),
            ) if not torch.equal(before, after)
        ]
        emit(f"[P3.1] frozen V_N: shape={tuple(reference.shape)}, drifted stats={drifted}, "
             f"training={frozen_agent.critic.training}")
        if drifted:
            failures.append(f"frozen V_N statistics drifted: {drifted}")
        if frozen_agent.critic.training:
            failures.append("the restored V_N agent is not in eval mode")
        if tuple(reference.shape) != (BATCH, 1):
            failures.append(f"V_N output shape {tuple(reference.shape)}")

    if failures:
        emit(f"[P3.1] FAILED ({len(failures)})")
        for failure in failures[:20]:
            emit(f"        - {failure}")
    else:
        emit("[P3.1] PASSED: shapes, gradient path, freeze toggle, and a drift-free frozen V_N.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p3_1_result.txt"), "w") as handle:
        handle.write(chr(10).join(report) + chr(10))

    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
