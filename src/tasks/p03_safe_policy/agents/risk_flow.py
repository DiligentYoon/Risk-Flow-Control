# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Risk-Flow actor-critic agent.

Brought up in three stages, one piece at a time (plan.md, Phase 4):

* P4.1 -- the multi-horizon critic, trained by one-step TD against a *fixed* policy.
* P4.2 -- the actor update with ``lambda = 0``.
* **P4.3 (current)** -- the primal-dual multiplier of the terminal recovery constraint.

Which pieces are live is a configuration choice (``update_actor`` / ``update_dual``), so a stage
can be re-entered later: the ``lambda = 0`` ablation of P5.2 is this file with the dual off.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent

from tasks.p02_safety_value.agent.safety import Safety

from ..buffer.risk_flow_buffer import RiskFlowBuffer
from ..models.risk_flow_models import LagrangeMultiplier


class RiskFlow(Agent):
    """Learns ``D_phi(s, a)``, the flow of the frozen safety value over every horizon up to ``H``."""

    def __init__(
        self,
        model: Dict[str, nn.Module],
        buffer: Optional[RiskFlowBuffer],
        safety_value: Safety,
        device: Union[str, torch.device],
        cfg: Dict,
    ) -> None:
        super().__init__(cfg, model, device)

        # Models
        self.critic = self.model["critic"].to(self.device)
        self.actor = self.model["actor"].to(self.device)

        # Frozen V_N. This agent is only allowed to read it.
        self.safety_value = safety_value

        # Buffer
        self.buffer = buffer

        # Checkpoint models
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["actor"] = self.actor

        # Load parameters from cfg
        self.horizon = self.cfg["horizon"]
        self.batch_size = self.cfg["batch_size"]
        self.learning_starts = self.cfg["learning_starts"]
        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.exploration_sigma = self.cfg["exploration_sigma"]
        self.critic_learning_rate = self.cfg["critic_learning_rate"]
        self.actor_learning_rate = self.cfg["actor_learning_rate"]
        self.dual_learning_rate = self.cfg["dual_learning_rate"]
        self.terminal_risk_threshold = self.cfg["terminal_risk_threshold"]
        self.update_actor = self.cfg["update_actor"]
        self.update_dual = self.cfg["update_dual"]

        if self.critic.horizon != self.horizon:
            raise ValueError(
                f"cfg horizon ({self.horizon}) disagrees with the critic's head count "
                f"({self.critic.horizon}). Read H from the environment as "
                f"`max_episode_length - 1` and build both from the same value."
            )

        # Target critic.
        self.target_update_tau = self.cfg["target_update_tau"]
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.target_critic.eval()

        # Target actor
        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.requires_grad_(False)
        self.target_actor.eval()

        # Set up Adam optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
        self.checkpoint_modules["target_critic"] = self.target_critic

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.checkpoint_modules["actor_optimizer"] = self.actor_optimizer
        self.checkpoint_modules["target_actor"] = self.target_actor

        # Primal-dual multiplier. 
        self.lagrange = LagrangeMultiplier(self.cfg["lagrange_init"], device=self.device)
        self.dual_optimizer = torch.optim.Adam(self.lagrange.parameters(), lr=self.dual_learning_rate)
        self.checkpoint_modules["lagrange"] = self.lagrange
        self.checkpoint_modules["dual_optimizer"] = self.dual_optimizer

        self.actor.requires_grad_(True)

        self.tensors_names = [
            "observations",
            "states",
            "safety_states",
            "actions",
            "final_observations",
            "final_states",
            "final_safety_states",
            "terminated",
            "truncated",
        ]

        self.set_running_mode("eval")

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            actions = self.actor(observations)
            if not deterministic and self.exploration_sigma > 0.0:
                actions = actions + self.exploration_sigma * torch.randn_like(actions)

        return actions

    def insert_data(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        safety_states: torch.Tensor,
        actions: torch.Tensor,
        final_observations: torch.Tensor,
        final_states: torch.Tensor,
        final_safety_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        self.buffer.add_samples(
            observations=observations,
            states=states,
            safety_states=safety_states,
            actions=actions,
            final_observations=final_observations,
            final_states=final_states,
            final_safety_states=final_safety_states,
            terminated=terminated,
            truncated=truncated,
        )

    @torch.no_grad()
    def compute_target(
        self,
        safety_states: torch.Tensor,
        final_safety_states: torch.Tensor,
        final_observations: torch.Tensor,
        final_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """Build the one-step TD target of every horizon head.

        ``D_h`` is the expected change of ``V_N`` over ``h`` steps, so the recursion is
        ``D_h(s, a) = Delta_N + D_{h-1}(s', a')`` with ``D_0 = 0``::

            y[:, 0]  = delta
            y[:, 1:] = delta + m * D_next[:, :-1]

        The shift by one head is the whole content of the update: head ``h`` is supervised by head
        ``h - 1`` of the next state, which is why an off-by-one here stays invisible in the loss
        curve -- the recursion would simply be consistent with a different quantity.
        
        """
        # Delta_N is recomputed from the stored states rather than read back from the buffer:
        # storing it would pin every sample to one particular V_N.
        value = self.safety_value.predict(safety_states)
        next_value = self.safety_value.predict(final_safety_states)
        delta = next_value - value

        next_actions = self.target_actor(final_observations)
        next_flow = self.target_critic(final_states, next_actions) # [B, H]

        mask = (~terminated).to(dtype=next_flow.dtype) # Assumption : risk stays at its maximum after failure.

        target = torch.empty_like(next_flow)
        target[:, :1] = delta
        # If the next state is terminated, no more change of risk is assumed.
        target[:, 1:] = delta + mask * next_flow[:, :-1] # [B, H-1] + [B, 1] x [B, H-1]

        return target

    @torch.no_grad()
    def update_target(self) -> None:
        for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_parameter.mul_(1.0 - self.target_update_tau).add_(parameter, alpha=self.target_update_tau)

        for target_buffer, buffer in zip(self.target_critic.buffers(), self.critic.buffers()):
            target_buffer.copy_(buffer)

        for target_parameter, parameter in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_parameter.mul_(1.0 - self.target_update_tau).add_(parameter, alpha=self.target_update_tau)

        for target_buffer, buffer in zip(self.target_actor.buffers(), self.actor.buffers()):
            target_buffer.copy_(buffer)

    def update(self) -> Optional[Dict[str, Any]]:
        if not (len(self.buffer) >= self.learning_starts * self.buffer.num_envs):
            return None

        (
            observations,
            states,
            safety_states,
            actions,
            final_observations,
            final_states,
            final_safety_states,
            terminated,
            truncated,
        ) = self.buffer.sample_batch(self.tensors_names, self.batch_size)

        # Update Critic Network
        target = self.compute_target(safety_states, final_safety_states, final_observations, final_states, terminated)
        flow = self.critic(states, actions, update_rms=True)
        critic_loss = F.mse_loss(flow, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()

        # Update Actor Network and Dual Parameter.
        if self.update_actor:
            actor_info, violation = self._update_actor(observations, states, safety_states)
            dual_info = self._update_dual(violation) if violation is not None else {}

        self.update_target()

        if self.update_actor:

            return {"critic_loss": critic_loss.item(),
                    **actor_info,
                    **dual_info}
        else:
            return {"critic_loss": critic_loss.item()}

    def _update_actor(self, observations: torch.Tensor, states: torch.Tensor, safety_states: torch.Tensor) -> Dict[str, Any]:
        """Run one actor update.

        Minimizes the mean predicted risk flow plus the control cost::

            loss_pi = mean_h D_h(s, pi(o)) + beta * C_reg(s, pi(o))

        The ``1/H`` normalization means every horizon contributes equally, which is what expresses
        the preference for recovering *early* rather than merely by the deadline. The constraint
        term of research 4.4 is absent at this stage (``lambda = 0``).

        The critic is evaluated inside its ``frozen()`` context: ``grad_a D`` still flows back to
        the action, which is the whole content of the deterministic policy gradient, but no
        gradient is accumulated into the critic's own parameters. A leak there would let the actor
        loss quietly train the critic towards states it prefers.
        """
        actions = self.actor(observations, update_rms=True)

        with self.critic.frozen():
            flow = self.critic(states, actions)

        objective = flow.mean(dim=1)

        violation = None
        if self.update_dual:
            with torch.no_grad():
                value = self.safety_value.predict(safety_states)
            violation = value.squeeze(-1) + flow[:, -1] - self.terminal_risk_threshold
            objective = objective + self.lagrange().detach() * violation

        actor_loss = objective.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
        self.actor_optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "flow_mean": flow.mean().item(),
        }

        return info, (violation.detach() if violation is not None else None)

    def _update_dual(self, violation: torch.Tensor) -> Dict[str, Any]:
        """Run one dual update.

        Gradient *ascent* on ``lambda * mean(g)``, written as descent on its negative::

            loss_lam = -(lambda * mean(g))

        so a batch that violates the constraint (``mean(g) > 0``) pushes ``lambda`` up and a batch
        with slack pushes it down. The sign is the whole content of this update and it is easy to
        write backwards; a flipped one still produces a smooth training curve, just with the
        constraint pushing the wrong way for a long time before anything looks wrong.

        No gradient clipping here. ``nu`` is a single scalar stepped at ``dual_learning_rate``, so
        there is no norm to explode, and clipping would only mask a badly scaled ``g``.
        """
        multiplier = self.lagrange()
        dual_loss = -(multiplier * violation.mean())

        self.dual_optimizer.zero_grad()
        dual_loss.backward()
        self.dual_optimizer.step()

        return {
            "lambda": self.lagrange().detach().item(),
            "constraint_violation": violation.mean().item(),
            "violation_ratio": (violation > 0).to(dtype=violation.dtype).mean().item(),
        }
