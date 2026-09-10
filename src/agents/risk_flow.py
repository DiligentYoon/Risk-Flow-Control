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

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent

from buffer.risk_flow_buffer import RiskFlowBuffer
from models.risk_flow_models import LagrangeMultiplier


class RiskFlow(Agent):
    """Learns ``D_phi(s, a)``, the flow of the frozen safety value over every horizon up to ``H``.

    Args:
        model: ``{"critic": MultiHorizonCritic, "actor": DeterministicActor}``. The frozen value
            network is deliberately *not* a member of this dictionary -- see ``value_critic``.
        buffer: Replay buffer. Only required for training.
        value_critic: The frozen ``V_N``, i.e. the ``critic`` of a :class:`ReachAvoid` agent that
            was built and loaded exactly as it was during its own training.
        torque_model: Constants of the analytic PD torque surrogate, as published by task env.
        device: Device on which tensors are allocated.
        cfg: Configuration dictionary.
    """

    def __init__(
        self,
        model: Dict[str, nn.Module],
        buffer: Optional[RiskFlowBuffer],
        value_critic: nn.Module,
        torque_model: Dict[str, Any],
        device: Union[str, torch.device],
        cfg: Dict,
    ) -> None:
        super().__init__(cfg, model, device)

        # Models
        self.critic = self.model["critic"].to(self.device)
        self.actor = self.model["actor"].to(self.device)

        # Frozen V_N. This agent is only allowed to read it.
        self.value_critic = value_critic

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
        self.control_cost_scale = self.cfg["control_cost_scale"]
        self.dual_learning_rate = self.cfg["dual_learning_rate"]
        self.terminal_risk_threshold = self.cfg["terminal_risk_threshold"]
        self.update_dual = self.cfg["update_dual"]

        # Analytic PD torque surrogate. 
        # The environment publishes the gains and the offsets of `q - q_default` / `q_dot` inside the constraint state, 
        # so the cost is a differentiable function of the action with no extra stored channel.
        self.action_scale = torque_model["action_scale"]
        self.joint_stiffness = torque_model["stiffness"].to(self.device).unsqueeze(0)
        self.joint_damping = torque_model["damping"].to(self.device).unsqueeze(0)
        self.joint_pos_slice = slice(torque_model["joint_pos_id"],
                                     torque_model["joint_pos_id"] + self.critic.num_actions)
        self.joint_vel_slice = slice(torque_model["joint_pos_id"] + self.critic.num_actions,
                                     torque_model["joint_pos_id"] + self.critic.num_actions + self.critic.num_actions)

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
        # it carries how tight the constraint has turned out to be, which costs thousands of steps to rediscover.
        self.lagrange = LagrangeMultiplier(self.cfg["lagrange_init"], device=self.device)
        self.dual_optimizer = torch.optim.Adam(self.lagrange.parameters(), lr=self.dual_learning_rate)
        self.checkpoint_modules["lagrange"] = self.lagrange
        self.checkpoint_modules["dual_optimizer"] = self.dual_optimizer

        self.actor.requires_grad_(True)

        self.tensors_names = [
            "observations",
            "constraint_states",
            "actions",
            "final_observations",
            "final_constraint_states",
            "terminated",
            "truncated",
        ]

        # Default Mode : Evaluation for disconnecting gradient flow
        self.set_running_mode("eval")

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select the action to execute.

        The exploration noise is added here rather than inside the environment, so that the tensor
        returned is exactly the one that gets executed *and* the one that gets stored: an
        environment-side noise model would make the critic learn about an action that was never
        taken. It is uncorrelated Gaussian and is neither squashed nor clipped.

        Args:
            observations: Policy observations, shape (num_envs, observation_dim).
            deterministic: Drop the exploration noise.

        Returns:
            Actions, shape (num_envs, action_dim).
        """
        with torch.no_grad():
            actions = self.actor(observations)
            if not deterministic and self.exploration_sigma > 0.0:
                actions = actions + self.exploration_sigma * torch.randn_like(actions)

        return actions

    def insert_data(
        self,
        observations: torch.Tensor,
        constraint_states: torch.Tensor,
        actions: torch.Tensor,
        final_observations: torch.Tensor,
        final_constraint_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Store one transition per environment.

        ``final_*`` are the pre-reset snapshots published by :class:`ConstraintsEnv`, not the
        post-reset observations of the step tuple. On a terminal step the two differ, and it is the
        snapshot that carries the state the action actually led to.
        """
        self.buffer.add_samples(
            observations=observations,
            constraint_states=constraint_states,
            actions=actions,
            final_observations=final_observations,
            final_constraint_states=final_constraint_states,
            terminated=terminated,
            truncated=truncated,
        )

    def can_update(self) -> bool:
        """Whether enough transitions have been collected to start learning."""
        return len(self.buffer) >= self.learning_starts * self.buffer.num_envs

    @torch.no_grad()
    def compute_target(
        self,
        constraint_states: torch.Tensor,
        final_constraint_states: torch.Tensor,
        final_observations: torch.Tensor,
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

        ``m = (~terminated)`` cuts the bootstrap on a fall, encoding the assumption that risk stays
        at its maximum afterwards. A time-out is *not* masked: the snapshot preserved the true next
        state, so there is a real state to bootstrap from.

        The bootstrap reads the *target* critic, not the online one, so that the regression target
        of this batch does not move with the update that this batch produces.

        Args:
            constraint_states: ``s_t``, shape (batch, constraint_state_dim).
            final_constraint_states: ``s_{t+1}`` before the autoreset, same shape.
            final_observations: ``o_{t+1}`` before the autoreset, shape (batch, observation_dim).
            terminated: Termination flags, shape (batch, 1).

        Returns:
            Targets, shape (batch, horizon).
        """
        # Delta_N is recomputed from the stored states rather than read back from the buffer:
        # storing it would pin every sample to one particular V_N.
        value, _, _ = self.value_critic(constraint_states)
        next_value, _, _ = self.value_critic(final_constraint_states)
        delta = next_value - value

        next_actions = self.target_actor(final_observations)
        next_flow = self.target_critic(final_constraint_states, next_actions) # [B, H]

        mask = (~terminated).to(dtype=next_flow.dtype) # Assumption : risk stays at its maximum after failure.

        target = torch.empty_like(next_flow)
        target[:, :1] = delta # boundary condition D_0 = 0.
        # If the next state is terminated, no more change of risk is assumed.
        target[:, 1:] = delta + mask * next_flow[:, :-1] # [B, H-1] + [B, 1] x [B, H-1]

        return target

    def torque(self, constraint_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Analytic PD torque of an action, differentiable in that action.

        The action is a joint-position offset on top of the default pose and the low-level
        controller is PD, so::

            target = q_default + action_scale * a
            tau    = Kp * (target - q) - Kd * q_dot
                   = Kp * (action_scale * a - (q - q_default)) - Kd * q_dot

        The environment's own control-effort penalty cannot be used in its place for two reasons.
        It is a float the simulator already produced, so there is no graph to differentiate; and it
        describes the action that was *executed*, whereas the actor loss needs the cost of the
        action the current policy would take now in that replayed state. Those are different
        actions.

        Args:
            constraint_states: ``s``, carrying ``q - q_default`` and ``q_dot`` as channels.
            actions: Actions to price, shape (batch, num_actions).

        Returns:
            Joint torques, shape (batch, num_actions).
        """
        joint_pos_error = self.action_scale * actions - constraint_states[:, self.joint_pos_slice]

        return self.joint_stiffness * joint_pos_error - self.joint_damping * constraint_states[:, self.joint_vel_slice]

    def control_cost(self, constraint_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Mean squared PD torque, per sample.

        This is what keeps the raw, unsquashed actions bounded. It penalizes a physically
        meaningful quantity rather than enforcing an interval, so it carries more information than
        clipping and its gradient does not vanish anywhere. ``||a||^2`` is the special case of this
        expression at ``q - q_default = q_dot = 0``, which is why it is not a separate term.
        """
        return self.torque(constraint_states, actions).pow(2).mean(dim=-1)

    @torch.no_grad()
    def update_target(self) -> None:
        """Polyak-average the target critic towards the online one.

        Parameters are blended; the normalization statistics are *copied*. They are not learned
        quantities but a running description of the input distribution, and the online network has
        already moved on to them -- a lagging copy would standardize the target's input with
        statistics that no longer describe it, which is a second moving target rather than a
        stabilizer.
        """
        for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_parameter.mul_(1.0 - self.target_update_tau).add_(parameter, alpha=self.target_update_tau)

        for target_buffer, buffer in zip(self.target_critic.buffers(), self.critic.buffers()):
            target_buffer.copy_(buffer)

        for target_parameter, parameter in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_parameter.mul_(1.0 - self.target_update_tau).add_(parameter, alpha=self.target_update_tau)

        for target_buffer, buffer in zip(self.target_actor.buffers(), self.actor.buffers()):
            target_buffer.copy_(buffer)

    def update(self) -> Optional[Dict[str, Any]]:
        """Run one critic update.

        Returns:
            Logging quantities, or ``None`` while the buffer is still below ``learning_starts``.
        """
        if not self.can_update():
            return None

        (
            observations,
            constraint_states,
            actions,
            final_observations,
            final_constraint_states,
            terminated,
            truncated,
        ) = self.buffer.sample_batch(self.tensors_names, self.batch_size)

        # Update Critic Network
        target = self.compute_target(constraint_states, final_constraint_states, final_observations, terminated)
        flow = self.critic(constraint_states, actions, update_rms=True)
        critic_loss = F.mse_loss(flow, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()

        # Update Actor Network and Dual Parameter.
        actor_info, violation = self._update_actor(observations, constraint_states)
        dual_info = self._update_dual(violation) if violation is not None else {}

        self.update_target()

        with torch.no_grad():
            # The natural scale of D_h grows with h, so the aggregate loss is dominated by the far
            # heads. Per-head loss is what actually shows whether the horizon has converged.
            per_head_loss = (flow - target).pow(2).mean(dim=0)

        return {
            "critic_loss": critic_loss.item(),
            "per_head_loss": per_head_loss,
            **actor_info,
            **dual_info,
        }

    def _update_actor(self, observations: torch.Tensor, constraint_states: torch.Tensor) -> Dict[str, Any]:
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
            flow = self.critic(constraint_states, actions)

        control_cost = self.control_cost(constraint_states, actions)
        objective = flow.mean(dim=1) + self.control_cost_scale * control_cost

        violation = None
        if self.update_dual:
            # g = V_N(s) + D_H - delta_N, the predicted terminal risk against its budget. V_N(s)
            # does not depend on the action, so it shifts g without contributing any gradient.
            with torch.no_grad():
                value, _, _ = self.value_critic(constraint_states)
            violation = value.squeeze(-1) + flow[:, -1] - self.terminal_risk_threshold
            # lambda is a constant to the policy; the policy is a constant to lambda. Detaching on
            # exactly one side of each product is what keeps the two optimizations from chasing
            # each other through a shared graph.
            objective = objective + self.lagrange().detach() * violation

        actor_loss = objective.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
        self.actor_optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "control_cost": control_cost.mean().item(),
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
