# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Networks of the Risk-Flow actor-critic.

Three pieces:

* :class:`MultiHorizonCritic` -- ``D_phi(s, a) -> [D_1, ..., D_H]``, the predicted risk flow over
  every horizon up to the recovery deadline ``H``.
* :class:`DeterministicActor` -- ``pi_theta(o) -> a``, raw output, neither squashed nor clipped.
* :class:`LagrangeMultiplier` -- the primal-dual multiplier of the terminal recovery constraint.

The frozen safety value ``V_N`` is not defined here. It is restored with the very structure it was
trained under -- ``RA_Critic`` held by a ``ReachAvoid`` agent, loaded through ``agent.load()`` --
so that nothing about the network can differ between training and inference.

Unlike the submodule's models these forward methods return a single tensor rather than the
``(value, log_prob, mean)`` triple: there is no stochastic policy here, and the critic's output is
already a vector over horizons, so the extra slots would only be ``None``.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.model import Model
from lib.utils.Running_mean_std import RunningMeanStd


class MultiHorizonCritic(Model):
    """Risk-flow critic ``D_phi(s, a)``.

    Predicts, for every horizon ``h`` in ``1..H``, the expected change of the frozen safety value
    over the next ``h`` steps::

        D_h(s, a) = E[ V_N(s_{t+h}) - V_N(s_t) | s_t = s, a_t = a ]

    ``D_0 = 0`` holds by definition and is a constant, not a network output, so the head count is
    exactly ``H``.

    The last layer is initialized with a small orthogonal gain: the one-step TD target of head ``h``
    is built from head ``h-1``, so a large initial output at the far heads is propagated backwards
    through the whole horizon before it decays.

    Args:
        num_states: Dimension of the constraint state ``s`` (the frozen networks' input).
        num_actions: Dimension of the action.
        horizon: Recovery deadline ``H``, i.e. the number of heads. Read this from the environment
            as ``env.max_episode_length - 1``; the episode is one step shorter than the configured
            length, and a head that no episode ever reaches is never trained.
        device: Device the model lives on.
        output_gain: Orthogonal gain of the output layer.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        horizon: int,
        device: Union[str, torch.device] = "cuda:0",
        output_gain: float = 0.01,
    ) -> None:
        super().__init__()

        self.device = device
        self.num_states = num_states
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_inputs = num_states + num_actions

        # Running mean, standard deviation standardizer over concat(s, a).
        # Trained alongside the critic, and unrelated to the frozen V_N's own statistics.
        self.critic_standardizer = RunningMeanStd(shape=self.num_inputs, device=device)

        # Backbone
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.horizon),
        )

        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)
        nn.init.orthogonal_(self.net[-1].weight, gain=output_gain)

        self.to(device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        """Forward propagation of the risk-flow critic.

        Args:
            states: Constraint states, shape (batch, num_states).
            actions: Actions, shape (batch, num_actions).
            update_rms: Whether to update the input standardizer with this batch.

        Returns:
            Risk flow per horizon, shape (batch, horizon).
        """
        inputs = torch.cat([states, actions], dim=-1)
        standardized_input = self.critic_standardizer.standardize(inputs, update=update_rms)

        return self.net(standardized_input)

    def set_requires_grad(self, flag: bool) -> None:
        """Toggle gradient tracking of every critic parameter."""
        for parameter in self.parameters():
            parameter.requires_grad_(flag)

    @contextlib.contextmanager
    def frozen(self) -> Iterator["MultiHorizonCritic"]:
        """Evaluate the critic without accumulating gradients into its own parameters.

        Used by the actor update: ``grad_a D`` still flows back to the action, which is what the
        deterministic policy gradient needs, but ``grad_phi D`` is never built.
        """
        flags = [parameter.requires_grad for parameter in self.parameters()]
        self.set_requires_grad(False)
        try:
            yield self
        finally:
            for parameter, flag in zip(self.parameters(), flags):
                parameter.requires_grad_(flag)


class DeterministicActor(Model):
    """Deterministic policy ``pi_theta(o)``.

    The raw output is the action: no ``tanh``, no clipping anywhere on this path. In saturation
    ``da/dz -> 0`` would kill the policy gradient ``J_pi^T grad_a D``, and risk minimization asks
    for exactly the large corrective torques that saturate. Magnitude is penalized by the analytic
    PD torque surrogate in the actor loss instead.

    That leaves the output layer's small orthogonal gain as the only thing keeping the initial
    actions near zero, so it is deliberate rather than cosmetic.

    Args:
        num_observations: Dimension of the policy observation ``o``.
        num_actions: Dimension of the action.
        device: Device the model lives on.
        output_gain: Orthogonal gain of the output layer.
    """

    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        device: Union[str, torch.device] = "cuda:0",
        output_gain: float = 0.01,
    ) -> None:
        super().__init__()

        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions

        # Running mean, standard deviation standardizer
        self.actor_standardizer = RunningMeanStd(shape=self.num_observations, device=device)

        # Backbone (last layer linear -- no activation, no squashing)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )

        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)
        nn.init.orthogonal_(self.net[-1].weight, gain=output_gain)

        self.to(device)

    def forward(self, observations: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        """Forward propagation of the deterministic actor.

        Args:
            observations: Policy observations, shape (batch, num_observations).
            update_rms: Whether to update the observation standardizer with this batch.

        Returns:
            Raw actions, shape (batch, num_actions).
        """
        standardized_input = self.actor_standardizer.standardize(observations, update=update_rms)

        return self.net(standardized_input)

class LagrangeMultiplier(nn.Module):
    """The multiplier of the terminal recovery constraint ``V_N(s) + D_H <= delta_N``.

    ``lambda`` is not stored directly but as ``softplus(nu)``. The multiplier of an inequality
    constraint has to stay non-negative, and the usual way to enforce that -- projecting back onto
    ``[0, inf)`` after each step -- leaves a kink at zero and an extra piece of state to keep
    consistent. The softplus parameterization makes ``lambda >= 0`` hold by construction, for every
    value ``nu`` can ever take, with a smooth gradient everywhere.

    It is an :class:`nn.Module` rather than a bare parameter so that it travels in the checkpoint
    like any other learned quantity: an agent resumed with ``lambda`` reset to its initial value
    would spend the first thousands of steps re-discovering how tight the constraint is.

    Args:
        initial_value: Initial value of ``nu``. Note that this is not ``lambda``: at the default of
            zero the multiplier starts at ``softplus(0) = ln 2 ~ 0.69``, which against the ``1/H``
            weight of the mean flow already gives the constraint the dominant say.
        device: Device the parameter lives on.
    """

    def __init__(self, initial_value: float = 0.0, device: Union[str, torch.device] = "cuda:0") -> None:
        super().__init__()

        self.device = device
        self.nu = nn.Parameter(torch.tensor(float(initial_value), device=device))

    def forward(self) -> torch.Tensor:
        """The current multiplier, a non-negative scalar."""
        return F.softplus(self.nu)
