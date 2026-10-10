# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch.distributions import Normal

from lib.model.model import Model
from lib.utils.Running_mean_std import RunningMeanStd

class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

class SafetyQActor(Model):
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        min_log_std: float,
        max_log_std: float,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        super().__init__()
        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.actor_standardizer = RunningMeanStd(shape=self.num_observations, device=device)

        self.mean = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, self.num_actions),
        )

        self.log_std = nn.Parameter(torch.full((num_actions,), -1.0), requires_grad=True)
        self.init_weights()
        self.init_biases(val=0)
        with torch.no_grad():
            self.mean[-1].weight.mul_(0.01)
        self.to(device)

    def forward(self, observations: torch.Tensor, deterministic: bool = False, update_rms: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.actor_standardizer.standardize(observations, update=update_rms)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        distribution = Normal(mean, log_std.exp())
        if deterministic:
            raw_actions = mean
        else:
            raw_actions = distribution.rsample()
        # actions = raw_actions
        # log_prob = distribution.log_prob(raw_actions)
        actions = torch.tanh(raw_actions)
        log_prob = distribution.log_prob(raw_actions) - torch.log(1.0 - actions.pow(2) + 1e-6)
        return actions, log_prob.sum(dim=-1, keepdim=True)


class SafetyQCritic(Model):
    def __init__(self, num_states: int, num_actions: int, device: Union[str, torch.device] = "cuda:0") -> None:
        super().__init__()
        self.device = device
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_inputs = self.num_states + self.num_actions
        self.critic_standardizer = RunningMeanStd(shape=self.num_inputs, device=device)

        self.q1 = nn.Sequential(
            nn.Linear(self.num_inputs, 256), Sin(),
            nn.Linear(256, 256), Sin(),
            nn.Linear(256, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(self.num_inputs, 256), Sin(),
            nn.Linear(256, 256), Sin(),
            nn.Linear(256, 1),
        )

        self.init_weights()
        self.init_biases(val=0)

        nn.init.zeros_(self.q1[-1].weight)
        nn.init.constant_(self.q1[-1].bias, 0.0)
        nn.init.zeros_(self.q2[-1].weight)
        nn.init.constant_(self.q2[-1].bias, 0.0)
        self.to(device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, update_rms: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([states, actions], dim=-1)
        x = self.critic_standardizer.standardize(x, update=update_rms)
        return self.q1(x), self.q2(x)