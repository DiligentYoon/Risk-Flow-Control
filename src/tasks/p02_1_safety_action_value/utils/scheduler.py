# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


class StepScheduler(nn.Module):
    def __init__(self, init_value: float, period: int, decay: float, end_value: float, goal_value: float | None = None) -> None:
        super().__init__()
        self.init_value = init_value
        self.period = int(period)
        self.decay = decay
        self.end_value = end_value
        self.goal_value = None if goal_value is None else goal_value
        self.value = init_value

    @torch.no_grad()
    def step(self, timestep: int = 1) -> float:
        n = timestep // self.period
        if self.goal_value is None:
            value = self.init_value * self.decay ** n
            value = max(self.end_value, value) if self.init_value >= self.end_value else min(self.end_value, value)
        else:
            value = self.goal_value - (self.goal_value - self.init_value) * self.decay ** n
            value = min(self.end_value, value) if self.init_value <= self.end_value else max(self.end_value, value)
        self.value = value
        return value

    def get(self) -> float:
        return float(self.value)
