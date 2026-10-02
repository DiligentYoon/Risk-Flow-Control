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
        self.init_value = float(init_value)
        self.period = int(period)
        self.decay = float(decay)
        self.end_value = float(end_value)
        self.goal_value = None if goal_value is None else float(goal_value)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("value", torch.tensor(self.init_value, dtype=torch.float32))

    @torch.no_grad()
    def step(self, steps: int = 1) -> float:
        self.count.add_(int(steps))
        n = int(self.count.item()) // self.period
        if self.goal_value is None:
            value = self.init_value * self.decay ** n
            value = max(self.end_value, value) if self.init_value >= self.end_value else min(self.end_value, value)
        else:
            value = self.goal_value - (self.goal_value - self.init_value) * self.decay ** n
            value = min(self.end_value, value) if self.init_value <= self.end_value else max(self.end_value, value)
        self.value.fill_(value)
        return value

    def get(self) -> float:
        return float(self.value.item())
