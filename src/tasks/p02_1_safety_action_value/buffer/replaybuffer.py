# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import gymnasium
import torch

from lib.buffer.buffer import Buffer


class ReplayBuffer(Buffer):
    def __init__(self, buffer_size: int = 1, num_envs: int = 1, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__(buffer_size, num_envs, device)

    def init_buffer(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space | None,
        safety_state_space: gymnasium.Space,
        action_space: gymnasium.Space,
    ) -> None:
        self.create_tensor("observations", observation_space, dtype=torch.float32)
        self.create_tensor("next_observations", observation_space, dtype=torch.float32)
        if state_space is not None:
            self.create_tensor("states", state_space, dtype=torch.float32)
            self.create_tensor("next_states", state_space, dtype=torch.float32)
        self.create_tensor("safety_states", safety_state_space, dtype=torch.float32)
        self.create_tensor("next_safety_states", safety_state_space, dtype=torch.float32)
        self.create_tensor("safety_values", 1, dtype=torch.float32)
        self.create_tensor("next_safety_values", 1, dtype=torch.float32)
        self.create_tensor("actions", action_space, dtype=torch.float32)
        self.create_tensor("rewards", 1, dtype=torch.float32)
        self.create_tensor("terminated", 1, dtype=torch.bool)
        self.create_tensor("truncated", 1, dtype=torch.bool)
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.zero_()

    def add_samples(
        self,
        observations: torch.Tensor,
        safety_states: torch.Tensor,
        safety_values: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        next_safety_states: torch.Tensor,
        next_safety_values: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        states: torch.Tensor | None = None,
        next_states: torch.Tensor | None = None,
    ) -> None:
        samples = {
            "observations": observations,
            "safety_states": safety_states,
            "safety_values": self._ensure_column(safety_values),
            "actions": actions,
            "next_observations": next_observations,
            "next_safety_states": next_safety_states,
            "next_safety_values": self._ensure_column(next_safety_values),
            "rewards": self._ensure_column(rewards),
            "terminated": self._ensure_column(terminated),
            "truncated": self._ensure_column(truncated),
        }
        if states is not None and "states" in self.tensors:
            samples["states"] = states
        if next_states is not None and "next_states" in self.tensors:
            samples["next_states"] = next_states
        super().add_samples(**samples)

    def sample(self, names: Tuple[str], mini_batches: int = 1) -> List[List[torch.Tensor]]:
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indexes = torch.randperm(size, dtype=torch.long, device=self.device)
        self.sampling_indexes = indexes
        return [[self.tensors_view[name][batch] for name in names] for batch in torch.tensor_split(indexes, mini_batches)]

    def sample_batch(self, names: Tuple[str], batch_size: int) -> List[torch.Tensor]:
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indexes = torch.randint(0, size, (batch_size,), dtype=torch.long, device=self.device)
        self.sampling_indexes = indexes
        return [self.tensors_view[name][indexes] for name in names]

    @staticmethod
    def _ensure_column(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor
