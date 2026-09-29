# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay buffer of the Risk-Flow actor-critic."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import gymnasium
import torch

from lib.buffer.buffer import Buffer


class RiskFlowBuffer(Buffer):
    """Circular replay buffer holding exactly what the one-step TD target needs."""

    def __init__(self, buffer_size: int = 1, num_envs: int = 1, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__(buffer_size, num_envs, device)

    def init_buffer(self, observation_space: gymnasium.Space, state_space: gymnasium.Space, safety_state_space: gymnasium.Space, action_space: gymnasium.Space) -> None:
        """Allocate every tensor at once."""
        self.create_tensor("observations", observation_space, dtype=torch.float32)
        self.create_tensor("final_observations", observation_space, dtype=torch.float32)
        self.create_tensor("states", state_space, dtype=torch.float32)
        self.create_tensor("final_states", state_space, dtype=torch.float32)
        self.create_tensor("safety_states", safety_state_space, dtype=torch.float32)
        self.create_tensor("final_safety_states", safety_state_space, dtype=torch.float32)
        self.create_tensor("actions", action_space, dtype=torch.float32)
        self.create_tensor("terminated", 1, dtype=torch.bool)
        self.create_tensor("truncated", 1, dtype=torch.bool)

        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.zero_()

    def add_samples(
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
        """Store one transition per environment."""
        super().add_samples(
            observations=observations,
            states=states,
            safety_states=safety_states,
            actions=actions,
            final_observations=final_observations,
            final_states=final_states,
            final_safety_states=final_safety_states,
            terminated=self._ensure_2d_column(terminated),
            truncated=self._ensure_2d_column(truncated),
        )

    def sample(self, names: Tuple[str], mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Shuffle the valid transitions and split them into mini-batches.

        Args:
            names: Tensor names to sample.
            mini_batches: Number of mini-batches.

        Returns:
            List of mini-batches, each a list of tensors ordered as ``names``.
        """
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indexes = torch.randperm(size, dtype=torch.long, device=self.device)

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)

    def sample_batch(self, names: Tuple[str], batch_size: int) -> List[torch.Tensor]:
        """Draw one batch uniformly at random, with replacement.

        Args:
            names: Tensor names to sample.
            batch_size: Number of transitions in the batch.

        Returns:
            List of tensors ordered as ``names``, each of shape (batch_size, data size).
        """
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indexes = torch.randint(
            low=0,
            high=size,
            size=(batch_size,),
            dtype=torch.long,
            device=self.device,
        )

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=1)[0]

    @staticmethod
    def _ensure_2d_column(tensor: torch.Tensor) -> torch.Tensor:
        """Give a flag tensor the (num_envs, 1) shape the storage expects."""
        return tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor
