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
    """Circular replay buffer holding exactly what the one-step TD target needs.

    Stored per transition::

        observations, final_observations            # actor input, and the bootstrap observation
        constraint_states, final_constraint_states  # V_N input, from which Delta_N is computed
        actions                                     # the executed action, exploration noise included
        terminated, truncated

    The ``final_*`` channels come from the pre-reset snapshots that :class:`ConstraintsEnv`
    publishes. They are the true state that followed the action, for terminal and non-terminal
    steps alike, so nothing here has to branch on the done flags.

    Deliberately *not* stored:

    * ``Delta_N`` -- precomputing it would tie the whole buffer to one particular ``V_N``, and
      swapping the frozen network in would invalidate every sample. ``V_N`` is a small MLP, so two
      forward passes per batch are cheaper than that coupling.
    * ``episode_id`` / step index -- only an N-step return would need them.

    Args:
        buffer_size: Number of rows. Total capacity is ``buffer_size * num_envs`` transitions.
        num_envs: Number of parallel environments.
        device: Device the tensors are allocated on.
    """

    def __init__(
        self,
        buffer_size: int = 1,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(buffer_size, num_envs, device)

    def init_buffer(
        self,
        observation_space: gymnasium.Space,
        constraint_state_space: gymnasium.Space,
        action_space: gymnasium.Space,
    ) -> None:
        """Allocate every tensor at once.

        ``Buffer.create_tensor`` fills *all* existing float tensors with NaN on each call, which is
        why the whole set is created here in one go and zeroed afterwards, rather than added
        lazily as the training loop discovers it needs them.

        Args:
            observation_space: Policy observation space.
            constraint_state_space: Constraint state space, i.e. the frozen networks' input.
            action_space: Action space.
        """
        self.create_tensor("observations", observation_space, dtype=torch.float32)
        self.create_tensor("final_observations", observation_space, dtype=torch.float32)
        self.create_tensor("constraint_states", constraint_state_space, dtype=torch.float32)
        self.create_tensor("final_constraint_states", constraint_state_space, dtype=torch.float32)
        self.create_tensor("actions", action_space, dtype=torch.float32)
        self.create_tensor("terminated", 1, dtype=torch.bool)
        self.create_tensor("truncated", 1, dtype=torch.bool)

        # undo the NaN fill of the last create_tensor call: unwritten rows are never sampled
        # (``__len__`` guards that), but a NaN that leaks into a reduction is hard to trace back
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.zero_()

    def add_samples(
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

        Args:
            observations: ``o_t``, shape (num_envs, observation_dim).
            constraint_states: ``s_t``, shape (num_envs, constraint_state_dim).
            actions: Executed action ``a_t`` including exploration noise, shape (num_envs, action_dim).
            final_observations: ``o_{t+1}`` before the autoreset, shape (num_envs, observation_dim).
            final_constraint_states: ``s_{t+1}`` before the autoreset, shape (num_envs, constraint_state_dim).
            terminated: Termination flags, shape (num_envs,) or (num_envs, 1).
            truncated: Truncation flags, shape (num_envs,) or (num_envs, 1).
        """
        super().add_samples(
            observations=observations,
            constraint_states=constraint_states,
            actions=actions,
            final_observations=final_observations,
            final_constraint_states=final_constraint_states,
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
            size=(min(batch_size, size),),
            dtype=torch.long,
            device=self.device,
        )

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=1)[0]

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by the stored tensors.

        The simulator and the models share the GPU with this buffer, so its footprint is a budget
        item rather than an afterthought.
        """
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors.values())

    @staticmethod
    def _ensure_2d_column(tensor: torch.Tensor) -> torch.Tensor:
        """Give a flag tensor the (num_envs, 1) shape the storage expects."""
        return tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor
