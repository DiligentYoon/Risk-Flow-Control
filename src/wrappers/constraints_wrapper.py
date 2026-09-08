# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium
import torch

from typing import Any, Tuple, Union

from lib.utils.wrapper_utils import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space
from wrapper.isaaclab_wrapper import IsaacLabWrapper


class ConstraintsWrapper(IsaacLabWrapper):
    """Wrapper for :class:`ConstraintsEnv`.

    Relays the constraint-network input as an explicit element of the step tuple, and exposes its
    space so that models can be built without reaching into the unwrapped environment. The
    pre-reset snapshots stay in ``info`` -- they matter only on terminal steps -- but are put
    through the same tensorize/flatten path as the channels they mirror.

    Note:
        This package is named ``wrappers`` (plural) on purpose: the submodule already owns the
        top-level ``wrapper`` package, and a second one would shadow it.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)

        self._constraint_states = None

    @property
    def constraint_state_space(self) -> Union[gymnasium.Space, None]:
        """Constraint-state space, i.e. the input of the frozen networks."""
        try:
            return self._unwrapped.single_observation_space["constraints"]
        except KeyError:
            pass
        try:
            return self._unwrapped.constraint_state_space
        except AttributeError:
            return None

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        Args:
            actions: The actions to perform.

        Returns:
            Observation, state, constraint state, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, states, constraint_states, reward, terminated, truncated, self._info = self._env.step(actions)

        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        self._constraint_states = flatten_tensorized_space(tensorize_space(self.constraint_state_space, constraint_states))
        self._flatten_final_channels()

        return (
            self._observations,
            self._states,
            self._constraint_states,
            reward.reshape(-1, 1),
            terminated.reshape(-1, 1),
            truncated.reshape(-1, 1),
            self._info,
        )

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Reset the environment.

        Returns:
            Observation, state, constraint state, info.
        """
        if self._reset_once:
            observations, states, constraint_states, self._info = self._env.reset()

            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._constraint_states = flatten_tensorized_space(tensorize_space(self.constraint_state_space, constraint_states))
            self._flatten_final_channels()

            self._reset_once = False
        return self._observations, self._states, self._constraint_states, self._info

    def _flatten_final_channels(self) -> None:
        """Put the pre-reset snapshots through the same path as the channels they mirror."""
        if "final_observations" in self._info:
            self._info["final_observations"] = flatten_tensorized_space(tensorize_space(self.observation_space, self._info["final_observations"]))
        if "final_constraint_states" in self._info:
            self._info["final_constraint_states"] = flatten_tensorized_space(tensorize_space(self.constraint_state_space, self._info["final_constraint_states"]))
