# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium
import torch

from typing import Any, Tuple, Union

from lib.utils.wrapper_utils import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space
from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo


class SafetyEnvWrapper(IsaacLabWrapper):
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

        self._safety_states = None
        self._final_observations = None
        self._final_safety_states = None

    @property
    def safety_state_space(self) -> Union[gymnasium.Space, None]:
        """Constraint-state space, i.e. the input of the frozen networks."""
        try:
            return self._unwrapped.single_observation_space["safety"]
        except KeyError:
            pass
        try:
            return self._unwrapped.safety_state_space
        except AttributeError:
            return None

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment."""
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, states, safety_states, final_observations, final_safety_states, reward, terminated, truncated, self._info = self._env.step(actions)

        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        self._safety_states = flatten_tensorized_space(tensorize_space(self.safety_state_space, safety_states))
        self._final_observations = flatten_tensorized_space(tensorize_space(self.observation_space, final_observations))
        self._final_safety_states = flatten_tensorized_space(tensorize_space(self.safety_state_space, final_safety_states))

        return (
            self._observations,
            self._states,
            self._safety_states,
            self._final_observations,
            self._final_safety_states,
            reward.reshape(-1, 1),
            terminated.reshape(-1, 1),
            truncated.reshape(-1, 1),
            self._info,
        )

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Reset the environment."""
        if self._reset_once:
            observations, states, safety_states, final_observations, final_safety_states, self._info = self._env.reset()

            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._safety_states = flatten_tensorized_space(tensorize_space(self.safety_state_space, safety_states))
            self._final_observations = flatten_tensorized_space(tensorize_space(self.observation_space, final_observations))
            self._final_safety_states = flatten_tensorized_space(tensorize_space(self.safety_state_space, final_safety_states))

            self._reset_once = False
        return self._observations, self._states, self._safety_states, self._final_observations, self._final_safety_states, self._info


class SafetyEnvRecordVideo(RecordVideo):
    """:class:`RecordVideo` for an environment that publishes a safety-state channel.

    The submodule's recorder unpacks the base environment's tuples by arity -- three from ``reset``,
    six from ``step`` -- so wrapping a :class:`SafetyEnv` raises before a single frame is
    captured. Only the two signatures differ; the recording itself is unchanged, so the frame
    capture is inherited rather than restated and stays in step with the submodule.

    It sits *inside* :class:`SafetyEnvWrapper`, between it and the raw environment, so it sees the
    unflattened tuples the environment produces.
    """

    def reset(self, *, seed=None, options=None):
        """Reset the environment and eventually start a new recording."""
        observations, states, safety_states, final_observations, final_safety_states, info = self.env.reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return observations, states, safety_states, final_observations, final_safety_states, info

    def step(self, action):
        """Step the environment, recording a frame while :attr:`recording` is set."""
        observations, states, safety_states, final_observations, final_safety_states, reward, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return observations, states, safety_states, final_observations, final_safety_states, reward, terminated, truncated, info
