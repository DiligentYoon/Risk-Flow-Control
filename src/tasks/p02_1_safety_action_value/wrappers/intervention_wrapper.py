# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Tuple

import torch

from lib.utils.wrapper_utils import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space
from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo


class InterventionEnvWrapper(IsaacLabWrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._final_observations = None
        self._final_states = None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        actions = unflatten_tensorized_space(self.action_space, actions)
        (
            observations,
            states,
            safety_values,
            final_observations,
            final_states,
            final_safety_values,
            reward,
            terminated,
            truncated,
            self._info,
        ) = self._env.step(actions)

        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        self._final_observations = flatten_tensorized_space(tensorize_space(self.observation_space, final_observations))
        if final_states is not None:
            self._final_states = flatten_tensorized_space(tensorize_space(self.state_space, final_states))

        return (
            self._observations,
            self._states,
            safety_values.reshape(-1, 1),
            self._final_observations,
            self._final_states,
            final_safety_values.reshape(-1, 1),
            reward.reshape(-1, 1),
            terminated.reshape(-1, 1),
            truncated.reshape(-1, 1),
            self._info,
        )

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        if self._reset_once:
            observations, states, safety_values, self._info = self._env.reset()
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._safety_values = safety_values.reshape(-1, 1)
            self._reset_once = False
        return self._observations, self._states, self._safety_values, self._info


class InterventionEnvRecordVideo(RecordVideo):
    def reset(self, *, seed=None, options=None):
        observations, states, safety_values, info = self.env.reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()
        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return observations, states, safety_values, info

    def step(self, action):
        (
            observations,
            states,
            safety_values,
            final_observations,
            final_states,
            final_safety_values,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return observations, states, safety_values, final_observations, final_states, final_safety_values, reward, terminated, truncated, info
