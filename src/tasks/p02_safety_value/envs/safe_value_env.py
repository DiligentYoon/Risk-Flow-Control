# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Union

import gymnasium as gym
import torch

from lib.domain_randomizer.noise_model import constant_noise, gaussian_noise, uniform_noise
from lib.env.env import Env
from lib.utils.space_utils import compute_space_size, spec_to_gym_space

from .safe_value_env_cfg import SafeValueEnvCfg


class SafeValueEnv(Env):
    """Base environment that serves the inputs of safety value networks.

    Adds two things on top of :class:`Env`:

    1. **A declared safety-state channel.** Subclasses implement :meth:`_get_safety_states`
       and the tensor is returned as an explicit element of the step tuple, validated once against
       ``cfg.safety_state_space``. A safety value function consume this every step.

    2. **Pre-reset snapshots.** Isaac Lab resets terminated environments inside the same step, so
       by the time :meth:`Env.step` computes observations, the state that actually followed the
       last action is gone and has been replaced by the initial state of the next episode. 
       Any algorithm that differences consecutive states -- rather than merely masking terminals --
       reads a meaningless transition at exactly the most informative moment. This class captures
       the true next state in the one window where it still exists (after ``_get_dones`` and
       ``_get_rewards``, before ``_reset_idx``) and publishes it through
       ``extras["final_observations"]`` / ``extras["final_constraint_states"]``.
    """

    cfg: SafeValueEnvCfg

    def __init__(self, cfg: SafeValueEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Latest safety-network input, and the pre-reset snapshots of the true next state.
        # `_get_observations` / `_get_safety_states` build a fresh tensor on every call, so
        # holding references is enough -- copying into pre-allocated buffers would only add work.
        self.safety_state_buf: torch.Tensor | None = None
        self.push_event_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    """
    Properties
    """

    @property
    def num_safety_states(self) -> int:
        """Dimension of the safety-state vector for each environment instance."""
        return compute_space_size(self.cfg.safety_state_space)

    """
    Operations
    """

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Reset all environments and return observations along with the safety-state input.

        There is no preceding step at reset time, so the snapshots are set to the initial state.
        This keeps them readable unconditionally instead of being ``None`` until the first step.

        Returns:
            A tuple containing the observations, states, safety states and extras.
        """
        obs, states, extras = super().reset(seed=seed, options=options)

        self.safety_state_buf = self._get_safety_states()
        self.push_event_buf.zero_()

        extras = dict(self.extras)

        return obs, states, self.safety_state_buf, extras

    def step(
        self, action: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one time-step and additionally return the safety-network input.

        Mirrors :meth:`Env.step`, with the pre-reset capture inserted between ``_get_rewards`` and
        ``_reset_idx``. See the class docstring for why the body is duplicated instead of wrapped.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, states, safety states, rewards, resets
            (terminated and truncated) and extras. ``extras["final_observations"]`` and
            ``extras["final_constraint_states"]`` hold the pre-reset snapshots.
        """
        if isinstance(action, Dict):
            for k in action.keys():
                action[k] = action[k].to(self.device)
        else:
            action = action.to(self.device)
        action = self._apply_action_noise(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # update curriculum difficulty
        if self.curriculum_manager is not None:
            self.curriculum_manager.update(self.common_step_counter)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # capture the state that actually followed the action, before the same-step reset overwrites it for the terminated environments
        final_safety_state_buf, final_safety_value_buf, final_push_event_buf = self._capture_final_states()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # command update
        if self.cfg.commands is not None and hasattr(self, "commands"):
            self.commands.update()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        # note: no noise is applied to the state space nor to the safety states
        self.obs_buf = self._apply_observation_noise(self._get_observations())
        self.state_buf = self._get_states()
        self.safety_state_buf = self._get_safety_states()
        self.safety_value_buf = self._get_safety_values()

        # update viz data
        if self.cfg.viz_data is not None:
            self.extras = self._update_viz_data()

        # return observations, rewards, resets and extras with shallow copy
        extras = dict(self.extras)

        return (
            self.obs_buf,
            self.state_buf,
            self.safety_state_buf,
            self.safety_value_buf,
            final_safety_state_buf,
            final_safety_value_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            final_push_event_buf,
            extras,
        )

    """
    Implementation specific.
    """

    @abstractmethod
    def _get_safety_states(self) -> torch.Tensor:
        """Compute and return the input of the safety networks.

        Returns:
            The safety states for the environment. Shape is
            (num_envs, safety_state_space).
        """
        raise NotImplementedError(
            f"Please implement the '_get_safety_states' method for {self.__class__.__name__}."
        )

    @abstractmethod
    def _get_safety_values(self) -> torch.Tensor:
        """Compute and return the safety value.

        Returns:
            The safety values for the environment. Shape is
            (num_envs, 1).
        """
        raise NotImplementedError(
            f"Please implement the '_get_safety_values' method for {self.__class__.__name__}."
        )

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action, observation and safety-state spaces for the Gym environment."""
        super()._configure_gym_env_spaces()

        # the safety states are an environment output like the critic state, so they are
        # exposed the same way: an entry in the single-observation dict plus a batched space
        self.single_observation_space["safety"] = spec_to_gym_space(self.cfg.safety_state_space)
        self.safety_state_space = gym.vector.utils.batch_space(self.single_observation_space["safety"], self.num_envs)

    def _capture_final_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Snapshot the true next state before the same-step autoreset discards it.

        Called once per step, after the termination flags and rewards have been computed and before
        any environment is reset. At this point the simulator and the cached intermediate values
        still describe the state reached by the last action, for every environment.
        """
        final_safety_state_buf = self._get_safety_states()
        final_safety_value_buf = self._get_safety_values()
        final_push_event_buf   = self.push_event_buf.clone()

        # reset push event buf
        self.push_event_buf.zero_()

        return final_safety_state_buf, final_safety_value_buf, final_push_event_buf

    def _apply_action_noise(self, action: torch.Tensor) -> torch.Tensor:
        """Apply the configured noise model to the actions."""
        if not self.cfg.action_noise_type:
            return action
        if self.cfg.action_noise_type == "gaussian":
            return gaussian_noise(action, **self.cfg.action_noise_params)
        if self.cfg.action_noise_type == "uniform":
            return uniform_noise(action, **self.cfg.action_noise_params)
        if self.cfg.action_noise_type == "constant":
            return constant_noise(action, **self.cfg.action_noise_params)
        raise RuntimeError(f"Unknown action noise type: {self.cfg.action_noise_type}")

    def _apply_observation_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply the configured noise model to the observations."""
        if not self.cfg.observation_noise_type:
            return obs
        if self.cfg.observation_noise_type == "gaussian":
            return gaussian_noise(obs, **self.cfg.observation_noise_params)
        if self.cfg.observation_noise_type == "uniform":
            return uniform_noise(obs, **self.cfg.observation_noise_params)
        if self.cfg.observation_noise_type == "constant":
            return constant_noise(obs, **self.cfg.observation_noise_params)
        raise RuntimeError(f"Unknown observation noise type: {self.cfg.observation_noise_type}")
