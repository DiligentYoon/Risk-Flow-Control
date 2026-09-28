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
from lib.utils.space_utils import compute_space_size, spec_to_gym_space

from tasks.p02_safety_value.envs.safe_value_env import SafeValueEnv
from tasks.p03_safe_policy.envs.intervention_env_cfg import InterventionEnvCfg


class InterventionEnv(SafeValueEnv):
    """Base environment that serves the inputs of frozen safety value networks."""

    cfg: InterventionEnvCfg

    def __init__(self, cfg: InterventionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.intervention_obs_buf: torch.Tensor | None = None
        self.intervention_states_buf: torch.Tensor | None = None
        self.final_intervention_obs_buf: torch.Tensor | None = None
        self.final_intervention_state_buf: torch.Tensor | None = None

    """
    Properties
    """

    @property
    def num_intervention_states(self) -> int:
        return compute_space_size(self.cfg.intervention_state_space)
    
    @property
    def num_intervention_obs(self) -> int:
        return compute_space_size(self.cfg.intervention_observation_space)

    """
    Operations
    """

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, states, safety_state, extras = super().reset(seed=seed, options=options)

        self.intervention_obs_buf = self._get_intervention_observations()
        self.intervention_states_buf = self._get_intervention_states()

        if self.cfg.intervention_observation_noise_type:
            if self.cfg.intervention_observation_noise_type == "gaussian":
                self.intervention_obs_buf = gaussian_noise(self.intervention_obs_buf, **self.cfg.intervention_observation_noise_params)
            elif self.cfg.intervention_observation_noise_type == "uniform":
                self.intervention_obs_buf = uniform_noise(self.intervention_obs_buf, **self.cfg.intervention_observation_noise_params)
            elif self.cfg.intervention_observation_noise_type == "constant":
                self.intervention_obs_buf = constant_noise(self.intervention_obs_buf, **self.cfg.intervention_observation_noise_params)
            else:
                raise RuntimeError(f"Unknown observation noise type: {self.cfg.intervention_observation_noise_type}")

        extras = dict(self.extras)

        return obs, states, safety_state, self.intervention_obs_buf, self.intervention_states_buf, extras

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
        (
            final_safety_state_buf, 
            final_safety_value_buf, 
            final_push_event_buf, 
            final_intervention_obs_buf,
            final_intervention_states_buf,
        ) = self._capture_final_states()

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
    def _get_intervention_states(self) -> torch.Tensor:
        raise NotImplementedError(f"Please implement the '_get_intervention_states' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_intervention_observations(self) -> torch.Tensor:
        raise NotImplementedError(f"Please implement the '_get_intervention_observations' method for {self.__class__.__name__}.")

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()

        self.single_observation_space["intervention"] = spec_to_gym_space(self.cfg.intervention_observation_space)
        self.intervention_observation_space = gym.vector.utils.batch_space(self.single_observation_space["intervention"], self.num_envs)
        self.intervention_state_space = gym.vector.utils.batch_space(spec_to_gym_space(self.cfg.intervention_state_space), self.num_envs)
        self.intervention_action_space = gym.vector.utils.batch_space(spec_to_gym_space(self.cfg.intervention_action_space), self.num_envs)

    def _capture_final_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        final_safety_state, final_safety_value, final_push_event = super()._capture_final_states()

        final_intervention_obs = self._get_intervention_observations()
        final_intervention_states = self._get_intervention_states()

        if self.cfg.intervention_observation_noise_type:
            if self.cfg.intervention_observation_noise_type == "gaussian":
                final_intervention_obs = gaussian_noise(final_intervention_obs, **self.cfg.intervention_observation_noise_params)
            elif self.cfg.intervention_observation_noise_type == "uniform":
                final_intervention_obs = uniform_noise(final_intervention_obs, **self.cfg.intervention_observation_noise_params)
            elif self.cfg.intervention_observation_noise_type == "constant":
                final_intervention_obs = constant_noise(final_intervention_obs, **self.cfg.intervention_observation_noise_params)
            else:
                raise RuntimeError(f"Unknown observation noise type: {self.cfg.intervention_observation_noise_type}")

        return final_safety_state, final_safety_value, final_push_event, final_intervention_obs, final_intervention_states