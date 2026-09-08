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

from envs.constraints_env_cfg import ConstraintsEnvCfg


class ConstraintsEnv(Env):
    """Base environment that serves the inputs of frozen constraint networks.

    Adds two things on top of :class:`Env`:

    1. **A declared constraint-state channel.** Subclasses implement :meth:`_get_constraint_states`
       and the tensor is returned as an explicit element of the step tuple, validated once against
       ``cfg.constraint_state_space``. Frozen networks (e.g. a safety value function) consume this
       every step, so it does not belong in ``extras`` behind a string key.

    2. **Pre-reset snapshots.** Isaac Lab resets terminated environments inside the same step, so
       by the time :meth:`Env.step` computes observations, the state that actually followed the
       last action is gone and has been replaced by the initial state of the next episode. Any
       algorithm that differences consecutive states -- rather than merely masking terminals --
       reads a meaningless transition at exactly the most informative moment. This class captures
       the true next state in the one window where it still exists (after ``_get_dones`` and
       ``_get_rewards``, before ``_reset_idx``) and publishes it through
       ``extras["final_observations"]`` / ``extras["final_constraint_states"]``.

    The snapshots are *always* populated, terminal step or not, so consumers never branch on the
    termination flags to decide which tensor to read.

    Note:
        :meth:`step` re-implements :meth:`Env.step` rather than delegating to it. The capture point
        sits in the middle of that method, and a subclass hook is not usable here: the concrete task
        environment overrides ``_get_dones``/``_get_rewards``/``_reset_idx`` itself, which would
        shadow any hook installed on those names further down the inheritance chain.

    Note:
        Flat tensor observations are assumed (single-agent). The dictionary observation layout used
        by the multi-agent environments is not supported.
    """

    cfg: ConstraintsEnvCfg

    def __init__(self, cfg: ConstraintsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Latest constraint-network input, and the pre-reset snapshots of the true next state.
        # `_get_observations` / `_get_constraint_states` build a fresh tensor on every call, so
        # holding references is enough -- copying into pre-allocated buffers would only add work.
        self.constraint_state_buf: torch.Tensor | None = None
        self.final_obs_buf: torch.Tensor | None = None
        self.final_constraint_state_buf: torch.Tensor | None = None

        # The constraint-state dimension is validated on the first tensor produced, not on every
        # step. A mismatch is a wiring bug, and it never appears halfway through a run.
        self._constraint_states_validated = False

    """
    Properties
    """

    @property
    def num_constraint_states(self) -> int:
        """Dimension of the constraint-state vector for each environment instance."""
        return compute_space_size(self.cfg.constraint_state_space)

    """
    Operations
    """

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict]:
        """Reset all environments and return observations along with the constraint-state input.

        There is no preceding step at reset time, so the snapshots are set to the initial state.
        This keeps them readable unconditionally instead of being ``None`` until the first step.

        Returns:
            A tuple containing the observations, states, constraint states and extras.
        """
        obs, states, extras = super().reset(seed=seed, options=options)

        self.constraint_state_buf = self._get_constraint_states()
        self.final_obs_buf = obs
        self.final_constraint_state_buf = self.constraint_state_buf

        extras = dict(extras)
        extras["final_observations"] = self.final_obs_buf
        extras["final_constraint_states"] = self.final_constraint_state_buf

        return obs, states, self.constraint_state_buf, extras

    def step(
        self, action: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one time-step and additionally return the constraint-network input.

        Mirrors :meth:`Env.step`, with the pre-reset capture inserted between ``_get_rewards`` and
        ``_reset_idx``. See the class docstring for why the body is duplicated instead of wrapped.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, states, constraint states, rewards, resets
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

        # -- capture the state that actually followed the action, before the same-step autoreset
        #    below overwrites it for the terminated environments
        self._capture_final_states()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.cfg.commands is not None and hasattr(self, "commands"):
            self.commands.update()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        # note: no noise is applied to the state space (it is used for critic networks) nor to the
        #       constraint states (they are the input of frozen networks trained without it)
        self.obs_buf = self._apply_observation_noise(self._get_observations())
        self.state_buf = self._get_states()
        self.constraint_state_buf = self._get_constraint_states()

        # update viz data
        if self.cfg.viz_data is not None:
            self.extras = self._update_viz_data()

        # return observations, rewards, resets and extras with shallow copy
        extras = dict(self.extras)
        extras["final_observations"] = self.final_obs_buf
        extras["final_constraint_states"] = self.final_constraint_state_buf

        return (
            self.obs_buf,
            self.state_buf,
            self.constraint_state_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            extras,
        )

    """
    Implementation specific.
    """

    @abstractmethod
    def _get_constraint_states(self) -> torch.Tensor:
        """Compute and return the input of the frozen constraint networks.

        Returns:
            The constraint states for the environment. Shape is
            (num_envs, constraint_state_space).
        """
        raise NotImplementedError(
            f"Please implement the '_get_constraint_states' method for {self.__class__.__name__}."
        )

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action, observation and constraint-state spaces for the Gym environment."""
        super()._configure_gym_env_spaces()

        # the constraint states are an environment output like the critic state, so they are
        # exposed the same way: an entry in the single-observation dict plus a batched space
        self.single_observation_space["constraints"] = spec_to_gym_space(self.cfg.constraint_state_space)
        self.constraint_state_space = gym.vector.utils.batch_space(self.single_observation_space["constraints"], self.num_envs)

    def _capture_final_states(self):
        """Snapshot the true next state before the same-step autoreset discards it.

        Called once per step, after the termination flags and rewards have been computed and before
        any environment is reset. At this point the simulator and the cached intermediate values
        still describe the state reached by the last action, for every environment.
        """
        self.final_obs_buf = self._apply_observation_noise(self._get_observations())
        self.final_constraint_state_buf = self._get_constraint_states()

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
