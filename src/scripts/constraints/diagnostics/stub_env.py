# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A simulator-free :class:`ConstraintsEnv` used by the P1 diagnostics.

The state is a per-environment step counter that is also returned as the observation and the
constraint state, which makes the expected values readable by eye: on a terminal step the snapshot
must still hold the counter of the episode that just ended, while the returned tensors must already
hold the counter of the new one.

``Env.__init__`` is deliberately not called -- it would build a simulation context and a scene, and
neither matters to the ordering under test. Import this module only after the Isaac Sim app has been
launched, since ``lib.env.env`` imports ``omni`` at module level.
"""

import gymnasium as gym
import numpy as np
import torch

from envs.constraints_env import ConstraintsEnv


class _StubSim:
    """Minimal stand-in for the simulation context."""

    def has_gui(self) -> bool:
        return False

    def has_rtx_sensors(self) -> bool:
        return False

    def step(self, render: bool = False):
        pass

    def render(self):
        pass


class _StubScene:
    """Minimal stand-in for the interactive scene."""

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        pass


class _StubSimCfg:
    dt = 0.005
    render_interval = 1


class StubCfg:
    """Minimal stand-in for :class:`ConstraintsEnvCfg`."""

    sim = _StubSimCfg()
    decimation = 1
    observation_space = 1
    state_space = 1
    action_space = 1
    constraint_state_space = 1
    num_rerenders_on_reset = 0
    wait_for_textures = False
    viz_data = None
    events = None
    commands = None
    action_noise_type = None
    action_noise_params = None
    observation_noise_type = None
    observation_noise_params = None


def _box(dim: int) -> gym.spaces.Box:
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


class StubConstraintsEnv(ConstraintsEnv):
    """A ConstraintsEnv whose state is a step counter."""

    def __init__(self, num_envs: int, episode_length: int, device: str = "cpu"):
        self.cfg = StubCfg()
        self.sim = _StubSim()
        self.scene = _StubScene()

        self._num_envs = num_envs
        self._device = device
        self.episode_length = episode_length

        # the "state": how many steps the current episode has run, per environment
        self.counter = torch.zeros(num_envs, 1, device=device)

        # buffers normally allocated by Env.__init__
        self._sim_step_counter = 0
        self.common_step_counter = 0
        self.curriculum_manager = None
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros_like(self.reset_terminated)
        self.extras = {}

        # spaces normally built by Env._configure_gym_env_spaces, needed by the wrapper
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": _box(self.cfg.observation_space),
                "critic": _box(self.cfg.state_space),
                "constraints": _box(self.cfg.constraint_state_space),
            }
        )
        self.single_action_space = _box(self.cfg.action_space)

        # Env.__del__ calls close(), which reads a flag set by Env.__init__
        self._is_closed = True

        # buffers allocated by ConstraintsEnv.__init__
        self.constraint_state_buf = None
        self.final_obs_buf = None
        self.final_constraint_state_buf = None
        self._constraint_states_validated = False

    @property
    def unwrapped(self):
        return self

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self):
        return self._device

    def close(self):
        pass

    def _pre_physics_step(self, action):
        pass

    def _apply_action(self):
        # advancing the counter here mirrors a physics step changing the state
        self.counter += 1.0

    def _get_observations(self):
        return self.counter.clone()

    def _get_states(self):
        return self.counter.clone()

    def _get_constraint_states(self):
        return self.counter.clone()

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        # environment 0 terminates (absorbing), the others time out, both at the same period
        at_period = self.counter.squeeze(-1) >= self.episode_length
        terminated = at_period.clone()
        truncated = at_period.clone()
        terminated[1:] = False
        truncated[0] = False
        return terminated, truncated

    def _reset_idx(self, env_ids):
        self.counter[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
