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