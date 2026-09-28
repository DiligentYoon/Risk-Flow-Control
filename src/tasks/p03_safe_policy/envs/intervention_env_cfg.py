# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.common import SpaceType
from isaaclab.utils import configclass

from lib.env.env_cfg import EnvCfg

from tasks.p02_safety_value.envs.safe_value_env_cfg import SafeValueEnvCfg

@configclass
class InterventionEnvCfg(SafeValueEnvCfg):
    """Configuration for an environment that also feeds frozen safety value networks."""

    intervention_observation_space: SpaceType = MISSING
    """Observation space definition for intervention policy."""

    intervention_state_space: SpaceType = MISSING
    """State space definition for intervention policy."""

    intervention_action_space: SpaceType = MISSING
    """Action space definition for intervention policy."""

    intervention_observation_noise_type: str = None
    intervention_observation_noise_params: dict = None
