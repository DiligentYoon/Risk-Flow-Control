# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.common import SpaceType
from isaaclab.utils import configclass

from lib.env.env_cfg import EnvCfg


@configclass
class SafeValueEnvCfg(EnvCfg):
    """Configuration for training safety value networks."""

    safety_state_space: SpaceType = MISSING
    """Safety-state space definition."""
