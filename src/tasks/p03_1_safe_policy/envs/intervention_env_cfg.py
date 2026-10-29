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
    
