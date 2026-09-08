# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.common import SpaceType
from isaaclab.utils import configclass

from lib.env.env_cfg import EnvCfg


@configclass
class ConstraintsEnvCfg(EnvCfg):
    """Configuration for an environment that also feeds frozen constraint networks.

    Extends :class:`EnvCfg` with one declared space. Environments used for constrained control
    have to serve two consumers every step: the learning agent, and one or more frozen networks
    (e.g. a safety value function) whose input is not the agent observation. Declaring that input
    as a space -- instead of pushing tensors through ``extras`` under hard-coded string keys --
    makes its dimension checkable and lets model factories query it without reaching into the
    unwrapped environment.
    """

    constraint_state_space: SpaceType = MISSING
    """Constraint-state space definition.

    Follows the same convention as :attr:`EnvCfg.observation_space`: either a Gymnasium space or
    a basic Python data type (an ``int`` for a flat vector). This is the input of the frozen
    networks, not of the learning agent.
    """
