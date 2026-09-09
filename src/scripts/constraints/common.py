# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction shared by ``train.py`` and ``play.py``.

research.md 6 states the rule for the frozen ``V_N``: it is rebuilt with the very structure it was
trained under and restored through ``agent.load()``, with no wrapper class standing in between,
because a wrapper would let the training-time structure and the inference-time structure drift
apart. The same argument applies one level up. If ``play.py`` assembled the actor and the critic on
its own, "the checkpoint plays back what training produced" would rest on two code paths being kept
in step by hand -- and the failure is silent, since a mismatched network still loads and still
returns actions. They are built here once, and both scripts call the same functions.

Nothing here touches the simulator, so it imports cleanly before ``AppLauncher`` has run.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from lib.agent.reach_avoid import ReachAvoid
from lib.model.MLP import RA_Critic

from agents.risk_flow import RiskFlow
from buffer.risk_flow_buffer import RiskFlowBuffer
from models.risk_flow_models import DeterministicActor, MultiHorizonCritic


def resolve_horizon(cfg_value: Union[str, int, None], env: Any) -> int:
    """Resolve the recovery deadline H.

    ``H`` is a property of the environment (``max_episode_length - 1``, research.md 3.3), not a free
    hyperparameter, so the config either defers to it or has to agree with it.
    """
    horizon = int(env.max_episode_length) - 1
    if cfg_value == "auto" or cfg_value is None:
        return horizon
    if int(cfg_value) != horizon:
        raise ValueError(
            f"cfg horizon ({cfg_value}) disagrees with the environment "
            f"({env.max_episode_length} - 1 = {horizon}). H is the recovery deadline the episode "
            f"length defines; change `episode_length_s`, not this."
        )
    return int(cfg_value)


def build_predictor(
    cfg: Dict,
    constraint_state_dim: int,
    device: Union[str, torch.device],
    checkpoint: Optional[str] = None) -> nn.Module:
    """Restore the frozen ``V_N`` with the structure it was trained under.

    Args:
        cfg: The ``predictor.agent`` section, carrying at least ``seed``.
        constraint_state_dim: Dimension of the constraint state, i.e. of this network's input.
        device: Device the network lives on.
        checkpoint: Path to the Phase 0 checkpoint.
        allow_untrained: Return a randomly initialized network when no checkpoint was given. Only
            for smoke runs that exercise the plumbing -- every number a run reports is meaningless
            under it, which is why it has to be asked for explicitly.

    Returns:
        The frozen ``RA_Critic``. ``ReachAvoid.__init__`` leaves it in eval mode, so its own
        normalization statistics never move again.
    """
    model = {"critic": RA_Critic(constraint_state_dim, device)}
    agent = ReachAvoid(model, None, device=device, cfg=cfg)

    if checkpoint is not None:
        path = os.path.abspath(checkpoint)
        agent.load(path)
        print(f"[INFO] Frozen V_N restored from {path}")
    else:
        raise RuntimeError("No V_N checkpoint given.")

    return agent.critic


def build_models(
    models_cfg: Dict,
    observation_dim: int,
    constraint_state_dim: int,
    action_dim: int,
    horizon: int,
    device: Union[str, torch.device],
) -> Dict[str, nn.Module]:
    """Build the critic and the actor.

    The critic's input is the constraint state, the actor's is the policy observation; they are
    different channels of the same step and mixing them up produces a network that trains and
    evaluates without complaint, so the two dimensions are named rather than passed positionally by
    the callers.
    """
    critic_cfg = models_cfg.get("critic", {}) if models_cfg else {}
    actor_cfg = models_cfg.get("actor", {}) if models_cfg else {}

    return {
        "critic": MultiHorizonCritic(constraint_state_dim, action_dim, horizon, device=device,
                                     output_gain=critic_cfg.get("output_gain", 0.01)),
        "actor": DeterministicActor(observation_dim, action_dim, device=device,
                                    output_gain=actor_cfg.get("output_gain", 0.01)),
    }


def build_agent(
    agent_cfg: Dict,
    models: Dict[str, nn.Module],
    buffer: Optional[RiskFlowBuffer],
    value_critic: nn.Module,
    torque_model: Dict[str, Any],
    device: Union[str, torch.device],
    checkpoint: Optional[str] = None,
) -> RiskFlow:
    """Build the :class:`RiskFlow` agent and, if one was given, load a checkpoint into it.

    ``buffer`` is ``None`` for evaluation: the replay buffer is the largest allocation in the run
    and an evaluation never samples from it.
    """
    agent = RiskFlow(models, buffer, value_critic, torque_model, device=device, cfg=agent_cfg)

    if checkpoint is not None:
        path = os.path.abspath(checkpoint)
        agent.load(path)
        print(f"[INFO] RiskFlow restored from {path}")

    return agent
