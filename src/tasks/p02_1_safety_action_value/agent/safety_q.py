# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent
from tasks.p02_1_safety_action_value.buffer.replay_buffer import ReplayBuffer
from tasks.p02_1_safety_action_value.utils.scheduler import StepScheduler


class SafetyQ(Agent):
    def __init__(
        self,
        model: Dict[str, nn.Module],
        buffer: Optional[ReplayBuffer],
        device: Union[str, torch.device],
        cfg: Dict,
    ) -> None:
        super().__init__(cfg, model, device)
        self.actor = self.model["actor"].to(self.device)
        self.critic = self.model["critic"].to(self.device)
        self.buffer = buffer

        self.batch_size = self.cfg["batch_size"]
        self.learning_starts = self.cfg["learning_starts"]
        self.discount_factor = self.cfg["discount_factor"]
        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.target_tau = self.cfg["tau"]
        self.update_period = self.cfg.get("update_period", 1)
        self.target_entropy = -self.actor.num_actions
        self.update_counter = 0

        gamma_cfg = self.cfg.get("gamma_schedule", {})
        self.gamma_scheduler = None
        if gamma_cfg.get("enabled", False):
            self.gamma_scheduler = StepScheduler(init_value=self.discount_factor,
                                                 period=gamma_cfg["period"],
                                                 decay=gamma_cfg["decay"],
                                                 end_value=gamma_cfg["end_value"],
                                                 goal_value=gamma_cfg.get("goal_value", 1.0))

        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.requires_grad_(False)
        self.target_critic.eval()

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg["critic_learning_rate"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg["actor_learning_rate"])
        self.log_alpha = nn.ParameterDict({"value": nn.Parameter(torch.tensor(float(self.cfg["alpha"]), device=self.device).log(), requires_grad=True)})
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=self.cfg["alpha_learning_rate"])

        self.checkpoint_modules.update({
            "actor": self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "critic_optimizer": self.critic_optimizer,
            "actor_optimizer": self.actor_optimizer,
            "log_alpha": self.log_alpha,
            "alpha_optimizer": self.alpha_optimizer,
        })
        if self.gamma_scheduler is not None:
            self.checkpoint_modules["gamma_scheduler"] = self.gamma_scheduler

        self.tensors_names = (
            "observations",
            "states",
            "next_observations",
            "next_states",
            "next_safety_values",
            "actions",
            "terminated",
            "truncated",
        )
        self.set_running_mode("eval")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha["value"].exp()

    @torch.no_grad()
    def act(self, observations: torch.Tensor, deterministic: bool = False, update_rms: bool = False) -> torch.Tensor:
        actions, _ = self.actor(observations, deterministic=deterministic, update_rms=update_rms)
        return actions

    @torch.no_grad()
    def predict(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.critic(states, actions, update_rms=False)
        return torch.maximum(q1, q2)

    def insert_data(self, **samples: torch.Tensor) -> None:
        if self.buffer is None:
            raise RuntimeError("Replay buffer is not initialized.")
        self.buffer.add_samples(**samples)

    @torch.no_grad()
    def compute_target(
        self,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        next_safety_values: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        next_actions, _ = self.actor(next_observations, deterministic=False, update_rms=False)
        next_q1, next_q2 = self.target_critic(next_states, next_actions, update_rms=False)
        next_q = torch.maximum(next_q1, next_q2)
        target = (1.0 - self.discount_factor) * next_safety_values + self.discount_factor * torch.maximum(next_safety_values, next_q)
        return torch.where(terminated | truncated, next_safety_values, target)

    @torch.no_grad()
    def step_gamma(self, steps: int = 1) -> float:
        if self.gamma_scheduler is not None:
            self.discount_factor = self.gamma_scheduler.step(steps)
        return self.discount_factor

    @torch.no_grad()
    def update_target(self) -> None:
        for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_parameter.mul_(1.0 - self.target_tau).add_(parameter, alpha=self.target_tau)
        for target_buffer, buffer in zip(self.target_critic.buffers(), self.critic.buffers()):
            target_buffer.copy_(buffer)

    def update(self) -> Optional[Dict[str, Any]]:
        if self.buffer is None or len(self.buffer) < self.learning_starts * self.buffer.num_envs:
            return None

        self.set_running_mode("train")
        (
            observations, 
            states,
            next_observations, 
            next_states,
            next_safety_values, 
            actions, 
            terminated, truncated
        ) = self.buffer.sample_batch(self.tensors_names, self.batch_size)

        target = self.compute_target(next_observations, next_states, next_safety_values, terminated, truncated)
        q1, q2 = self.critic(states, actions, update_rms=True)
        loss_q1 = F.mse_loss(q1, target)
        loss_q2 = F.mse_loss(q2, target)
        critic_loss = loss_q1 + loss_q2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()

        actor_loss = torch.zeros((), device=self.device)
        entropy_loss = torch.zeros((), device=self.device)
        alpha_loss = torch.zeros((), device=self.device)
        if self.update_counter % self.update_period == 0:
            actor_loss, entropy_loss, alpha_loss = self._update_actor(observations, states)
            self.update_target()

        self.update_counter += 1
        self.set_running_mode("eval")

        return {
            "critic_loss": critic_loss.item(),
            "critic_loss_1": loss_q1.item(),
            "critic_loss_2": loss_q2.item(),
            "actor_loss": actor_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.detach().item(),
            "gamma": self.discount_factor,
            "q_mean": torch.maximum(q1, q2).detach().mean().item(),
            "target_mean": target.mean().item(),
        }

    def _update_actor(self, observations: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_prob = self.actor(observations, deterministic=False, update_rms=True)

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)

        q1, q2 = self.critic(states, actions, update_rms=False)
        q = torch.maximum(q1, q2)
        entropy_loss = log_prob.mean()
        actor_loss = q.mean() + self.alpha * entropy_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        return actor_loss.detach(), entropy_loss.detach(), alpha_loss.detach()