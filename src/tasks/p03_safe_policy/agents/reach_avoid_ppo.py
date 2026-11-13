# Copyright (c) 2026, AISL.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import itertools
from typing import Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent
from lib.utils.Running_mean_std import RunningMeanStd
from lib.utils.Learning_rate_scheduler import KLAdaptiveLR

from ..buffer.reach_avoid_buffer import ReachAvoidBuffer


class ReachAvoidPPO(Agent):
    def __init__(self, model: Dict[str, nn.Module], buffer: ReachAvoidBuffer, device: Union[str, torch.device], cfg: Dict) -> None:
        super().__init__(cfg, model, device)
        # modules
        self.buffer = buffer
        self.actor = self.model["actor"].to(self.device)
        self.value_critic = self.model["value_critic"].to(self.device)
        self.ra_critic = self.model["ra_critic"].to(self.device)

        # standardizers
        self.value_standardizer = RunningMeanStd(shape=1, device=device)
        self.ra_standardizer = RunningMeanStd(shape=1, device=device)
        
        # hyperparameters (common)
        self.learning_epochs = self.cfg["learning_epochs"]
        self.mini_batches = self.cfg["mini_batches"]
        self.gae_lambda = self.cfg["lambda"]
        self.learning_rate = self.cfg["learning_rate"]
        self.learning_rate_scheduler = self.cfg["learning_rate_scheduler"]
        self.kl_threshold = self.cfg["kl_threshold"]
        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.ratio_clip = self.cfg["ratio_clip"]
        self.entropy_loss_scale = self.cfg["entropy_loss_scale"]
        # hyperparameters (value critic)
        self.value_discount_factor = self.cfg["value_discount_factor"]
        self.time_limit_bootstrap  = self.cfg["time_limit_bootstrap"]
        self.value_loss_scale = self.cfg["value_loss_scale"]
        # hyperparameters (reach-avoid critic)
        self.ra_discount_factor = self.cfg["ra_discount_factor"]
        self.ra_loss_scale = self.cfg["ra_loss_scale"]
        self.M = float(self.ra_critic.M)

        # optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.actor.parameters(), 
                                                            self.value_critic.parameters(),
                                                            self.ra_critic.parameters()), lr=self.learning_rate)

        # learning rate scheduler
        if self.learning_rate_scheduler is not None:
                self.learning_rate_scheduler = KLAdaptiveLR(self.optimizer, self.kl_threshold)

        # checkpoints
        self.checkpoint_modules["actor"] = self.actor
        self.checkpoint_modules["value_critic"] = self.value_critic
        self.checkpoint_modules["value_standardizer"] = self.value_standardizer
        self.checkpoint_modules["ra_critic"] = self.ra_critic
        self.checkpoint_modules["ra_standardizer"] = self.ra_standardizer
        self.checkpoint_modules["optimizer"] = self.optimizer

        # buffer interface
        self.tensors_names = ["observations", "next_observations",
                              "states", "next_states",
                              "actions", "action_log_probs",
                              "value_preds", "rewards",
                              "g_values", "l_values", 
                              "value_returns", "ra_returns", 
                              "value_advantages", "ra_advantages",
                              "terminated", "truncated"]

        self.tensors_names_for_updates = ["observations", "states", "next_states",
                                          "actions", "action_log_probs",
                                          "g_values", "l_values",
                                          "value_returns", "ra_returns", 
                                          "value_advantages", "ra_advantages",]
        
        self.set_running_mode("eval")

    def act(self, observations: torch.Tensor, infos: dict[str, torch.Tensor], timestep: int, deterministic: bool = False, update_rms: bool = False) -> torch.Tensor:
        # Normal act
        raw_actions, log_prob, entropy = self.actor(observations=observations,
                                                    taken_actions=None,
                                                    deterministic=deterministic,
                                                    update_rms=update_rms)

        return raw_actions, log_prob, entropy                       # Add raw action which is inserted into buffer

    def insert_data(self,
                    observations: torch.Tensor,
                    states: Union[torch.Tensor | None],
                    actions: torch.Tensor,
                    action_log_probs: torch.Tensor,
                    rewards: torch.Tensor,
                    g_values: torch.Tensor,
                    l_values: torch.Tensor,
                    next_observations: torch.Tensor,
                    next_states: Union[torch.Tensor | None],
                    truncated: torch.Tensor,
                    terminated: torch.Tensor,
                    infos: Any) -> None:
        """Record an environment transition in buffer."""
        buffer_rewards = rewards.clone()

        with torch.no_grad():
            value_preds, _, _ = self.value_critic(states)
            ra_preds, _, _ = self.ra_critic(states)
            value_preds = self.value_standardizer.destandardize(value_preds)
            ra_preds = self.ra_standardizer.destandardize(ra_preds)

        # time-limit (truncation) bootstrapping
        if self.time_limit_bootstrap:
            buffer_rewards += self.value_discount_factor * value_preds * truncated

        self.buffer.add_samples(observations=observations,
                                states=states,
                                actions=actions,
                                rewards=buffer_rewards,
                                g_values=g_values,
                                l_values=l_values,
                                next_observations=next_observations,
                                next_states=next_states,
                                truncated=truncated,
                                terminated=terminated,
                                action_log_probs=action_log_probs,
                                value_preds=value_preds,
                                ra_preds=ra_preds)

    def update(self) -> torch.Tensor:
        with torch.no_grad():
            # Last value for reward-based GAE bootstrap
            critic_input = self.buffer.get_tensor_by_name("next_states")[-1]
            last_values, _, _ = self.value_critic(critic_input)
            last_values = self.value_standardizer.destandardize(last_values)

            # RA value for every actual next state
            next_states = self.buffer.get_tensor_by_name("next_states")
            next_states_flat = next_states.reshape(-1, next_states.shape[-1])
            next_ra_preds, _, _ = self.ra_critic(next_states_flat)
            next_ra_preds = self.ra_standardizer.destandardize(next_ra_preds)
            next_ra_preds = next_ra_preds.reshape(self.buffer.buffer_size, -1, 1)
        
        # GAE calculation (value critic)
        self.buffer.compute_value_gae(last_values, self.value_discount_factor, self.gae_lambda)
        # standardization
        value_preds = self.value_standardizer.standardize(self.buffer.get_tensor_by_name("value_preds").reshape(-1, 1), update=True)
        returns = self.value_standardizer.standardize(self.buffer.get_tensor_by_name("value_returns").reshape(-1, 1), update=True)
        self.buffer.set_tensor_by_name("value_preds", value_preds.reshape(self.buffer.buffer_size, -1, 1))
        self.buffer.set_tensor_by_name("value_returns", returns.reshape(self.buffer.buffer_size, -1, 1))

        # GAE calculation (ra critic)
        self.buffer.compute_ra_gae(next_ra_preds, self.ra_discount_factor, self.gae_lambda)
        # standardization
        ra_preds = self.ra_standardizer.standardize(self.buffer.get_tensor_by_name("ra_preds").reshape(-1, 1), update=True)
        ra_returns = self.ra_standardizer.standardize(self.buffer.get_tensor_by_name("ra_returns").reshape(-1, 1), update=True)
        self.buffer.set_tensor_by_name("ra_preds", ra_preds.reshape(self.buffer.buffer_size, -1, 1))
        self.buffer.set_tensor_by_name("ra_returns", ra_returns.reshape(self.buffer.buffer_size, -1, 1))
        
        # Loss initialization
        cumulative_reward_policy_loss = 0
        cumulative_ra_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_ra_loss = 0

        # Grad initialization
        cumulative_grad_cosine = 0
        cumulative_grad_conflict = 0
        minimum_grad_cosine = 0
        maximum_grad_cosine = 0

        # Parameter Update
        kl_divergences = []
        self.set_running_mode("train")

        # Sample mini batch
        mini_batches = self.buffer.sample(names=self.tensors_names_for_updates, mini_batch=self.mini_batches) 
           
        for epoch in range(self.learning_epochs):
            for batch in mini_batches:
                (
                    sampled_observations,
                    sampled_states,
                    sampled_next_states,
                    sampled_actions,
                    sampled_action_log_probs,
                    sampled_g,
                    sampled_l,
                    sampled_value_returns,
                    sampled_ra_returns,
                    sampled_value_advantages,
                    sampled_ra_advantages,
                ) = batch

                _, new_log_probs, dist_entropy = self.actor(observations=sampled_observations, 
                                                taken_actions=sampled_actions,
                                                deterministic=False,
                                                update_rms=not epoch)

                # shape syncronization
                if len(new_log_probs.shape) != len(sampled_action_log_probs.shape):
                    new_log_probs = new_log_probs.reshape(sampled_action_log_probs.shape)

                # policy loss 
                # NOTE: Temporarily it is computed only by the value critic advantages.
                with torch.no_grad():
                    ratio = new_log_probs - sampled_action_log_probs
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)
                
                # entropy loss
                if self.entropy_loss_scale:
                    entropy_loss = -self.entropy_loss_scale * dist_entropy
                else:
                    entropy_loss = 0

                # policy loss (reward-driven)
                ratio = torch.exp(new_log_probs - sampled_action_log_probs)
                surrogate = sampled_value_advantages * ratio
                surrogate_clipped = sampled_value_advantages * torch.clip(ratio, 
                                                                          1.0 - self.ratio_clip, 
                                                                          1.0 + self.ratio_clip)
                reward_policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # policy loss (ra-driven)
                ra_surrogate = sampled_ra_advantages * ratio
                ra_surrogate_clipped = sampled_ra_advantages * torch.clip(ratio, 
                                                                          1.0 - self.ratio_clip,
                                                                          1.0 + self.ratio_clip)
                ra_policy_loss = -torch.min(ra_surrogate, ra_surrogate_clipped).mean()

                policy_loss = self.value_loss_scale * reward_policy_loss + self.ra_loss_scale * ra_policy_loss

                # reach-avoid critic loss
                continuation, _, _ = self.ra_critic(sampled_states, update_rms=not epoch)
                ra_loss = F.mse_loss(sampled_ra_returns, continuation)

                # value critic loss
                predicted_values, _, _ = self.value_critic(sampled_states, update_rms=not epoch)
                value_loss = F.mse_loss(sampled_value_returns, predicted_values)

                # Update cumulative losses
                cumulative_reward_policy_loss += reward_policy_loss.item()
                cumulative_ra_policy_loss += ra_policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_ra_loss += ra_loss.item()
                if self.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

                # Update cumulative gradient conflict
                actor_params = [p for p in self.actor.parameters() if p.requires_grad]
                grad_reward = torch.autograd.grad(reward_policy_loss, actor_params, retain_graph=True, allow_unused=True)
                grad_ra = torch.autograd.grad(ra_policy_loss, actor_params, retain_graph=True, allow_unused=True)

                dot = torch.zeros((), device=self.actor.device)
                norm_r = torch.zeros((), device=self.actor.device)
                norm_ra = torch.zeros((), device=self.actor.device)
                for gr, gra in zip(grad_reward, grad_ra):
                    norm_r += gr.detach().square().sum()
                    norm_ra += gra.detach().square().sum()
                    dot += (gr.detach() * gra.detach()).sum()
                grad_cosine = dot / (torch.sqrt(norm_r) * torch.sqrt(norm_ra) + 1e-12)
                grad_conflict = (dot < 0).float()

                cumulative_grad_cosine += grad_cosine.item()
                cumulative_grad_conflict += grad_conflict.item()
                minimum_grad_cosine = torch.minimum(minimum_grad_cosine, grad_cosine)
                maximum_grad_cosine = torch.maximum(maximum_grad_cosine, grad_cosine)

                # Optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss + ra_loss).backward()

                if self.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(itertools.chain(self.actor.parameters(), 
                                                             self.value_critic.parameters(), 
                                                             self.ra_critic.parameters()), self.grad_norm_clip)
                
                self.optimizer.step()

        # Learning rate scheduler update
        kl = torch.tensor(kl_divergences, device=self.device).mean().item()
        if self.learning_rate_scheduler is not None:
            self.learning_rate_scheduler.step(kl)
            learning_rate = self.learning_rate_scheduler.get_last_lr()[0]
        else:
            learning_rate = self.optimizer.param_groups[0]["lr"]

        self.set_running_mode("eval")

        mean_reward_policy_loss = cumulative_reward_policy_loss / (self.learning_epochs * self.mini_batches)
        mean_ra_policy_loss = cumulative_ra_policy_loss / (self.learning_epochs * self.mini_batches)
        mean_value_loss = cumulative_value_loss / (self.learning_epochs * self.mini_batches)
        mean_ra_loss = cumulative_ra_loss / (self.learning_epochs * self.mini_batches)
        mean_entropy_loss = cumulative_entropy_loss / (self.learning_epochs * self.mini_batches)
        mean_kl_divergence = kl

        mean_grad_cosine = cumulative_grad_cosine / (self.learning_epochs * self.mini_batches)
        grad_conflict_rate = cumulative_grad_conflict / (self.learning_epochs * self.mini_batches * self.num_agents)
        
        return {
            "reward_policy_loss": mean_reward_policy_loss, 
            "ra_policy_loss": mean_ra_policy_loss, 
            "value_loss": mean_value_loss, 
            "ra_loss": mean_ra_loss, 
            "entropy_loss": mean_entropy_loss,

            "grad_cosine": mean_grad_cosine,
            "max_grad_cosine": maximum_grad_cosine,
            "min_grad_cosine": minimum_grad_cosine,
            "grad_conflict_rate": grad_conflict_rate,

            "approx_kl": mean_kl_divergence,
            "learning_rate": learning_rate,
        }
