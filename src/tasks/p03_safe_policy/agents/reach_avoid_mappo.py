from typing import Any, Union, Dict, Sequence, Mapping

import copy
import itertools
import gymnasium as gym
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.multi_agent import MultiAgent
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.utils.Running_mean_std import RunningMeanStd
from lib.utils.Learning_rate_scheduler import KLAdaptiveLR

from lib.utils.wrapper_utils import unflatten_tensorized_space

class ReachAvoidMAPPO(MultiAgent):
    def __init__(self,
                 observation_space: gym.Space,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 possible_agents: Sequence[str],
                 model: Dict[str, Union[nn.Module, Dict[str, nn.Module]]],
                 buffer: Dict[str, RolloutBuffer] | None,
                 device: Union[str, torch.device],
                 cfg: Dict) -> None:
        super().__init__(possible_agents, observation_space, state_space, action_space, cfg, model, device)

        # Models
        self.actors = {uid: self.model[uid].get("actor", None).to(self.device) for uid in self.possible_agents}
        self.value_critics = {uid: self.model[uid].get("value_critic", None).to(self.device) for uid in self.possible_agents}
        self.ra_critics = {uid: self.model[uid].get("ra_critic", None).to(self.device) for uid in self.possible_agents}
        
        self.value_standardizers = {uid: RunningMeanStd(shape=1, device=device) for uid in self.possible_agents}
        self.ra_standardizers = {uid: RunningMeanStd(shape=1, device=device) for uid in self.possible_agents}
        
        # Buffers
        if buffer is not None:
            self.buffer = {uid: buffer[uid] for uid in self.possible_agents}
        else:
            self.buffer = None

        # Load parameters form cfg
        self.rollouts = self.cfg["rollouts"]
        self.learning_epochs = self.cfg["learning_epochs"]
        self.mini_batches = self.cfg["mini_batches"]
        self.learning_rate = self.cfg["learning_rate"]
        self.learning_rate_scheduler = self.cfg["learning_rate_scheduler"]
        self.kl_threshold = self.cfg["kl_threshold"]
        self.discount_factor = self.cfg["discount_factor"]
        self.gae_lambda = self.cfg["lambda"]
        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.ratio_clip = self.cfg["ratio_clip"]
        self.value_clip = self.cfg["value_clip"]
        self.entropy_loss_scale = self.cfg["entropy_loss_scale"]
        # hyperparameters (value critic)
        self.value_discount_factor = self.cfg["value_discount_factor"]
        self.time_limit_bootstrap  = self.cfg["time_limit_bootstrap"]
        self.value_loss_scale = self.cfg["value_loss_scale"]
        # hyperparameters (reach-avoid critic)
        self.ra_discount_factor = self.cfg["ra_discount_factor"]
        self.ra_loss_scale = self.cfg["ra_loss_scale"]

        # Set up Adam optimizers
        self.optimizers = {}
        for uid in self.possible_agents:
            actor = self.actors[uid]
            value_critic = self.value_critics[uid]
            ra_critic = self.ra_critics[uid]

            self.optimizers[uid] = torch.optim.Adam(itertools.chain(actor.parameters(), 
                                                                    value_critic.parameters(),
                                                                    ra_critic.parameters()), lr=self.learning_rate)
            self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]

        # Set up learning rate scheduler
        if self.learning_rate_scheduler is not None:
            self.learning_rate_scheduler = {}
            for uid in self.possible_agents:
                self.learning_rate_scheduler[uid] = KLAdaptiveLR(self.optimizers[uid], self.kl_threshold)

        # Checkpoint models
        for uid in self.possible_agents:
            self.checkpoint_modules[uid] = {
                'actor': self.actors[uid],
                'value_critic': self.value_critics[uid],
                'ra_critic': self.ra_critics[uid],
                'value_standardizer': self.value_standardizers[uid],
                'ra_standardizer': self.ra_standardizers[uid],
                'optimizer': self.optimizers[uid],
            }

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
        # From Tensor to Dict
        observations = unflatten_tensorized_space(self.observation_space, observations)

        # Normal act list[(action, log_prob, entropy), (...), ()]
        data = []
        for uid in self.possible_agents:
            raw_action, log_prob, entropy = self.actors[uid](observations=observations[uid],
                                                                    taken_actions=None,
                                                                    deterministic=deterministic, 
                                                                    update_rms=update_rms)

            data.append((raw_action, log_prob, entropy))

        
        raw_actions  = torch.cat([d[0] for d in data], dim=-1) # [B, A]
        log_probs = torch.stack([d[1] for d in data], dim=-1) # [B, A]
        entropy  = torch.stack([d[2] for d in data], dim=-1) # [A]
        
        return raw_actions, log_probs, entropy
    

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
        # Unflatten from Tensor to Dict
        observations = unflatten_tensorized_space(self.observation_space, observations)
        next_observations = unflatten_tensorized_space(self.observation_space, next_observations)
        actions = unflatten_tensorized_space(self.action_space, actions)
        states = unflatten_tensorized_space(self.state_space, states)
        next_states = unflatten_tensorized_space(self.state_space, next_states)
        # Reshape for multi agent scale [B * N, 1] -> [B, N]
        buffer_action_log_probs = action_log_probs.view(-1, self.num_agents).clone()
        buffer_rewards = rewards.view(-1, self.num_agents).clone()

        for i, uid in enumerate(self.possible_agents):
            with torch.no_grad():
                value_preds, _, _ = self.value_critics[uid](states[uid]) # [E, 1]
                ra_preds, _, _ = self.ra_critics[uid](states[uid])
                value_preds = self.value_standardizers[uid].destandardize(value_preds)
                ra_preds = self.ra_standardizers[uid].destandardize(ra_preds)
            # time-limit (truncation) bootstrapping
            if self.time_limit_bootstrap:
                buffer_rewards[:, i:i+1] += self.discount_factor * value_preds * truncated # [E, 1]

            self.buffer[uid].add_samples(observations=observations[uid],
                                         states=states[uid],
                                         actions=actions[uid],
                                         rewards=buffer_rewards[:, i].unsqueeze(-1),
                                         g_values=g_values,
                                         l_values=l_values,
                                         next_observations=next_observations[uid],
                                         nexT_states=next_states[uid],
                                         truncated=truncated,
                                         terminated=terminated,
                                         action_log_probs=buffer_action_log_probs[:, i].unsqueeze(-1),
                                         value_preds = value_preds,
                                         ra_preds=ra_preds)
    
    def update(self) -> float:
        """
        Algorithm's main update step
        """
        # Loss initialization
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_approx_kl = 0
        learning_rate = {uid: 0.0 for uid in self.possible_agents}

        for uid in self.possible_agents:
            with torch.no_grad():
                critic_input = self.buffer[uid].get_tensor_by_name("next_states")[-1]
                last_values, _, _ = self.value_critics[uid](critic_input)
                last_values = self.value_standardizers[uid].destandardize(last_values)

                # RA value for every actual next state
                next_states = self.buffer[uid].get_tensor_by_name("next_states")
                next_states_flat = next_states.reshape(-1, next_states.shape[-1])
                next_ra_preds, _, _ = self.ra_critics[uid](next_states_flat)
                next_ra_preds = self.ra_standardizers[uid].destandardize(next_ra_preds)
                next_ra_preds = next_ra_preds.reshape(self.buffer[uid].buffer_size, -1, 1)
            
            # GAE Calculation
            self.buffer[uid].compute_gae(last_values, self.discount_factor, self.gae_lambda)
            # Value Standardization
            value_preds = self.value_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("value_preds").reshape(-1, 1), update=True)
            returns = self.value_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("returns").reshape(-1, 1), update=True)
            self.buffer[uid].set_tensor_by_name("value_preds", value_preds.reshape(self.buffer[uid].buffer_size, -1, 1))
            self.buffer[uid].set_tensor_by_name("returns", returns.reshape(self.buffer[uid].buffer_size, -1, 1))

            # GAE calculation (ra critic)
            self.buffer[uid].compute_ra_gae(next_ra_preds, self.ra_discount_factor, self.gae_lambda)
            # standardization
            ra_preds = self.ra_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("ra_preds").reshape(-1, 1), update=True)
            ra_returns = self.ra_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("ra_returns").reshape(-1, 1), update=True)
            self.buffer[uid].set_tensor_by_name("ra_preds", ra_preds.reshape(self.buffer[uid].buffer_size, -1, 1))
            self.buffer[uid].set_tensor_by_name("ra_returns", ra_returns.reshape(self.buffer[uid].buffer_size, -1, 1))
            
            # Parameter Update
            kl_divergences = []
            self.set_running_mode(mode="train", uid=uid)

            # Sample mini batch
            mini_batches = self.buffer[uid].sample(names=self.tensors_names_for_updates, mini_batch=self.mini_batches)
        
            for epoch in range(self.learning_epochs):
                for mb in mini_batches:
                    # (mini batch size, Data-specific)
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
                    ) = mb
                    
                    actor_input = sampled_observations
                    critic_input = sampled_states
                
                    _, new_log_probs, dist_entropy = self.actors[uid](observations=actor_input, 
                                                                        taken_actions=sampled_actions,
                                                                        deterministic=False,
                                                                        update_rms=not epoch)

                    # Shape syncronization
                    if len(new_log_probs.shape) != len(sampled_action_log_probs.shape):
                        new_log_probs = new_log_probs.reshape(sampled_action_log_probs.shape)

                    # Compute approximate KL divergence
                    with torch.no_grad():
                        ratio = new_log_probs - sampled_action_log_probs
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)
                    
                    # Compute entropy loss
                    if self.entropy_loss_scale:
                        entropy_loss = -self.entropy_loss_scale * dist_entropy
                    else:
                        entropy_loss = 0

                    # Compute policy loss
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

                    # Compute value loss
                    predicted_values, _, _ = self.value_critics[uid](critic_input, update_rms=not epoch)
                    value_loss = self.value_loss_scale * F.mse_loss(sampled_value_returns, predicted_values)
                    
                    # Optimization step
                    self.optimizers[uid].zero_grad()
                    (policy_loss + entropy_loss + value_loss).backward()

                    if self.grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(itertools.chain(self.actors[uid].parameters(), self.value_critics[uid].parameters()), self.grad_norm_clip)
                    
                    self.optimizers[uid].step()

                    # Update cumulative losses
                    cumulative_policy_loss += policy_loss.item()
                    cumulative_value_loss += value_loss.item()
                    cumulative_approx_kl += kl_divergence.item()
                    if self.entropy_loss_scale:
                        cumulative_entropy_loss += entropy_loss.item()
                
            # Learning rate scheduler update
            if self.learning_rate_scheduler is not None:
                kl = torch.tensor(kl_divergences, device=self.device).mean()
                self.learning_rate_scheduler[uid].step(kl.item())
                learning_rate[uid] = self.learning_rate_scheduler[uid].get_last_lr()[0]
            else:
                learning_rate[uid] = self.optimizers[uid].param_groups[0]["lr"]
                    

        self.set_running_mode("eval")
        mean_policy_loss = cumulative_policy_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_value_loss = cumulative_value_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_entropy_loss = cumulative_entropy_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_kl_divergence = cumulative_approx_kl / (self.learning_epochs * self.mini_batches * self.num_agents)


        return mean_policy_loss, mean_value_loss, mean_entropy_loss, mean_kl_divergence, learning_rate
            