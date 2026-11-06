# Copyright (c) 2026, AISL.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from torch.distributions import Normal
from lib.utils.Running_mean_std import RunningMeanStd
from lib.model.model import Model

class ReachAvoidActor(Model):
    def __init__(self, num_observations, num_actions, min_log_std, max_log_std, squash, device):
        super().__init__()

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # Define instances 
        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions
        
        # Action Squashing
        self.squash = squash

        # Running mean, standard deviation standardizer
        self.actor_standardizer = RunningMeanStd(shape=self.num_observations, device=device)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device), requires_grad=True) # State independent log std

        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)

    def forward(self, observations: torch.Tensor, taken_actions: torch.Tensor | None, deterministic: bool = False, update_rms: bool = False):
        """
        Forward propagation of actor NN
        
        :param observations: Observation vector
        :type observations: torch.Tensor
        :param taken_actions: Action vector
        :type taken_actions: torch.Tensor
        :param deterministic: Is actor evaluation mode
        :type deterministic: bool
        :param update_rms: Update a Runningmeanstd distrubution
        :type update_rms: bool 
        """
        # eps
        eps = 1e-6
        # Input standardization
        standardized_input = self.actor_standardizer.standardize(observations, update=update_rms)

        mean_actions = self.net(standardized_input)
        log_std = self.log_std_parameter
        
        # Clamp log standard deviations
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        # Action distribution
        action_distribution = Normal(mean_actions, log_std.exp())

        if deterministic:
            raw_actions = mean_actions
        else:
            # Sample using the reparameterization trick
            raw_actions = action_distribution.rsample()

        # Log of the probability density function
        if self.squash:
            # tanh squasing with log probability dorrection
            actions = torch.tanh(raw_actions)
            if taken_actions is not None:
                taken_actions = torch.clip(taken_actions, -1.0 + eps, 1.0 - eps)
                raw_taken_actions = torch.atanh(taken_actions)
                log_prob = action_distribution.log_prob(raw_taken_actions) - torch.log(1 - taken_actions.pow(2) + eps)
            else:
                log_prob = action_distribution.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + eps)

        else:
            # no squasing without correction
            actions = raw_actions
            if taken_actions is not None:
                log_prob = action_distribution.log_prob(taken_actions)
            else:
                log_prob = action_distribution.log_prob(actions)

        log_prob = log_prob.sum(dim=-1)

        # Entropy : mean of (Batch, Action) dimension
        entropy = action_distribution.entropy().mean()

        return actions, log_prob, entropy
    
    def random_act(self, observations: torch.Tensor):
        """
        Random act for RL's beginning
        """
        eps = 1e-6
        batch_size = observations.shape[0]

        # Create a standard normal distribution for random actions (mean=0, std=1)
        mean = torch.zeros((batch_size, self.num_actions), device=self.device)
        std = torch.ones((batch_size, self.num_actions), device=self.device)
        action_distribution = Normal(mean, std)

        # Sample raw actions and squash them to [-1, 1] using tanh
        raw_actions = action_distribution.rsample()
        if self.squash:
            # tanh squasing with log probability dorrection
            actions = torch.tanh(raw_actions)
            log_prob = action_distribution.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + eps)

        else:
            # no squasing without correction
            actions = raw_actions
            log_prob = action_distribution.log_prob(actions)

        log_prob = log_prob.sum(dim=-1)

        # Entropy of the base (Normal) distribution
        entropy = action_distribution.entropy().mean()

        return actions, log_prob, entropy


class ReachAvoidCritic(Model):
    def __init__(self, num_states, device, M = 1.0):
        super().__init__()

        # Define instances 
        self.device = device
        self.num_states = num_states
        self.M = M

        # Running mean, standard deviation standardizer
        self.critic_standardizer = RunningMeanStd(shape=self.num_states, device=device)

        # Backbone
        self.net = nn.Sequential(nn.Linear(self.num_states, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 1))

        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)

    def forward(self, inputs: torch.Tensor, deterministic: bool = False, update_rms: bool = False):
        standardized_input = self.critic_standardizer.standardize(inputs, update=update_rms)
        expected_return = self.net(standardized_input)

        return expected_return, None, None
        # return self.M * torch.tanh(expected_return), None, None


class ValueCritic(Model):
    def __init__(self, num_states, device):
        super().__init__()

        # Define instances 
        self.device = device
        self.num_states = num_states

        # Running mean, standard deviation standardizer
        self.critic_standardizer = RunningMeanStd(shape=self.num_states, device=device)

        # Backbone
        self.net = nn.Sequential(nn.Linear(self.num_states, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 1))

        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)

    def forward(self, inputs: torch.Tensor, deterministic: bool = False, update_rms: bool = False):
        standardized_input = self.critic_standardizer.standardize(inputs, update=update_rms)
        expected_return = self.net(standardized_input)

        return expected_return, None, None