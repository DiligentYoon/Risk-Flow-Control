import torch
import torch.nn as nn
from lib.utils.Running_mean_std import RunningMeanStd
from lib.model.model import Model


class SafetyCritic(Model):
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

        # optimistic initialization for self-reinforcing error propagation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -0.8) # minimum of safety target value = -1

    def forward(self, inputs: torch.Tensor, deterministic: bool = False, update_rms: bool = False):
        """
        Forward propagation of Safety critic NN
        
        :param inputs: State vector
        :type inputs: torch.Tensor
        :param deterministic: Is critic evaluation mode
        :type deterministic: bool 
        """

        standardized_input = self.critic_standardizer.standardize(inputs, update=update_rms)
        expected_return = self.net(standardized_input)

        return expected_return, None, None