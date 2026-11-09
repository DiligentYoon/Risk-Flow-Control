
import gymnasium
import torch

from typing import Optional, Union, Tuple, List
from lib.buffer.buffer import Buffer


class ReachAvoidBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int = 1,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__(buffer_size, num_envs, device)

    def init_buffer(self, observation_space:gymnasium.Space, state_space:Union[gymnasium.Space | None], action_space:gymnasium.Space):
        """Initialization Rollout Buffer for Reach-Avoid On-Policy Setting including GAE Buffer"""
        super().init_buffer(observation_space, state_space, action_space)
        # GAE Buffer
        self.create_tensor("value_returns", 1, dtype=torch.float32)
        self.create_tensor("value_advantages", 1, dtype=torch.float32)
        self.create_tensor("ra_preds", 1, dtype=torch.float32)
        self.create_tensor("ra_returns", 1, dtype=torch.float32)
        self.create_tensor("ra_advantages", 1, dtype=torch.float32)
        self.create_tensor("g_values", 1, dtype=torch.float32)
        self.create_tensor("l_values", 1, dtype=torch.float32)


    def sample(self, names:Tuple[str], mini_batch: int) -> List[List[torch.Tensor]]:
        """Random sampling from Stochastic Gradient Descent in On-policy Algorithm"""
        # compute valid memory size
        size = len(self)

        # generate random samples
        indexes = torch.randperm(size, dtype=torch.long)

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batch)


    def compute_value_gae(self, next_value: torch.Tensor, gamma: float, lamb: float):
        """Compute Generalized Advantage Expectation (GAE)"""
        return_tensor = torch.zeros_like(self.tensors['value_returns'], dtype=torch.float32, device=self.device)
        advantage_tensor = torch.zeros_like(self.tensors['value_advantages'], dtype=torch.float32, device=self.device)
        rollout = self.buffer_size

        reward = self.tensors["rewards"] # [T, E, 1]
        value_preds = self.tensors["value_preds"] # [T, E, 1]
        mask = torch.logical_not(torch.logical_or(self.tensors["terminated"], self.tensors["truncated"])).float() # [T, E, 1]

        gae = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        for step in reversed(range(rollout)):
            if step == (rollout-1):
                value_pred = next_value
            else:
                value_pred = value_preds[step + 1]

            # Predicted value : Syncronization
            delta = reward[step] + gamma * value_pred * mask[step] - value_preds[step]
            gae = delta + gamma * lamb * mask[step] * gae

            advantage_tensor[step] = gae
            return_tensor[step] = gae + value_preds[step]
            
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
        
        self.set_tensor_by_name('value_returns', return_tensor)
        self.set_tensor_by_name('value_advantages', advantage_tensor)


    def compute_ra_gae(self, next_ra_preds: torch.Tensor, gamma: float, lamb: float):
        """Compute Generalized Advantage Expectation (GAE) with perspective of Reach-Avoid"""
        return_tensor = torch.zeros_like(self.tensors['ra_returns'], dtype=torch.float32, device=self.device)
        advantage_tensor = torch.zeros_like(self.tensors['ra_advantages'], dtype=torch.float32, device=self.device)
        rollout = self.buffer_size

        ra_preds = self.tensors["ra_preds"] # [T, E, 1]
        g_values = self.tensors["g_values"]
        l_values = self.tensors["l_values"]

        ra_targets = gamma * torch.maximum(g_values, torch.minimum(l_values, next_ra_preds))

        advantage_tensor = ra_preds - ra_targets
        return_tensor = ra_targets

        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)

        self.set_tensor_by_name("ra_returns", return_tensor)
        self.set_tensor_by_name("ra_advantages", advantage_tensor)
        



    