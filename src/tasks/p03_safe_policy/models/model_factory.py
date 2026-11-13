import torch
import gymnasium as gym

from typing import Union, Any

from lib.model.model import Model
from lib.env.env import Env

from tasks.p03_safe_policy.buffer.reach_avoid_buffer import ReachAvoidBuffer
from tasks.p03_safe_policy.models.reach_avoid_model import ReachAvoidCritic, ReachAvoidActor, ValueCritic

class ReachAvoidModelFactory:
    def __init__(self,
                 buffer_cfg: dict[str, Union[dict, Any]],
                 model_cfg: dict[str, Union[dict, Any]],
                 device: Union[torch.device, str]):
        """
        ModelFactory class for generating nn Model and corresponding buffer class.

        :param buffer_cfg: configuration dictionary of buffer class
        :param model_cfg: configuration dictionary of model class
        :param device: torch device
        """
        self.buffer_cfg = buffer_cfg
        self.model_cfg = model_cfg
        self.device = device

        self.num_agents = 1
        self.possible_agents = None
        self.observation_size = None
        self.state_size = None
        self.action_size = None

    def generate_buffers(self, env: Env) -> Union[ReachAvoidBuffer, dict[str, ReachAvoidBuffer]]:
        if self.model_cfg["multi_agent"]:
            obs_size = {}
            state_size = {}
            act_size = {}
            buffer = {}
            possible_agents = env._unwrapped.cfg.possible_agents
            num_agents = len(possible_agents)
            for uid in possible_agents:
                observation_space = env.observation_space[uid]
                action_space = env.action_space[uid]
                state_space = env.state_space[uid]
                
                buffer_uid = ReachAvoidBuffer(self.buffer_cfg["buffer_size"], env.num_envs, device=env.device)
                buffer_uid.init_buffer(observation_space, state_space, action_space)
                buffer[uid] = buffer_uid
                obs_size[uid] = buffer_uid.tensors["observations"].shape[-1]
                state_size[uid] = buffer_uid.tensors["states"].shape[-1] if env.state_space else obs_size[uid]
                act_size[uid] = buffer_uid.tensors["actions"].shape[-1]
        else:
            possible_agents = None
            num_agents = 1
            observation_space = env.observation_space
            action_space = env.action_space
            state_space = env.state_space
            state_space = env.state_space

            buffer = ReachAvoidBuffer(self.buffer_cfg["buffer_size"], env.num_envs, device=env.device)
            buffer.init_buffer(observation_space, state_space, action_space)
            obs_size = buffer.tensors["observations"].shape[-1]
            state_size = buffer.tensors["states"].shape[-1] if env.state_space else obs_size
            act_size = buffer.tensors["actions"].shape[-1]

        self.num_agents = num_agents
        self.possible_agents = possible_agents
        self.observation_size = obs_size
        self.state_size = state_size
        self.action_size = act_size

        return buffer

    def generate_models(self) -> dict[str, Model]:
        if self.model_cfg["multi_agent"]:
            models = {}
            for uid in self.possible_agents:
                actor = ReachAvoidActor(num_observations=self.observation_size[uid],
                                        num_actions=self.action_size[uid],
                                        min_log_std=self.model_cfg["policy"]["min_log_std"],
                                        max_log_std=self.model_cfg["policy"]["max_log_std"],
                                        squash=self.model_cfg["squashed"],
                                        device=self.device)
                value_critic = ValueCritic(num_states=self.state_size[uid],
                                           device=self.device)
                ra_critic = ReachAvoidCritic(num_states=self.state_size[uid],
                                             device=self.device,
                                             M=self.model_cfg["ra_value"]["M"])

                models[uid] = {
                                'actor': actor,
                                'value_critic': value_critic,
                                'ra_critic': ra_critic,
                              }
        else:
            actor = ReachAvoidActor(num_observations=self.observation_size,
                                    num_actions=self.action_size,
                                    min_log_std=self.model_cfg["policy"]["min_log_std"],
                                    max_log_std=self.model_cfg["policy"]["max_log_std"],
                                    squash=self.model_cfg["squashed"],
                                    device=self.device)
            value_critic = ValueCritic(num_states=self.state_size,
                                       device=self.device)
            ra_critic = ReachAvoidCritic(num_states=self.state_size,
                                         device=self.device,
                                         M=self.model_cfg["ra_value"]["M"])

            models = {
                        "actor": actor,
                        "value_critic": value_critic,
                        "ra_critic": ra_critic,
                     }

        return models