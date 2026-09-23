"""
Script to train a Reach-avoid value function.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="R1-fall", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

parser.add_argument("--rollout_steps",
                    type=int,
                    default=3000,
                    help="Number of vectorized environment steps to collect.")

parser.add_argument("--algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the agent.")

parser.add_argument("--model",
                    type=str,
                    default="Shared",
                    choices=["MLP", "Shared", "Communet"],
                    help="The NN model used for training the agent.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
args_cli.headless = True                    # Headless mode
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from datetime import datetime

import lib
import tasks

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.model.model_factory import ModelFactory

from tasks.p02_safety_value.buffer.rollout_collector import SafetyRolloutCollector
from tasks.p02_safety_value.wrappers.safety_wrapper import SafetyEnvWrapper

# config shortcuts
algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None

def main():
    # ============================= Config Parsing ===============================
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    # specify directory for logging experiments (load checkpoint)
    if args_cli.checkpoint is not None:
        base_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        log_dir  = os.path.join(base_dir, "Predictor", "Dataset")
        log_dir  = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        dataset_path = os.path.join(log_dir, "data_raw.hdf5")
    else:
        dataset_path = None

    # ============================ Env & Wrapper Spawn ================================

    # Create isaac environment
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", None)
        cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
    
    env_cfg.total_timesteps = args_cli.rollout_steps
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # Wrap around environment
    env = SafetyEnvWrapper(env)  

    # ======================= Buffer =========================
    multi_agent = algorithm == "mappo"
    cfg["models"]["multi_agent"] = multi_agent
    # Initialization
    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")
    
    possible_agents = None
    if multi_agent:
        obs_size = {}
        state_size = {}
        act_size = {}
        buffers = {}
        possible_agents = env._unwrapped.cfg.possible_agents
        num_agent = len(possible_agents)
        for uid in possible_agents:
            observation_space = env.observation_space[uid]
            action_space = env.action_space[uid]
            if env.state_space:
                state_space = env.state_space[uid]
                cfg["agent"]["async_actor_critic"] = True
            else:
                state_space = None
                cfg["agent"]["async_actor_critic"] = False
            
            buffer = RolloutBuffer(cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
            buffer.init_buffer(observation_space, state_space, action_space)
            buffers[uid] = buffer
            obs_size[uid] = buffer.tensors["observations"].shape[-1]
            state_size[uid] = buffer.tensors["states"].shape[-1] if env.state_space else obs_size[uid]
            act_size[uid] = buffer.tensors["actions"].shape[-1]

    else:
        observation_space = env.observation_space
        action_space = env.action_space
        if env.state_space:
            state_space = env.state_space
            cfg["agent"]["async_actor_critic"] = True
        else:
            state_space = None
            cfg["agent"]["async_actor_critic"] = False
        
        buffer = RolloutBuffer(cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
        buffer.init_buffer(observation_space, state_space, action_space)
        obs_size = buffer.tensors["observations"].shape[-1]
        state_size = buffer.tensors["states"].shape[-1] if env.state_space else obs_size
        act_size = buffer.tensors["actions"].shape[-1]

    # ====================== Model Spawn  ==========================
    # Overwrite cfg by cli argument
    if model is not None:
        cfg["models"]["model_type"] = model
    
    model_manager = ModelFactory(cfg=cfg["models"], device=env.device)
    if model_manager.model_class == "mlp":
        models = model_manager.generate_mlp_models(observation_size=obs_size,
                                                   state_size=state_size,
                                                   action_size=act_size,
                                                   possible_agents=possible_agents)
    else:
        raise RuntimeError("Not supported class")

    # ====================== Agent Spawn  ==========================
    if multi_agent:
        if model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO
            agent = MAPPO(observation_space=env.observation_space,
                          state_space=env.state_space,
                          action_space=env.action_space,
                          possible_agents=possible_agents,
                          model=models,
                          buffer=buffers,
                          device=env.device,
                          cfg=cfg["agent"])
        
        elif model_manager.model_type == "shared":
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            agent = CooperativeMAPPO(observation_space=env.observation_space,
                                    state_space=env.state_space,
                                    action_space=env.action_space,
                                    possible_agents=possible_agents,
                                    model=models,
                                    buffer=buffers,
                                    device=env.device,
                                    cfg=cfg["agent"])
        
        else:
            raise RuntimeError("Unvalid model type.")

    else:
        from lib.agent.ppo import PPO
        agent = PPO(model=models,
                    buffer=buffer, 
                    device=env.device,
                    cfg=cfg["agent"])

    # Checkpoint (Nominal Policy)
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get checkpoint of nominal policy from {resume_path}")
    else:
        resume_path = None
        print("[INFO] Unfortunately a pre-trained nominal policy is not found for this task.")

    # ====================== Dataset Collector ================================ #

    metadata = {
        "task": args_cli.task,
        "seed": env_cfg.seed,
        "step_dt": float(env._unwrapped.step_dt),
        "episode_length_s": float(env._unwrapped.cfg.episode_length_s),
        "nominal_checkpoint": resume_path,
        "algorithm": algorithm,
        "model": model,
    }

    collector = SafetyRolloutCollector(
        file_path=dataset_path,
        num_envs=env.num_envs,
        safety_state_dim=env._unwrapped.num_safety_states,
        max_episode_length=env._unwrapped.max_episode_length,
        metadata=metadata,
    )

    if dataset_path is not None:
        print(f"[INFO] Dataset path: {dataset_path}")
    else:
        print("[INFO] Dataset saving disabled (dry-run mode).")

    print(f"[INFO] Number of environments: {env.num_envs}")
    print(f"[INFO] Maximum episode length: {env._unwrapped.max_episode_length}")
    print(f"[INFO] Safety-state dimension: {env._unwrapped.num_safety_states}")

    # ====================== Rollout Collection =============================== #

    obs, _, _, infos = env.reset()

    CLI_interval = 300
    CLI_num_written = 0
    timestep = 0
    start_time = time.time()

    try:
        while (simulation_app.is_running() and timestep < args_cli.rollout_steps):
            with torch.inference_mode():
                actions, _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)

                (
                    next_obs,
                    next_states,
                    next_safety_states,
                    next_safety_values,
                    final_safety_states,
                    final_safety_values,
                    rewards,
                    terminated,
                    truncated,
                    final_push_events,
                    next_infos,
                ) = env.step(actions)

            num_written = collector.append(
                final_safety_states=final_safety_states,
                final_safety_values=final_safety_values,
                final_push_events=final_push_events,
                terminated=terminated,
                truncated=truncated,
            )

            CLI_num_written += num_written
            timestep += 1

            # update policy inputs
            obs = next_obs
            states = next_states
            safety_states = next_safety_states
            safety_values = next_safety_values
            infos = next_infos

            if timestep % CLI_interval == 0 or timestep == args_cli.rollout_steps:

                print(
                    f"[COLLECT] "
                    f"step={timestep}/{args_cli.rollout_steps} | "
                    f"episodes={collector.episode_count} | "
                    f"episodes_interval={CLI_num_written} | "
                    f"samples={collector.sample_count}"
                )

                CLI_num_written = 0

    finally:
        # Keep only complete episodes.
        collector.close(save_partial=False)
        env.close()

    print(
        f"[INFO] Collection completed. "
        f"Episodes saved: {collector.episode_count}"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()