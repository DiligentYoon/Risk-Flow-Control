"""
Script to play Safety-Q shielding.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import gymnasium as gym
import os
import time
import torch
import numpy as np
import collections

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a Safety-Q shielding policy.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="R1-intervention-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to Safety-Q checkpoint.")
parser.add_argument("--nominal_checkpoint", type=str, default=None, help="Path to frozen nominal PPO checkpoint.")
parser.add_argument("--nominal_algorithm", type=str, default="PPO", choices=["PPO", "SAC", "TD3", "MAPPO"], help="The RL algorithm used for training the agent.")
parser.add_argument("--model", type=str, default="MLP", choices=["MLP", "Shared"], help="The NN model used for training the agent.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from torch.utils.tensorboard import SummaryWriter

import lib
import tasks

from lib.model.model_factory import ModelFactory
from lib.utils.parse_utils import load_cfg_from_registry, parse_env_cfg
from tasks.p02_1_safety_action_value.agent.safety_q import SafetyQ
from tasks.p02_1_safety_action_value.buffer.replay_buffer import ReplayBuffer
from tasks.p02_1_safety_action_value.model.safety_q import SafetyQActor, SafetyQCritic
from tasks.p02_1_safety_action_value.utils.scheduler import StepScheduler
from tasks.p02_1_safety_action_value.wrappers.intervention_wrapper import InterventionEnvRecordVideo, InterventionEnvWrapper
from tasks.p02_1_safety_action_value.utils.evaluator import RolloutEvaluator

# config shortcuts
algorithm = args_cli.nominal_algorithm.lower()

def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        nominal_cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
        cfg = load_cfg_from_registry(args_cli.task, "safety_q_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        nominal_cfg["agent"]["seed"] = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", 42)
        nominal_cfg["agent"]["seed"] = cfg.get("seed", 42)
        cfg["agent"]["seed"] = cfg.get("seed", 42)

    if args_cli.checkpoint is not None:
        log_dir = os.path.dirname(args_cli.checkpoint)
        print(f"[INFO] Loading experiment from directory: {log_dir}")
    else:
        log_dir = None
        print(f"[INFO] No Checkpoint Mode.")

    # ============================= Environment =================================
    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video and log_dir is not None:
        args_cli.video_interval = int(cfg["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = InterventionEnvRecordVideo(env, **video_kwargs)

    env = InterventionEnvWrapper(env)

    # ======================= Config Setting ======================== 
    model_type = args_cli.model.lower() if args_cli.model is not None else None
    multi_agent = (algorithm == "mappo")
    possible_agents = env._unwrapped.cfg.possible_agents if multi_agent else None

    # ======================== Safety Agent ============================ #
    buffer = ReplayBuffer(buffer_size=cfg["buffer"]["buffer_size"], num_envs=env.num_envs, device=env.device)
    buffer.init_buffer(env.observation_space, env.state_space, env.action_space)

    observation_dim = buffer.tensors["observations"].shape[-1]
    state_dim = buffer.tensors["states"].shape[-1]
    action_dim = buffer.tensors["actions"].shape[-1]
    model_cfg = cfg["agent"]["model"]

    model = {
        "actor": SafetyQActor(observation_dim, action_dim, model_cfg["min_log_std"], model_cfg["max_log_std"], device=env.device),
        "critic": SafetyQCritic(state_dim, action_dim, device=env.device),
    }
    agent = SafetyQ(model=model, buffer=buffer, device=env.device, cfg=cfg["agent"])

    # ================= Task (Nominal) Policy ==================== #
    nominal_cfg["models"]["model_type"] = model_type
    nominal_cfg["models"]["multi_agent"] = multi_agent
    model_manager = ModelFactory(cfg=nominal_cfg["models"], device=env.device)

    models = model_manager.generate_mlp_models(observation_size=observation_dim,
                                                state_size=state_dim,
                                                action_size=action_dim,
                                                possible_agents=possible_agents)

    if multi_agent:
        if model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO
            nominal_agent = MAPPO(observation_space=env.observation_space,
                                  state_space=env.state_space,
                                  action_space=env.action_space,
                                  possible_agents=possible_agents,
                                  model=models,
                                  buffer=None,
                                  device=env.device,
                                  cfg=nominal_cfg["agent"]
                                )
        elif model_manager.model_type == "shared":
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            nominal_agent = CooperativeMAPPO(observation_space=env.observation_space,
                                             state_space=env.state_space,
                                             action_space=env.action_space,
                                             possible_agents=possible_agents,
                                             model=models,
                                             buffer=None,
                                             device=env.device,
                                             cfg=nominal_cfg["agent"]
                                            )
        else:
            raise RuntimeError("Unvalid model type.")
    else:
        from lib.agent.ppo import PPO
        nominal_agent = PPO(model=models,
                            buffer=None, 
                            device=env.device,
                            cfg=nominal_cfg["agent"])

    if args_cli.nominal_checkpoint is not None:
        nominal_checkpoint_path = os.path.abspath(args_cli.nominal_checkpoint)
        nominal_agent.load(nominal_checkpoint_path)
        print(f"[INFO] Get Nominal Policy from {nominal_checkpoint_path}")

    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get Safety-Q checkpoint from {resume_path}")

    # ============================= Evaluator =============================
    evaluator = RolloutEvaluator(
        num_envs=env.num_envs,
        max_episode_length=env._unwrapped.max_episode_length,
        device=env.device,
        discount_factor=cfg["agent"]["discount_factor"],
        threshold=cfg["agent"]["threshold"],
    )

    # ======================= Initial Setting ========================= #
    CLI_track_timesteps = collections.deque(maxlen=env.num_envs)
    cumulative_timesteps = torch.zeros((env.num_envs, 1), dtype=torch.int32, device=env.device)

    obs, states, reach_values, safety_values, infos = env.reset()
    timestep = 0
    logstep = 0
    elapsed_time = 0.0
    start_time = time.time()
    CLI_interval = 500

    # ======================= Interaction ========================= #
    try:
        while simulation_app.is_running() and timestep < cfg["train"]["timesteps"]:
            with torch.inference_mode():
                backup_actions = agent.act(observations=obs, deterministic=True)
                pred_values = agent.predict(states=states, actions=backup_actions).squeeze(-1)
                executed_actions = backup_actions

                (
                    next_obs,
                    next_states,
                    next_reach_values,
                    next_safety_values,
                    final_obs,
                    final_states,
                    final_reach_values,
                    final_safety_values,
                    rewards,
                    terminated,
                    truncated,
                    next_infos,
                ) = env.step(executed_actions)

                final_backup_actions = agent.act(observations=final_obs, deterministic=True)
                final_pred_values = agent.predict(states=final_states, actions=final_backup_actions).squeeze(-1)

            timestep += 1

            evaluator.append(
                g_values=safety_values,
                l_values=reach_values,
                pred_values=pred_values,
                final_g_values=final_safety_values,
                final_l_values=final_reach_values,
                final_pred_values=final_pred_values,
                terminated=terminated,
                truncated=truncated,
            )

            cumulative_timesteps.add_(1)
            done = (terminated | truncated).squeeze(-1)
            finished_episodes = done.nonzero(as_tuple=False).squeeze(-1)
            if finished_episodes.numel():
                CLI_track_timesteps.extend(cumulative_timesteps[finished_episodes, 0].detach().cpu().tolist())
                cumulative_timesteps[finished_episodes] = 0

            if timestep % CLI_interval == 0:
                end_time = time.time()
                avg_ep_step = float(np.mean(CLI_track_timesteps)) if CLI_track_timesteps else float("nan")
                avg_ep_step_str = "-" if np.isnan(avg_ep_step) else f"{avg_ep_step:6.3f} steps"

                elapsed_time += end_time - start_time
                e_h = int(elapsed_time // 3600)
                e_m = int((elapsed_time % 3600) // 60)
                e_s = int(elapsed_time % 60)
                total_rollout = int(cfg["train"]["timesteps"] // CLI_interval)
                complete_time = (end_time - start_time) * max(total_rollout - logstep, 0)
                c_h = int(complete_time // 3600)
                c_m = int((complete_time % 3600) // 60)
                c_s = int(complete_time % 60)

                content_width = 64
                line_header = f"Step Progress {timestep} / {args_cli.video_length}"
                line_time_header = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
                lines = [
                    f"Rollout Time            : {end_time - start_time:6.3f} sec",
                    f"Avg Episode Step        : {avg_ep_step_str}",
                ]

                print(" ________________________________________________________________")
                print("|                                                                |")
                print(f"|{line_header.center(content_width)}|")
                print(f"|{line_time_header.center(content_width)}|")
                print("|________________________________________________________________|")
                print("|                                                                |")
                for line in lines:
                    print(f"| {line:<{content_width-1}}|")
                print("|________________________________________________________________|")

                CLI_track_timesteps.clear()
                logstep += 1
                start_time = end_time

            obs = next_obs
            states = next_states
            reach_values = next_reach_values
            safety_values = next_safety_values
            infos = next_infos

            if timestep == args_cli.video_length:
                break

        if evaluator.num_episodes > 0:
            step_dt = float(env._unwrapped.step_dt)
            plot_dir = os.path.join(log_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            evaluator.save_timeseries_plot(file_path=os.path.join(plot_dir, "value_trajectory.png"), step_dt=step_dt)
            print(f"[INFO] Evaluation plots saved to: {log_dir}")


    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()