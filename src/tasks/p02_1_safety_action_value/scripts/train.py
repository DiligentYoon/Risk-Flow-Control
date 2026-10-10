"""
Script to train Safety-Q shielding.
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
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="R1-intervention", help="Name of the task.")
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

    if args_cli.nominal_checkpoint is not None:
        log_root_path = os.path.join(os.path.dirname(args_cli.nominal_checkpoint))
    else:
        log_root_path = os.path.abspath(os.path.join("logs", cfg["agent"]["experiment"]["directory"]))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # ============================= Environment =================================
    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video and log_dir is not None:
        args_cli.video_interval = int(cfg["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = InterventionEnvRecordVideo(env, **video_kwargs)

    env = InterventionEnvWrapper(env)

    write_interval = cfg["agent"]["experiment"]["write_interval"]
    checkpoint_interval = cfg["agent"]["experiment"]["checkpoint_interval"]
    if write_interval == "auto":
        write_interval = max(1, int(cfg["train"]["timesteps"] / 100))
    if checkpoint_interval == "auto":
        checkpoint_interval = max(1, int(cfg["train"]["timesteps"] / 5))

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

    # ======================= Initial Setting ========================= #
    writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
    tracking_data = collections.defaultdict(list)
    tracking_timesteps = collections.deque(maxlen=env.num_envs)
    CLI_track_timesteps = collections.deque(maxlen=env.num_envs)
    CLI_track_q_term = collections.deque(maxlen=env.num_envs)
    CLI_track_g_term = collections.deque(maxlen=env.num_envs)
    CLI_track_l_term = collections.deque(maxlen=env.num_envs)
    CLI_track_q_term_min = collections.deque(maxlen=env.num_envs)
    CLI_track_g_term_min = collections.deque(maxlen=env.num_envs)
    CLI_track_l_term_min = collections.deque(maxlen=env.num_envs)
    CLI_track_g = collections.deque(maxlen=env.num_envs)
    CLI_track_l = collections.deque(maxlen=env.num_envs)
    cumulative_timesteps = torch.zeros((env.num_envs, 1), dtype=torch.int32, device=env.device)

    obs, states, reach_values, safety_values, infos = env.reset()
    timestep = 0
    logstep = 0
    elapsed_time = 0.0
    update_info = None
    start_time = time.time()
    CLI_interval = min(buffer.buffer_size, 256)

    # ======================= Interaction ========================= #
    try:
        while simulation_app.is_running() and timestep < cfg["train"]["timesteps"]:
            with torch.inference_mode():
                backup_actions = agent.act(observations=obs, deterministic=False)
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

            timestep += 1

            agent.insert_data(
                observations=obs,
                states=states,
                reach_values=reach_values,
                safety_values=safety_values,
                next_observations=final_obs,
                next_states=final_states,
                next_reach_values=final_reach_values,
                next_safety_values=final_safety_values,
                actions=executed_actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
            )

            if cfg["train"]["learning_starts"] <= timestep:
                update_info = agent.update()

            # Update Scheduler
            agent.step_gamma(timestep)

            tracking_data["Safety / value mean"].append(safety_values.mean().item())
            tracking_data["Safety / next value mean"].append(final_safety_values.mean().item())

            if update_info is not None:
                for k, v in update_info.items():
                    tracking_data[f"Learning / {k}"].append(v)

            cumulative_timesteps.add_(1)
            CLI_track_l.append(torch.mean(reach_values, dim=0).item())
            CLI_track_g.append(torch.mean(safety_values, dim=0).item())

            done = (terminated | truncated).squeeze(-1)
            finished_episodes = done.nonzero(as_tuple=False).squeeze(-1)
            terminated_episodes = terminated.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
            if finished_episodes.numel():
                tracking_timesteps.extend(cumulative_timesteps[finished_episodes, 0].detach().cpu().tolist())
                CLI_track_timesteps.extend(cumulative_timesteps[finished_episodes, 0].detach().cpu().tolist())
                cumulative_timesteps[finished_episodes] = 0
            if tracking_timesteps:
                tracking_data["Episode / Total timesteps (mean)"].append(float(np.mean(tracking_timesteps)))
                tracking_timesteps.clear()
            if terminated_episodes.numel():
                with torch.no_grad():
                    q_term = agent.predict(states[terminated_episodes], executed_actions[terminated_episodes])
                    g_term = final_safety_values[terminated_episodes]
                    l_term = final_reach_values[terminated_episodes]
                    q_term_min = q_term.min(dim=0).values
                    g_term_min = g_term.min(dim=0).values
                    l_term_min = l_term.min(dim=0).values
                CLI_track_q_term.extend(q_term.detach().cpu().tolist())
                CLI_track_g_term.extend(g_term.detach().cpu().tolist())
                CLI_track_l_term.extend(l_term.detach().cpu().tolist())
                CLI_track_q_term_min.extend(q_term_min.detach().cpu().tolist())
                CLI_track_g_term_min.extend(g_term_min.detach().cpu().tolist())
                CLI_track_l_term_min.extend(l_term_min.detach().cpu().tolist())

            if (timestep % write_interval == 0) and log_dir is not None:
                for k, v in tracking_data.items():
                    if not v:
                        continue
                    if k.endswith("(min)"):
                        writer.add_scalar(k, np.min(v), timestep)
                    elif k.endswith("(max)"):
                        writer.add_scalar(k, np.max(v), timestep)
                    else:
                        writer.add_scalar(k, np.mean(v), timestep)
                tracking_data.clear()

            if timestep % CLI_interval == 0:
                end_time = time.time()
                avg_ep_step = float(np.mean(CLI_track_timesteps)) if CLI_track_timesteps else float("nan")
                avg_q_term = float(np.mean(CLI_track_q_term)) if CLI_track_q_term else float("nan")
                avg_g_term = float(np.mean(CLI_track_g_term)) if CLI_track_g_term else float("nan")
                avg_l_term = float(np.mean(CLI_track_l_term)) if CLI_track_l_term else float("nan")
                avg_q_term_min = float(np.mean(CLI_track_q_term_min)) if CLI_track_q_term_min else float("nan")
                avg_g_term_min = float(np.mean(CLI_track_g_term_min)) if CLI_track_g_term_min else float("nan")
                avg_l_term_min = float(np.mean(CLI_track_l_term_min)) if CLI_track_l_term_min else float("nan")
                avg_g = float(np.mean(CLI_track_g)) if CLI_track_g else float("nan")
                avg_l = float(np.mean(CLI_track_l)) if CLI_track_l else float("nan")

                avg_ep_step_str = "-" if np.isnan(avg_ep_step) else f"{avg_ep_step:6.3f} steps"
                avg_q_term_str = "-" if np.isnan(avg_q_term) else f"{avg_q_term:6.3f}"
                avg_g_term_str = "-" if np.isnan(avg_g_term) else f"{avg_g_term:6.3f}"
                avg_l_term_str = "-" if np.isnan(avg_l_term) else f"{avg_l_term:6.3f}"
                avg_q_term_min_str = "-" if np.isnan(avg_q_term_min) else f"{avg_q_term_min:6.3f}"
                avg_g_term_min_str = "-" if np.isnan(avg_g_term_min) else f"{avg_g_term_min:6.3f}"
                avg_l_term_min_str = "-" if np.isnan(avg_l_term_min) else f"{avg_l_term_min:6.3f}"
                avg_g_str = "-" if np.isnan(avg_g) else f"{avg_g:6.3f}"
                avg_l_str = "-" if np.isnan(avg_l) else f"{avg_l:6.3f}"
                critic_loss_str = "-" if update_info is None else f"{update_info['critic_loss']:8.5f}"
                actor_loss_str = "-" if update_info is None else f"{update_info['actor_loss']:8.5f}"
                alpha_str = "-" if update_info is None else f"{update_info['alpha']:7.4f}"
                gamma_str = f"{agent.discount_factor:7.4f}"

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
                line_header = f"Step Progress {timestep} / {cfg['train']['timesteps']}"
                line_time_header = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
                lines = [
                    f"Rollout Time            : {end_time - start_time:6.3f} sec",
                    f"Value Loss              : {critic_loss_str}",
                    f"Policy Loss             : {actor_loss_str}",
                    f"Alpha                   : {alpha_str}",
                    f"Gamma                   : {gamma_str}",
                    f"Avg g                   : {avg_g_str}",
                    f"Avg l                   : {avg_l_str}",
                    f"Avg Q Termination       : {avg_q_term_str}",
                    f"Avg g Termination       : {avg_g_term_str}",
                    f"Avg l Termination       : {avg_l_term_str}",
                    f"Min Q Termination       : {avg_q_term_min_str}",
                    f"Min g Termination       : {avg_g_term_min_str}",
                    f"Min l Termination       : {avg_l_term_min_str}",
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
                CLI_track_q_term.clear()
                CLI_track_g_term.clear()
                logstep += 1
                start_time = end_time

            if (timestep % checkpoint_interval == 0) and log_dir is not None:
                agent.save(os.path.join(log_dir, f"agent_{timestep}.pt"))

            obs = next_obs
            states = next_states
            reach_values = next_reach_values
            safety_values = next_safety_values
            infos = next_infos

    finally:
        if log_dir is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
