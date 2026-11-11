"""Online rollout evaluation of a trained Risk-Flow intervention policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="R1-safe-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default="logs/r1_loco/2026-10-29_17-29-17_mappo/Predictor/2026-10-29_22-54-23/2026-10-29_23-01-25/safe_policy/2026-11-11_14-43-24_ppo/agent_52000.pt", help="Path to model checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default="logs/r1_loco/2026-10-29_17-29-17_mappo/Predictor/2026-10-29_22-54-23/2026-10-29_23-01-25/agent_10000.pt", help="Path to safety value network checkpoint.")
parser.add_argument("--initial_dataset", type=str, default="logs/r1_loco/2026-10-29_17-29-17_mappo/Predictor/2026-10-29_22-54-23/2026-10-29_23-01-25/collected/risk_init_dataset.pt", help="Path to Risk Initial dataset.")

parser.add_argument("--algorithm",
                    type=str,
                    default="PPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the agent.")

parser.add_argument("--model",
                    type=str,
                    default="MLP",
                    choices=["MLP", "Shared"],
                    help="The NN model used for training the agent.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
import gymnasium as gym
import numpy as np
import torch
import collections
from datetime import datetime

import lib
import tasks

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.utils.plot_utils import GIFSavePlotter

from tasks.p02_safety_value.wrappers.safety_wrapper import SafetyEnvWrapper, SafetyEnvRecordVideo

# config shortcuts
algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None

def main():
    # ============================= Config Parsing =============================
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"ra_{algorithm}_cfg_entry_point")
        pred_cfg = load_cfg_from_registry(args_cli.task, "predictor_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
        pred_cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", 42)
        cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
        pred_cfg["agent"]["seed"] = cfg.get("seed", 42)

    if args_cli.checkpoint is not None:
        checkpoint = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(checkpoint)
    else:
        checkpoint = None
        log_dir = None

    if args_cli.predictor_checkpoint is not None:
        predictor_checkpoint = os.path.abspath(args_cli.predictor_checkpoint)
    else:
        predictor_checkpoint = None

    # ============================ Env & Wrapper Spawn ================================
    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env_cfg.events.reset_base.params["dataset_path"] = args_cli.initial_dataset
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if (args_cli.video) and log_dir is not None:
        args_cli.video_interval = int(cfg["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = SafetyEnvRecordVideo(env, **video_kwargs)

    # Wrap around environment
    env = SafetyEnvWrapper(env)  

    # ============================= Safety Predictor ============================= #
    from tasks.p02_safety_value.agent.safety import Safety
    from tasks.p02_safety_value.model.safety import SafetyCritic

    num_safety_states = env._unwrapped.num_safety_states
    pred_model = {"critic_1": SafetyCritic(num_states=num_safety_states, device=env.device),
                  "critic_2": SafetyCritic(num_states=num_safety_states, device=env.device)}
    pred_agent = Safety(model=pred_model, device=env.device, cfg=pred_cfg["agent"])

    # ======================= Model Factory (Model & Buffer) =========================
    from tasks.p03_safe_policy.models.model_factory import ReachAvoidModelFactory
    multi_agent = algorithm == "mappo"
    cfg["models"]["multi_agent"] = multi_agent
    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")

    model_factory = ReachAvoidModelFactory(buffer_cfg=cfg["buffer"],
                                           model_cfg=cfg["models"],
                                           device=env.device)
    buffer = model_factory.generate_buffers(env=env)
    models = model_factory.generate_models()

    # ====================== Agent Spawn  ==========================
    if multi_agent:
        raise RuntimeError("Not yet!")
    else:
        from tasks.p03_safe_policy.agents.reach_avoid_ppo import ReachAvoidPPO
        agent = ReachAvoidPPO(model=models,
                              buffer=buffer, 
                              device=env.device,
                              cfg=cfg["agent"])

    # ============================= Checkpoints ============================= #
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get checkpoint of policy from {resume_path}")
    else:
        print("[INFO] Unfortunately a pre-trained policy is not found for this task.")

    if args_cli.predictor_checkpoint is not None:
        pred_resume_path = os.path.abspath(args_cli.predictor_checkpoint)
        pred_agent.load(predictor_checkpoint)
        print(f"[INFO] Get checkpoint of value network from {pred_resume_path}")
    else:
        print("[INFO] Unfortunately a pre-trained value network is not found for this task.")

    # ======================================================================================================================
    # ======================================== Env Interaction Test ========================================================
    # ======================================================================================================================

    # reset environment
    plotter_cls = getattr(env._unwrapped.cfg, "plotter", None)
    if plotter_cls is not None:
        plot_cfg = env._unwrapped.cfg.viz_data
        plot_dir = os.path.join(log_dir, "plot") if args_cli.checkpoint else None
        plot: GIFSavePlotter = plotter_cls(env, plot_cfg, plot_dir)
    else:
        plot = None

    num_finished = 0
    num_terminated = 0
    tracking_data = collections.defaultdict(list)
    track_cumulative_rewards   = collections.deque(maxlen=500)
    track_cumulative_timesteps = collections.deque(maxlen=500)
    cumulative_rewards = None
    cumulative_timesteps = None

    obs, states, safety_states, safety_values, infos = env.reset()
    write_interval = env._unwrapped.max_episode_length
    timestep = 0

    anchor_safety_value = pred_agent.predict(safety_states)
    next_pred_safety_value    = torch.zeros_like(anchor_safety_value, dtype=torch.float32, device=env.device)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.no_grad():
            # agent stepping
            actions,  _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            # env stepping
            (
                next_obs,
                next_states,
                next_safety_states,
                next_safety_values,
                final_states,
                final_safety_states,
                final_safety_values,
                rewards,
                terminated,
                truncated,
                final_push_events,
                next_infos,
            ) = env.step(actions)
            # safety value-based reward
            current_pred_reach_avoid_value, _, _ = agent.ra_critic(states)
            current_pred_reach_avoid_value = agent.ra_standardizer.destandardize(current_pred_reach_avoid_value)
            current_pred_safety_value = pred_agent.predict(safety_states)
            next_pred_safety_value = pred_agent.predict(final_safety_states)
            delta_safety_reward =  (anchor_safety_value - next_pred_safety_value).clamp(min=-1, max=1)
            pred_error_penalty = -torch.abs(next_pred_safety_value - final_safety_values)
            # add safety value-based reward
            if multi_agent:
                rewards = (rewards.reshape(-1, model_factory.num_agents) 
                           + env._unwrapped.cfg.r_delta_safety * delta_safety_reward
                           + env._unwrapped.cfg.p_pred_error * pred_error_penalty).reshape(-1, 1)
            else:
                rewards += (env._unwrapped.cfg.r_delta_safety * delta_safety_reward
                            + env._unwrapped.cfg.p_pred_error * pred_error_penalty)
            # Add delta safety reward to info dict for tensorboard logging
            next_infos["reward"]["Task Reward / Common_Delta_Safety_Reward"] = delta_safety_reward
            next_infos["reward"]["Task Penalty / Common_Prediction_Error"] = pred_error_penalty
                
            timestep += 1

        # ============ Logging phase =============

        # Data setting for logging
        if multi_agent:
            logged_reward = rewards.view(-1, model_factory.num_agents)
            logged_reward = torch.mean(logged_reward, dim=-1).unsqueeze(-1) # Mean value of agent axis
        else:
            logged_reward = rewards

        if cumulative_rewards is None:
            cumulative_rewards = torch.zeros_like(logged_reward, dtype=torch.float32)
            cumulative_timesteps = torch.zeros_like(logged_reward, dtype=torch.int32)

        # Record data
        tracking_data["Per step Reward / Maximum Value"].append(torch.max(logged_reward).item())
        tracking_data["Per step Reward / Minimum Value"].append(torch.min(logged_reward).item())
        tracking_data["Per step Reward / Mean Value"].append(torch.mean(logged_reward).item())

        # Accumulates per-step rewards
        cumulative_rewards.add_(logged_reward)
        cumulative_timesteps.add_(1)

        done = (terminated | truncated).squeeze(-1)
        finished_episodes = done.nonzero(as_tuple=False).squeeze(-1)
        if finished_episodes.numel():
            # Storage cumulative rewards and timesteps
            track_cumulative_rewards.extend(cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            track_cumulative_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())
            # Reset the cumulative rewards and timesteps
            cumulative_rewards[finished_episodes] = 0
            cumulative_timesteps[finished_episodes] = 0

            num_terminated += (terminated.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)).numel()
            num_finished += finished_episodes.numel()

        # Record cumulative data
        if len(track_cumulative_rewards):
            track_reward_np = np.array(track_cumulative_rewards)
            track_timestep_np = np.array(track_cumulative_timesteps)

            tracking_data["Total Reward / Maximum Value"].append(np.max(track_reward_np))
            tracking_data["Total Reward / Minimum Value"].append(np.min(track_reward_np))
            tracking_data["Total Reward / Mean Value"].append(np.mean(track_reward_np))

            tracking_data["Total Episode / Maximum Value"].append(np.max(track_timestep_np))
            tracking_data["Total Episode / Minimum Value"].append(np.min(track_timestep_np))
            tracking_data["Total Episode / Mean Value"].append(np.mean(track_timestep_np))

            # reset data containers for next iteration
            track_cumulative_rewards.clear()
            track_cumulative_timesteps.clear()

        # Record Task Reward data (Only per step)
        task_reward = next_infos.get("reward", None)
        if task_reward is not None:
            for k, v in task_reward.items():
                # Mean value of env axis
                key = f"Per step {k}"
                tracking_data[key].append(torch.mean(v).item())

        # CLI Logging with specific interval
        if timestep % write_interval == 0:
            last_prefix = None
            # sorted_keys = sorted(tracking_data.keys())
            content_width = 90
            line_header = f"Metric Table of {args_cli.task} Task"
            print(f" {'_' * content_width}")
            print(f"|{' ' * content_width}|")
            print(f"|{line_header.center(content_width)}|")
            print(f"|{'_' * content_width}|")
            print(f"|{' ' * content_width}|")

            for k, v in tracking_data.items():
                current_prefix = k.split('/')[0].strip() if '/' in k else "Other"

                if last_prefix is not None and last_prefix != current_prefix:
                    print(f"|{'-' * (content_width)}|")
                last_prefix = current_prefix

                if k.endswith("(min)"):
                    print(f"| {k:<50}: {np.min(v):<36.3f} |")
                elif k.endswith("(max)"):
                    print(f"| {k:<50}: {np.max(v):<36.3f} |")
                else:
                    print(f"| {k:<50}: {np.mean(v):<36.3f} |")
            print(f"|{'_' * content_width}|")

            # reset data containers for next iteration
            tracking_data.clear()

        # Plot Phase
        if plot is not None:
            # Add Safety value for plotting
            infos["viz_data"]["real_risk_value"] = safety_values
            infos["viz_data"]["pred_risk_value"] = current_pred_safety_value
            infos["viz_data"]["reach_avoid_value"] = current_pred_reach_avoid_value
            infos["viz_data"]["prediction_error"] = torch.abs(current_pred_safety_value - safety_values)
            done_env0 = terminated[0] | truncated[0]
            plot.append(viz_data=infos["viz_data"], episode_end=done_env0)

        # Video update
        if timestep == args_cli.video_length:
            # exit the play loop after recording one video
            break
        
        # state update
        obs = next_obs
        states = next_states
        safety_states = next_safety_states
        safety_values = next_safety_values
        infos = next_infos
    
        # update anchor value
        if finished_episodes.numel():
            anchor_safety_value[finished_episodes] = pred_agent.predict(next_safety_states[finished_episodes])


    line_header = f"Final Metrics of {args_cli.task} Task"
    print(f" {'_' * content_width}")
    print(f"|{' ' * content_width}|")
    print(f"|{line_header.center(content_width)}|")
    print(f"|{'_' * content_width}|")
    print(f"|{' ' * content_width}|")
    
    print(f"| {'Success Rate':<50}: {100*(1- num_terminated / num_finished):<36.3f} |")

    print(f"|{'_' * content_width}|")

    # close the simulator
    env.close()

    # close and save GIF plotter
    if plot is not None:
        plot.save()
        plot.close()


if __name__ == "__main__":
    main()
    simulation_app.close()