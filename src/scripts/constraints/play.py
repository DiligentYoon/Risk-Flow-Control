# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of the Risk-Flow intervention policy.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of the Risk-Flow intervention policy.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-risk-play", help="Name of the task.")
parser.add_argument("--timesteps", type=int, default=None, help="Override the number of evaluation steps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Path to the frozen predictor checkpoint.")

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
import torch
import numpy as np
import collections

from datetime import datetime

import lib
import envs

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.utils.plot_utils import GIFSavePlotter

from scripts.utils import build_agent, build_models, build_predictor, resolve_horizon
from wrappers.constraints_wrapper import ConstraintsRecordVideo, ConstraintsWrapper

def main():
    """
    main evaluation method
    """

    # ============================= Config Parsing ===============================
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, "rl_risk_flow_cfg_entry_point")
    except ValueError as e:
        print(e)
        return 1

    # ============================ Env & Wrapper Spawn ================================
    # create isaac environment
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
        cfg["predictor"]["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", None)
        cfg["agent"]["seed"] = cfg.get("seed", 42)              # 42 is a default seed (equal to env)
        cfg["predictor"]["agent"]["seed"] = cfg.get("seed", 42)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Logging directory
    if args_cli.checkpoint is not None:
        log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    else:
        log_root_path = os.path.join("runs", cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir += f"_{cfg['agent']['experiment']['experiment_name']}"
        log_dir = os.path.join(log_root_path, log_dir)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playing.")
        env = ConstraintsRecordVideo(env, **video_kwargs)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment
    env = ConstraintsWrapper(env)

    # nothing is written or checkpointed while playing
    cfg["agent"]["experiment"]["write_interval"] = 0
    cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    timesteps = args_cli.timesteps if args_cli.timesteps is not None else cfg["train"]["timesteps"]

    # ====================== Model & Agent Spawn  ==========================
    horizon = resolve_horizon(cfg["agent"]["horizon"], env)
    cfg["agent"]["horizon"] = horizon

    observation_dim = env.observation_space.shape[-1]
    constraint_state_dim = env.constraint_state_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    value_critic = build_predictor(cfg["predictor"]["agent"], constraint_state_dim, env.device, checkpoint=args_cli.predictor_checkpoint)

    model = build_models(cfg.get("models", {}), observation_dim, constraint_state_dim, action_dim, horizon, env.device)

    agent = build_agent(cfg["agent"], model, None, value_critic, env.get_torque_model(), device=env.device, checkpoint=args_cli.checkpoint)

    if args_cli.checkpoint is None:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")

    threshold = float(cfg["agent"]["terminal_risk_threshold"])

    # ======================= Evaluation ============================

    # reset environment
    plotter_cls = getattr(env._unwrapped.cfg, "plotter", None)
    if plotter_cls is not None:
        plot_cfg = env._unwrapped.cfg.viz_data
        plot_dir = os.path.join(log_dir, "plot") if args_cli.checkpoint else None
        plot: GIFSavePlotter = plotter_cls(env, plot_cfg, plot_dir)
    else:
        plot = None

    agent.set_running_mode("eval")
    obs, _, constraint_states, infos = env.reset()
    write_interval = int(env._unwrapped.cfg.episode_length_s / (env._unwrapped.cfg.sim_dt * env._unwrapped.cfg.decimation))
    timestep = 0
    tracking_data = collections.defaultdict(list)

    episode_steps       = torch.zeros(env.num_envs, device=env.device)
    episode_cost        = torch.zeros(env.num_envs, device=env.device)
    
    episode_initial_v   = torch.zeros(env.num_envs, device=env.device)
    episode_predicted   = torch.zeros(env.num_envs, device=env.device)
    episode_started     = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    track_steps       = collections.deque(maxlen=500)
    track_recovered   = collections.deque(maxlen=500)
    track_fell        = collections.deque(maxlen=500)
    track_cost        = collections.deque(maxlen=500)
    track_predicted   = collections.deque(maxlen=500)
    track_realized    = collections.deque(maxlen=500)

    # simulate environment
    while simulation_app.is_running() and timestep < timesteps:

        # ================== Interaction Phase =====================
        with torch.no_grad():
            # agent stepping
            actions = agent.act(obs, deterministic=True)

            # frozen V_N and the critic it is compared against
            risk_value, _, _ = value_critic(constraint_states)
            risk_value = risk_value.squeeze(-1) # V_N
            risk_flow = agent.critic(constraint_states, actions)[:, -1] # D_H
            control_cost = agent.control_cost(constraint_states, actions)

            # env stepping
            next_obs, _, next_constraint_states, _, terminated, truncated, next_infos = env.step(actions)
            # update rollout number
            timestep += 1

        terminated = terminated.squeeze(-1)
        truncated = truncated.squeeze(-1)
        done = terminated | truncated

        # The pre-reset snapshot is the state the action actually led to; on a terminal step the
        # step tuple already carries the first state of the *next* episode instead.
        with torch.no_grad():
            next_value, _, _ = value_critic(next_infos["final_constraint_states"])
        next_value = next_value.squeeze(-1) # V_N(s_{t+1})

        # ============ Logging phase =============

        # Per-step signals
        tracking_data["Per step Risk / D_H"].append(torch.mean(risk_flow).item())
        tracking_data["Per step Policy / Control Cost"].append(torch.mean(control_cost).item())

        # Episode accumulation.
        fresh = ~episode_started # New Episode
        if fresh.any():
            # initial value setting
            episode_initial_v = torch.where(fresh, risk_value, episode_initial_v)
            # initial prediction value setting
            episode_predicted = torch.where(fresh, risk_flow, episode_predicted)
            # Start marking
            episode_started |= fresh

        episode_steps += 1.0
        episode_cost += control_cost

        finished_episodes = done.nonzero(as_tuple=False).squeeze(-1) # End Episode
        if finished_episodes.numel():
            # episode steps
            track_steps.extend(episode_steps[finished_episodes].float().reshape(-1).tolist())
            # whether move from risk to safe
            track_recovered.extend((next_value[finished_episodes] <= threshold).float().reshape(-1).tolist())
            # terminated episode
            track_fell.extend(terminated[finished_episodes].float().reshape(-1).tolist())
            # control cost
            track_cost.extend((episode_cost[finished_episodes] / episode_steps[finished_episodes]).reshape(-1).tolist())
            # \hat{V_N(s_H) - V_N(s_0)}
            track_predicted.extend(episode_predicted[finished_episodes].reshape(-1).tolist())
            # V_N(s_{H}) - V_N(s_0)
            track_realized.extend((next_value[finished_episodes] - episode_initial_v[finished_episodes]).reshape(-1).tolist())

            # Reset the accumulators of the environments that just started a new episode
            episode_steps[finished_episodes] = 0.0
            episode_cost[finished_episodes] = 0.0
            episode_started[finished_episodes] = False

        # Record cumulative data
        if len(track_steps):
            tracking_data["Episode / Episode Steps"].append(np.mean(track_steps))
            tracking_data["Episode / Recovery Success Rate"].append(np.mean(track_recovered))
            tracking_data["Episode / Fall Rate"].append(np.mean(track_fell))
            tracking_data["Episode / Control Cost"].append(np.mean(track_cost))

            # Critic over-estimation: D_H is the head that telescopes to V_N(s_H) - V_N(s_0).
            predicted_np = np.array(track_predicted)
            realized_np = np.array(track_realized)
            tracking_data["Prediction / D_H Prediction Error"].append(np.mean(np.abs(predicted_np - realized_np)))

            # reset data containers for next iteration
            track_steps.clear()
            track_recovered.clear()
            track_fell.clear()
            track_cost.clear()
            track_predicted.clear()
            track_realized.clear()

        # CLI Logging with specific interval
        if timestep % write_interval == 0:
            last_prefix = None
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
            done_0 = terminated[0] | truncated[0]
            infos["viz_data"]["risk_value"] = risk_value
            infos["viz_data"]["risk_flow"] = risk_flow
            # infos["viz_data"]["terminal_risk"] = risk_value + risk_flow - threshold
            plot.append(viz_data=infos["viz_data"], episode_end=done_0)

        # Video update
        if args_cli.video and timestep == args_cli.video_length:
            break

        # update
        obs = next_obs
        constraint_states = next_constraint_states
        infos = next_infos

    # close the simulator
    env.close()

    # close and save PNG plotter
    if plot is not None:
        plot.save()
        plot.close()

    return 0


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()