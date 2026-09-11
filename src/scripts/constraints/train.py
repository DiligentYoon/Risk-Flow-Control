# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Risk-Flow actor-critic intervention policy.

Two frozen assets have to come from elsewhere and the script refuses to guess at either:

* ``--predictor_checkpoint`` -- the safety value function of Phase 0. It *is* the learning signal here,
  so an untrained one would train the critic against noise while every curve still looked healthy.
* the risk-bucket reset dataset -- the initial-state distribution rho_I of research.md 2.4. Without
  it every episode starts from the default pose, which is not a state worth intervening in.

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Risk-Flow intervention policy.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the RL environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-risk", help="Name of the task.")
parser.add_argument("--timesteps", type=int, default=None, help="Override the number of training steps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a RiskFlow checkpoint to resume from.")
parser.add_argument("--predictor_checkpoint", type=str, default="/home/aisl/Repos/Risk-Flow-Control/logs/frozen/network/Reach_Avoid/2026-09-09_16-22-22/ra_agent_32000.pt", help="Path to the frozen predictor checkpoint.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import collections
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import lib
import envs 

from lib.utils.parse_utils import load_cfg_from_registry, parse_env_cfg

from buffer.risk_flow_buffer import RiskFlowBuffer
from scripts.utils import build_agent, build_models, build_predictor, resolve_horizon, write_tracking, print_progress
from wrappers.constraints_wrapper import ConstraintsRecordVideo, ConstraintsWrapper

def summarize(info) -> list:
    """The four numbers worth watching from the terminal, in the order they can go wrong.

    A diverging critic invalidates the actor loss, which in turn invalidates the multiplier, so
    reading them top to bottom is reading the dependency chain.
    """
    def field(label, key, fmt):
        if info is None or key not in info:
            return f"{label:<14}: -"
        return f"{label:<14}: {info[key]:{fmt}}"

    return [
        field("critic loss", "critic_loss", ".5f"),
        field("actor loss", "actor_loss", ".5f"),
        field("lambda", "lambda", ".5f"),
        field("violation g", "constraint_violation", ".5f"),
    ]


def main():
    # ============================= Config Parsing ===============================
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, "rl_risk_flow_cfg_entry_point")
    except ValueError as error:
        print(error)
        return 1

    seed = args_cli.seed if args_cli.seed is not None else cfg.get("seed", 42)
    env_cfg.seed = seed
    cfg["agent"]["seed"] = seed
    cfg["predictor"]["agent"]["seed"] = seed

    timesteps = args_cli.timesteps if args_cli.timesteps is not None else cfg["train"]["timesteps"]

    log_dir = os.path.join(os.getcwd(), "logs", cfg["agent"]["experiment"]["directory"],
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # ============================ Env & Wrapper Spawn ================================
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        args_cli.video_interval = int(cfg["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = ConstraintsRecordVideo(env, **video_kwargs)

    env = ConstraintsWrapper(env)

    write_interval = cfg["agent"]["experiment"]["write_interval"]
    checkpoint_interval = cfg["agent"]["experiment"]["checkpoint_interval"]
    if write_interval == "auto":
        write_interval = max(1, int(timesteps / 100))
    if checkpoint_interval == "auto":
        checkpoint_interval = max(1, int(timesteps / 5))

    # ======================= Buffer / Models / Agent =========================
    horizon = resolve_horizon(cfg["agent"]["horizon"], env)
    cfg["agent"]["horizon"] = horizon

    buffer = RiskFlowBuffer(buffer_size=cfg["buffer"]["buffer_size"], num_envs=env.num_envs, device=env.device)
    buffer.init_buffer(env.observation_space, env.constraint_state_space, env.action_space)

    observation_dim = buffer.tensors["observations"].shape[-1]
    constraint_state_dim = buffer.tensors["constraint_states"].shape[-1]
    action_dim = buffer.tensors["actions"].shape[-1]

    value_critic = build_predictor(cfg["predictor"]["agent"], constraint_state_dim, env.device, checkpoint=args_cli.predictor_checkpoint)

    model = build_models(cfg.get("models", {}), observation_dim, constraint_state_dim, action_dim, horizon, env.device)
    
    agent = build_agent(cfg["agent"], model, buffer, value_critic, env.get_torque_model(), device=env.device, checkpoint=args_cli.checkpoint)

    head_indices = sorted({min(horizon - 1, int(round(fraction * (horizon - 1)))) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)})

    # ======================= Training ============================
    writer = SummaryWriter(log_dir=log_dir)
    tracking_data = collections.defaultdict(list)
    track_timesteps = collections.deque(maxlen=env.num_envs)
    CLI_track_timesteps = collections.deque(maxlen=env.num_envs)

    # The per-head losses are summed on the device and averaged at the write interval.
    per_head_sum = torch.zeros(horizon, device=env.device)
    per_head_count = 0

    info = None
    timestep = 0
    start_time = time.time()
    print_interval = horizon
    gradient_steps = cfg["agent"].get("gradient_steps", 1)

    agent.set_running_mode("train")
    obs, _, constraint_states, infos = env.reset()

    while simulation_app.is_running() and timestep < timesteps:

        # ================== Interaction Phase =====================
        actions = agent.act(obs)
        next_obs, _, next_constraint_states, _, terminated, truncated, next_infos = env.step(actions)

        if not torch.all(torch.isfinite(obs)):
            print(f"The observation diverges at timestep {timestep}")
            break

        if not torch.all(torch.isfinite(actions)):
            print(f"The action diverges at timestep {timestep}")
            break
        
        # It is the snapshot that carries the state the action actually led to.
        final_observations = next_infos["final_observations"]
        final_constraint_states = next_infos["final_constraint_states"]

        agent.insert_data(
            observations=obs,
            constraint_states=constraint_states,
            actions=actions,
            final_observations=final_observations,
            final_constraint_states=final_constraint_states,
            terminated=terminated,
            truncated=truncated,
        )
        timestep += 1

        with torch.no_grad():
            value, _, _ = value_critic(constraint_states)
            final_value, _, _ = value_critic(final_constraint_states)
            delta = final_value - value

        tracking_data["Episode / truncated rate"].append(int(truncated.sum().item()) / env.num_envs)
        tracking_data["Value / Delta_N"].append(delta.mean().item())
        tracking_data["Value / Delta_N std"].append(delta.std().item())

        # ================== Learning Phase =====================
        # One gradient step per environment step.
        for _ in range(gradient_steps):
            info = agent.update()

        if info is not None:
            if not np.isfinite(info["critic_loss"]):
                print(f"The critic loss diverges at step {timestep}.")
                break

            tracking_data["Loss / critic"].append(info["critic_loss"])
            per_head_sum += info["per_head_loss"]
            per_head_count += 1

            if "actor_loss" in info:
                tracking_data["Loss / actor"].append(info["actor_loss"])
                tracking_data["Policy / control cost"].append(info["control_cost"])
                tracking_data["Policy / flow mean"].append(info["flow_mean"])

            if "lambda" in info:
                tracking_data["Constraint / lambda"].append(info["lambda"])
                tracking_data["Constraint / nu"].append(agent.lagrange.nu.item())
                tracking_data["Constraint / violating ratio"].append(info["violation_ratio"])

        # =============== Logging Phase ================
        if timestep % write_interval == 0:
            if per_head_count > 0:
                per_head_loss = (per_head_sum / per_head_count).cpu().numpy()
                for head in head_indices:
                    writer.add_scalar(f"Loss / critic head {head + 1:03d}", per_head_loss[head], timestep)
                per_head_sum.zero_()
                per_head_count = 0

            write_tracking(writer, tracking_data, timestep)

        # CLI progress, on the same cadence as the checkpoints
        if timestep % print_interval == 0 or timestep == timesteps:
            print_progress(timestep, timesteps, start_time, summarize(info))

        # Checkpoint save
        if timestep % checkpoint_interval == 0:
            agent.save(os.path.join(log_dir, f"agent_{timestep}.pt"))

        obs = next_obs
        constraint_states = next_constraint_states
        infos = next_infos

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
