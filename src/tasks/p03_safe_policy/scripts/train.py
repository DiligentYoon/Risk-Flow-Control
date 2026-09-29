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
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="R1-intervention", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Path to safety value network checkpoint.")
parser.add_argument("--initial_dataset", type=str, default=None, help="Path to Risk Initial dataset.")

parser.add_argument("--model",
                    type=str,
                    default="MLP",
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
import collections
import os
import time
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import lib
import tasks

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry

from tasks.p03_safe_policy.wrappers.intervention_wrapper import InterventionEnvWrapper, InterventionEnvRecordVideo

def main():
    # ============================= Config Parsing ===============================
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_single_cfg_entry_point")
        pred_cfg = load_cfg_from_registry(args_cli.task, "predictor_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    env_cfg.seed = args_cli.seed
    cfg["agent"]["seed"] = args_cli.seed
    pred_cfg["agent"]["seed"] = args_cli.seed

    if args_cli.predictor_checkpoint is not None:
        predictor_checkpoint = os.path.abspath(args_cli.predictor_checkpoint)
        log_dir = os.path.join(os.path.dirname(predictor_checkpoint), "intervention_policy", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = None

    # ============================ Env & Wrapper Spawn ================================
    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env_cfg.events.reset_base.params["dataset_path"] = args_cli.initial_dataset
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if (args_cli.video) and log_dir is not None:
        args_cli.video_interval = int(cfg["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = InterventionEnvRecordVideo(env, **video_kwargs)

    # Wrap around environment
    env = InterventionEnvWrapper(env)  

    # Interval
    if cfg["agent"]["experiment"]["write_interval"] == "auto":
        write_interval = int(cfg["train"]["timesteps"] / 100)
    if cfg["agent"]["experiment"]["checkpoint_interval"] == "auto":
        checkpoint_interval = int(cfg["train"]["timesteps"] / 5)

    # ============================= Safety Predictor ============================= #
    from tasks.p02_safety_value.agent.safety import Safety
    from tasks.p02_safety_value.model.safety import SafetyCritic

    num_safety_states = env._unwrapped.num_safety_states
    pred_model = {"critic_1": SafetyCritic(num_states=num_safety_states, device=env.device),
                  "critic_2": SafetyCritic(num_states=num_safety_states, device=env.device)}
    pred_agent = Safety(model=pred_model, device=env.device, cfg=pred_cfg["agent"])

    # ======================= Buffer & Model & Agent =========================
    from tasks.p03_safe_policy.buffer.risk_flow_buffer import RiskFlowBuffer
    from tasks.p03_safe_policy.agents.risk_flow import RiskFlow
    from tasks.p03_safe_policy.models.risk_flow_models import MultiHorizonCritic, DeterministicActor

    buffer = RiskFlowBuffer(buffer_size=cfg["buffer"]["buffer_size"], num_envs=env.num_envs, device=env.device)
    buffer.init_buffer(env.observation_space, env.state_space, env.safety_state_space, env.action_space)

    observation_dim = buffer.tensors["observations"].shape[-1]
    state_dim = buffer.tensors["states"].shape[-1]
    action_dim = buffer.tensors["actions"].shape[-1]
    
    model =  {"critic": MultiHorizonCritic(state_dim, action_dim, cfg["agent"]["horizon"], device=env.device, output_gain=0.01),
              "actor": DeterministicActor(observation_dim, action_dim, device=env.device, output_gain=0.01)}
    agent = RiskFlow(model, buffer, pred_agent, device=env.device, cfg=cfg["agent"])

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

    # ====================== Training =============================== #
    writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
    cumulative_timesteps = None
    tracking_data = collections.defaultdict(list)
    tracking_timesteps = collections.deque(maxlen=env.num_envs)
    CLI_track_timesteps = collections.deque(maxlen=env.num_envs)

    obs, states, safety_states, infos = env.reset()

    timestep = 0
    elapsed_time = 0
    start_time = time.time()
    CLI_interval = 100

    try:
        while (simulation_app.is_running() and timestep < cfg["train"]["timesteps"]):
            with torch.inference_mode():
                actions = agent.act(obs, deterministic=False)

                (
                    next_obs,
                    next_states,
                    next_safety_states,
                    next_safety_values,
                    final_obs,
                    final_states,
                    final_safety_states,
                    final_safety_values,
                    rewards,
                    terminated,
                    truncated,
                    next_infos,
                ) = env.step(actions)

            timestep += 1

            agent.insert_data(
                observations=obs,
                states=states,
                safety_states=safety_states,
                actions=actions,
                final_observations=final_obs,
                final_states=final_states,
                final_safety_states=final_safety_states,
                terminated=terminated,
                truncated=truncated
            )

            # ================== Learning Phase =====================
            info = agent.update()

            # =============== Logging Phase ================
            if info is not None:
                if not np.isfinite(info["critic_loss"]):
                    print(f"The critic loss diverges at step {timestep}.")
                    break
                value_loss = info["critic_loss"]
                tracking_data["Loss / critic"].append(value_loss)

                if "actor_loss" in info:
                    policy_loss = info["actor_loss"]
                    flow_mean = info["flow_mean"]
                    tracking_data["Loss / actor"].append(policy_loss)
                    tracking_data["Policy / flow mean"].append(flow_mean)
                else:
                    policy_loss = None
                    flow_mean = None

            with torch.no_grad():
                value = pred_agent.predict(safety_states)
                next_value = pred_agent.predict(final_safety_states)
                delta = next_value - value
            tracking_data["Value / Delta_N"].append(delta.mean().item())

            if cumulative_timesteps is None:
                cumulative_timesteps = torch.zeros((env.num_envs, 1), dtype=torch.int32)
            cumulative_timesteps.add_(1)

            done = (terminated | truncated).squeeze(-1)
            finished_episodes = done.nonzero(as_tuple=False).squeeze(-1).cpu()
            if finished_episodes.numel():
                tracking_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())
                CLI_track_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].detach().cpu().tolist())
                cumulative_timesteps[finished_episodes] = 0
            if len(tracking_timesteps):
                tracking_timesteps_np = np.array(tracking_timesteps)
                tracking_data["Episode / Total timesteps (mean)"].append(np.mean(tracking_timesteps_np))
                tracking_timesteps.clear()
                
            # Tensorboard logging
            if (timestep % write_interval == 0) and log_dir is not None: 
                for k, v in tracking_data.items():
                    if k.endswith("(min)"):
                        writer.add_scalar(k, np.min(v), timestep)
                    elif k.endswith("(max)"):
                        writer.add_scalar(k, np.max(v), timestep)
                    else:
                        writer.add_scalar(k, np.mean(v), timestep)
                # reset data containers for next iteration
                tracking_data.clear()

            # CLI Logging about the training process at each parameter update
            if timestep % CLI_interval == 0:
                end_time = time.time()
                avg_ep_step = float(np.mean(CLI_track_timesteps)) if len(CLI_track_timesteps) else float("nan")

                avg_ep_step_str = "-" if np.isnan(avg_ep_step) else f"{avg_ep_step:6.3f} steps"
                value_loss_str = f"{value_loss:6.3f}"
                policy_loss_str = "-" if policy_loss is None else f"{policy_loss:6.3f}"
                flow_mean_str = "-" if flow_mean is None else f"{flow_mean:6.3f}"

                elapsed_time += (end_time - start_time)
                e_h = int(elapsed_time // 3600)
                e_m = int((elapsed_time % 3600) // 60)
                e_s = int(elapsed_time % 60)
                total_rollout = int(cfg["train"]["timesteps"] // CLI_interval)
                complete_time = (end_time - start_time) * total_rollout
                c_h = int(complete_time // 3600)
                c_m = int((complete_time % 3600) // 60)
                c_s = int(complete_time % 60)

                content_width = 64
                line_header = f"Step Progress {timestep} / {cfg['train']['timesteps']}"
                line_time_header = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
                line_rollout_time = f"Rollout Time      : {end_time - start_time:6.3f} sec"
                line_value_loss = f"Value Loss        : {value_loss_str}"
                line_policy_loss = f"Policy Loss       : {policy_loss_str}"
                line_flow_mean = f"Flow Mean        : {flow_mean_str}"
                line_episode_step = f"Avg Episode Step  : {avg_ep_step_str}"

                print(f" ________________________________________________________________")
                print(f"|                                                                |")
                print(f"|{line_header.center(content_width)}|")
                print(f"|{line_time_header.center(content_width)}|")
                print(f"|________________________________________________________________|")
                print(f"|                                                                |")
                print(f"| {line_rollout_time:<{content_width-1}}|")
                print(f"| {line_value_loss:<{content_width-1}}|")
                print(f"| {line_policy_loss:<{content_width-1}}|")
                print(f"| {line_flow_mean:<{content_width-1}}|")
                print(f"| {line_episode_step:<{content_width-1}}|")
                print(f"|________________________________________________________________|")

                start_time = end_time

            # Checkpoint save
            if (timestep % checkpoint_interval == 0) and log_dir is not None:
                checkpoint_path = os.path.join(log_dir, f"agent_{timestep}.pt")
                agent.save(checkpoint_path)

            # update policy inputs
            obs = next_obs
            states = next_states
            safety_states = next_safety_states
            safety_values = next_safety_values
            infos = next_infos

    finally:
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()