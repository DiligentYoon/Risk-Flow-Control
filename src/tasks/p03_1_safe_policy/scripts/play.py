"""Online rollout evaluation of a trained Risk-Flow intervention policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained Risk-Flow intervention policy.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=500, help="Length of evaluation rollout in steps.")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="R1-safe-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to Risk-Flow checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Path to safety value network checkpoint.")
parser.add_argument("--initial_dataset", type=str, default=None, help="Path to Risk Initial dataset.")

parser.add_argument("--algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the agent.")

parser.add_argument("--model",
                    type=str,
                    default="Shared",
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
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
        pred_cfg = load_cfg_from_registry(args_cli.task, "predictor_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    env_cfg.seed = args_cli.seed
    cfg["agent"]["seed"] = args_cli.seed
    pred_cfg["agent"]["seed"] = args_cli.seed

    if args_cli.checkpoint is not None:
        checkpoint = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(checkpoint)
    else:
        log_dir = None

    # ============================= Env & Wrapper =============================
    env_cfg.total_timesteps = args_cli.video_length
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

    # ============================= Safety Predictor =============================
    from tasks.p02_safety_value.agent.safety import Safety
    from tasks.p02_safety_value.model.safety import SafetyCritic

    num_safety_states = env._unwrapped.num_safety_states
    pred_model = {"critic_1": SafetyCritic(num_states=num_safety_states, device=env.device),
                  "critic_2": SafetyCritic(num_states=num_safety_states, device=env.device)}
    pred_agent = Safety(model=pred_model, device=env.device, cfg=pred_cfg["agent"])

    # ======================= Buffer =========================
    from lib.buffer.rolloutbuffer import RolloutBuffer
    from lib.model.model_factory import ModelFactory
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
    model = args_cli.model.lower() if args_cli.model is not None else None
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

    # ============================= Checkpoints =============================
    if args_cli.checkpoint is not None:
        agent.load(os.path.abspath(args_cli.checkpoint))
        print(f"[INFO] Get checkpoint of safe policy from {args_cli.checkpoint}")
    if args_cli.predictor_checkpoint is not None:
        pred_agent.load(os.path.abspath(args_cli.predictor_checkpoint))
        print(f"[INFO] Get checkpoint of safety value network from {args_cli.predictor_checkpoint}")

    agent.set_running_mode("eval")
    pred_agent.set_running_mode("eval")

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
                final_safety_states,
                final_safety_values,
                rewards,
                terminated,
                truncated,
                final_push_events,
                next_infos,
            ) = env.step(actions)
            # safety value-based reward
            next_pred_safety_value = pred_agent.predict(final_safety_states)
            delta_safety_reward =  (anchor_safety_value - next_pred_safety_value).clamp(min=-1, max=1)
            pred_error_penalty = -torch.abs(next_pred_safety_value - final_safety_values)
            # add safety value-based reward
            if multi_agent:
                rewards = (rewards.reshape(-1, num_agent) 
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
            logged_reward = rewards.view(-1, num_agent)
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
            infos["viz_data"]["real_risk_value"] = final_safety_values
            infos["viz_data"]["pred_risk_value"] = next_pred_safety_value
            infos["viz_data"]["prediction_error"] = pred_error_penalty
            done = terminated[0] | truncated[0]
            plot.append(viz_data=infos["viz_data"], episode_end=done)

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


    # close the simulator
    env.close()

    # close and save GIF plotter
    if plot is not None:
        plot.save()
        plot.close()


if __name__ == "__main__":
    main()
    simulation_app.close()