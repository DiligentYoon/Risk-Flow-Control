"""Online rollout evaluation of a trained safety value function."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained safety value function with online rollouts.")
parser.add_argument("--seed", type=int, default=43, help="Evaluation seed.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--rollout_steps", type=int, default=1000)
parser.add_argument("--task", type=str, default="R1-fall-play")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to nominal policy checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, required=True, help="Path to safety value checkpoint.")
parser.add_argument("--algorithm", type=str, default="MAPPO", choices=["PPO", "SAC", "TD3", "MAPPO"])
parser.add_argument("--model", type=str, default="Shared", choices=["MLP", "Shared", "Communet"])

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
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter

import lib
import tasks

from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.model.model_factory import ModelFactory
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.utils.plot_utils import GIFSavePlotter

from tasks.p02_safety_value.utils.evaluator import RolloutEvaluator
from tasks.p02_safety_value.wrappers.safety_wrapper import SafetyEnvWrapper, SafetyEnvRecordVideo


algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None


def main() -> None:
    # ============================= Config ============================= #
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
        pred_cfg = load_cfg_from_registry(args_cli.task, "predictor_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    env_cfg.seed = args_cli.seed
    cfg["agent"]["seed"] = args_cli.seed
    pred_cfg["agent"]["seed"] = args_cli.seed

    predictor_checkpoint = os.path.abspath(args_cli.predictor_checkpoint)
    log_dir = os.path.dirname(predictor_checkpoint)
    os.makedirs(log_dir, exist_ok=True)

    # ============================= Environment ============================= #
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = SafetyEnvWrapper(env)

    # ============================= Nominal Buffer ============================= #
    multi_agent = algorithm == "mappo"
    cfg["models"]["multi_agent"] = multi_agent

    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for off-policy nominal algorithms is not implemented.")

    possible_agents = None

    if multi_agent:
        obs_size = {}
        state_size = {}
        act_size = {}
        buffers = {}

        possible_agents = env._unwrapped.cfg.possible_agents

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

    # ============================= Nominal Model ============================= #
    if model is not None:
        cfg["models"]["model_type"] = model

    model_manager = ModelFactory(cfg=cfg["models"], device=env.device)

    if model_manager.model_class != "mlp":
        raise RuntimeError("Not supported model class.")

    models = model_manager.generate_mlp_models(
        observation_size=obs_size,
        state_size=state_size,
        action_size=act_size,
        possible_agents=possible_agents,
    )

    # ============================= Nominal Agent ============================= #
    if multi_agent:
        if model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO

            agent = MAPPO(
                observation_space=env.observation_space,
                state_space=env.state_space,
                action_space=env.action_space,
                possible_agents=possible_agents,
                model=models,
                buffer=buffers,
                device=env.device,
                cfg=cfg["agent"],
            )

        elif model_manager.model_type == "shared":
            from lib.agent.cooperative_mappo import CooperativeMAPPO

            agent = CooperativeMAPPO(
                observation_space=env.observation_space,
                state_space=env.state_space,
                action_space=env.action_space,
                possible_agents=possible_agents,
                model=models,
                buffer=buffers,
                device=env.device,
                cfg=cfg["agent"],
            )

        else:
            raise RuntimeError("Invalid multi-agent model type.")

    else:
        from lib.agent.ppo import PPO

        agent = PPO(
            model=models,
            buffer=buffer,
            device=env.device,
            cfg=cfg["agent"],
        )

    # ============================= Safety Predictor ============================= #
    from tasks.p02_safety_value.agent.safety import Safety
    from tasks.p02_safety_value.model.safety import SafetyCritic

    num_safety_states = env._unwrapped.num_safety_states
    pred_model = {"critic_1": SafetyCritic(num_states=num_safety_states, device=env.device),
                  "critic_2": SafetyCritic(num_states=num_safety_states, device=env.device)}
    pred_agent = Safety(model=pred_model, device=env.device, cfg=pred_cfg["agent"])

    # ============================= Checkpoints ============================= #
    nominal_checkpoint = os.path.abspath(args_cli.checkpoint)
    agent.load(nominal_checkpoint)
    pred_agent.load(predictor_checkpoint)

    agent.set_running_mode("eval")
    pred_agent.set_running_mode("eval")

    print(f"[INFO] Nominal checkpoint: {nominal_checkpoint}")
    print(f"[INFO] Predictor checkpoint: {predictor_checkpoint}")

    # ============================= Evaluator ============================= #
    evaluator = RolloutEvaluator(
        num_envs=env.num_envs,
        max_segment_length=env._unwrapped.max_episode_length,
        discount_factor=pred_cfg["agent"]["discount_factor"],
        device=env.device,
        threshold=pred_cfg["eval"]["threshold"],
        file_path=os.path.join(log_dir, "safety_rollout.png")
    )

    print(f"[INFO] Number of environments: {env.num_envs}")
    print(f"[INFO] Rollout steps: {args_cli.video_length}")
    print(f"[INFO] Safety-state dimension: {num_safety_states}")
    print(f"[INFO] Risk threshold: {pred_cfg['eval']['threshold']}")
    print(f"[INFO] Evaluation seed: {args_cli.seed}")
    print(f"[INFO] Log directory: {log_dir}")

    # ============================= Online Rollout ============================= #
    obs, states, safety_states, infos = env.reset()

    timestep = 0
    start_time = time.time()
    while simulation_app.is_running():
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

            pred_values = pred_agent.predict(final_safety_states).squeeze(-1)

        evaluator.append(
            g_values=final_safety_values,
            pred_values=pred_values,
            push_events=final_push_events,
            terminated=terminated,
            truncated=truncated,
        )

        timestep += 1

        obs = next_obs
        states = next_states
        safety_states = next_safety_states
        safety_values = next_safety_values
        infos = next_infos

        # CLI Logging Phase
        if timestep % 300 == 0 or timestep == args_cli.video_length:
            metrics = evaluator.compute()
            elapsed_time = time.time() - start_time

            print(
                f"[EVAL] step={timestep}/{args_cli.video_length} | "
                f"segments={metrics['num_segments']} | "
                f"RCR={metrics['risk_coverage_rate']:.4f} | "
                f"RDR={metrics['risk_detection_rate']:.4f} | "
                f"RFAR={metrics['risk_false_alarm_rate']:.4f} | "
                f"Termination_DR={metrics['termination_detection_rate']:.4f} | "
                f"Termination_FAR={metrics['termination_false_alarm_rate']:.4f} | "
                f"PR={metrics['proactive_recall']:.4f} | "
                f"pred_risk={metrics['pred_risk_rate']:.4f} | "
                f"real_risk={metrics['real_risk_rate']:.4f} | "
                f"time={elapsed_time:.1f}s"
            )

        # Video Update Phase
        if timestep == args_cli.video_length:
            # exit the play loop after recording one video
            break

    # ============================= Final Result ============================= #
    metrics = evaluator.compute()

    result = {
        "nominal_checkpoint": nominal_checkpoint,
        "predictor_checkpoint": predictor_checkpoint,
        "seed": args_cli.seed,
        "num_envs": env.num_envs,
        "threshold": pred_cfg["eval"]["threshold"],
        "metrics": metrics,
    }

    result_path = os.path.join(log_dir, "metrics.yaml")
    with open(result_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    print("=" * 80)
    print("ONLINE SAFETY VALUE EVALUATION")
    print("=" * 80)
    print(f"Completed segments           : {metrics['num_segments']}")
    print(f"Risk coverage rate           : {metrics['risk_coverage_rate'] * 100:.2f}% ")
    print(f"Risk Detection rate          : {metrics['risk_detection_rate'] * 100:.2f}% ")
    print(f"Risk False alarm rate        : {metrics['risk_false_alarm_rate'] * 100:.2f}% ")
    print(f"Termination detection rate   : {metrics['termination_detection_rate'] * 100:.2f}% ")
    print(f"Termination false alarm rate : {metrics['termination_false_alarm_rate'] * 100:.2f}% ")
    print(f"Proactive recall             : {metrics['proactive_recall']* 100:.2f}% ")
    print(f"Accuracy                     : {metrics['accuracy'] * 100:.2f}% ")
    print(f"[INFO] Results saved to      : {result_path}")

    evaluator.save_timeseries_plot(step_dt=float(env._unwrapped.step_dt))
    print(f"[INFO] Rollout plot saved to: {evaluator.file_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()