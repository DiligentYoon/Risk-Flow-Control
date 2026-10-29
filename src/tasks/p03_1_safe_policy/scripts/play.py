"""Online rollout evaluation of a trained Risk-Flow intervention policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained Risk-Flow intervention policy.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=500, help="Length of evaluation rollout in steps.")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="R1-intervention-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to Risk-Flow checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Path to safety value network checkpoint.")
parser.add_argument("--initial_dataset", type=str, default=None, help="Path to Risk Initial dataset.")
parser.add_argument("--model", type=str, default="MLP", choices=["MLP", "Shared"], help="The NN model used for training the agent.")

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
import torch

import lib
import tasks

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry

from tasks.p03_safe_policy.utils.evaluator import RolloutEvaluator
from tasks.p03_safe_policy.wrappers.intervention_wrapper import InterventionEnvWrapper, InterventionEnvRecordVideo


def main():
    # ============================= Config Parsing =============================
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,)
    try:
        cfg = load_cfg_from_registry(args_cli.task, "rl_single_cfg_entry_point")
        pred_cfg = load_cfg_from_registry(args_cli.task, "predictor_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    env_cfg.seed = args_cli.seed
    cfg["agent"]["seed"] = args_cli.seed
    pred_cfg["agent"]["seed"] = args_cli.seed

    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluation.")
    if args_cli.predictor_checkpoint is None:
        raise ValueError("--predictor_checkpoint is required for evaluation.")

    checkpoint = os.path.abspath(args_cli.checkpoint)
    predictor_checkpoint = os.path.abspath(args_cli.predictor_checkpoint)
    log_dir = os.path.join(os.path.dirname(checkpoint))

    # ============================= Env & Wrapper =============================
    env_cfg.total_timesteps = args_cli.video_length
    env_cfg.events.reset_base.params["dataset_path"] = args_cli.initial_dataset

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during evaluation.")
        env = InterventionEnvRecordVideo(env, **video_kwargs)

    env = InterventionEnvWrapper(env)

    # ============================= Safety Predictor =============================
    from tasks.p02_safety_value.agent.safety import Safety
    from tasks.p02_safety_value.model.safety import SafetyCritic

    num_safety_states = env._unwrapped.num_safety_states
    pred_model = {
        "critic_1": SafetyCritic(num_states=num_safety_states, device=env.device),
        "critic_2": SafetyCritic(num_states=num_safety_states, device=env.device),
    }
    pred_agent = Safety(model=pred_model, device=env.device, cfg=pred_cfg["agent"])

    # ============================= Risk-Flow Agent =============================
    from tasks.p03_safe_policy.agents.risk_flow import RiskFlow
    from tasks.p03_safe_policy.models.risk_flow_models import MultiHorizonCritic, DeterministicActor

    observation_dim = env.observation_space.shape[-1]
    state_dim = env.state_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    model = {
        "critic": MultiHorizonCritic(state_dim, action_dim, cfg["agent"]["horizon"], device=env.device, output_gain=0.01),
        "actor": DeterministicActor(observation_dim, action_dim, device=env.device, output_gain=0.01),
    }

    agent = RiskFlow(model=model, buffer=None, safety_value=pred_agent, device=env.device, cfg=cfg["agent"])

    # ============================= Checkpoints =============================
    agent.load(checkpoint)
    print(f"[INFO] Get checkpoint of intervention policy from {checkpoint}")

    pred_agent.load(predictor_checkpoint)
    print(f"[INFO] Get checkpoint of safety value network from {predictor_checkpoint}")

    agent.set_running_mode("eval")
    pred_agent.set_running_mode("eval")

    # ============================= Evaluator =============================
    evaluator = RolloutEvaluator(
        num_envs=env.num_envs,
        max_episode_length=env._unwrapped.max_episode_length,
        horizon=cfg["agent"]["horizon"],
        device=env.device,
        discount_factor=pred_cfg["agent"]["discount_factor"],
        threshold=pred_cfg["eval"]["threshold"],
    )

    print(f"[INFO] Number of environments: {env.num_envs}")
    print(f"[INFO] Rollout steps: {args_cli.video_length}")
    print(f"[INFO] Safety-state dimension: {num_safety_states}")
    print(f"[INFO] Risk-Flow horizon: {cfg['agent']['horizon']}")
    print(f"[INFO] Risk threshold: {pred_cfg['eval']['threshold']}")
    print(f"[INFO] Evaluation seed: {args_cli.seed}")
    print(f"[INFO] Log directory: {log_dir}")

    # ============================= Rollout =============================
    obs, states, safety_states, safety_values, infos = env.reset()

    timestep = 0
    CLI_interval = 300
    start_time = time.time()

    while simulation_app.is_running():
        with torch.inference_mode():
            # Current transition origin: s_t
            actions = agent.act(obs, deterministic=True)
            pred_values = pred_agent.predict(safety_states).squeeze(-1)
            pred_flows = agent.critic(states, actions)

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

            # V_N(s_{t+1}) at the actual pre-reset state.
            final_pred_values = pred_agent.predict(final_safety_states).squeeze(-1)

        evaluator.append(
            g_values=safety_values,
            pred_values=pred_values,
            pred_flows=pred_flows,
            final_g_values=final_safety_values,
            final_pred_values=final_pred_values,
            terminated=terminated,
            truncated=truncated,
        )

        timestep += 1

        # Update current state for the next transition.
        obs = next_obs
        states = next_states
        safety_states = next_safety_states
        safety_values = next_safety_values
        infos = next_infos

        # ============================= CLI Logging =============================
        if timestep % CLI_interval == 0 or timestep == args_cli.video_length:
            metrics = evaluator.compute()

            elapsed_time = time.time() - start_time
            print(
                f"[INFO] Step {timestep}/{args_cli.video_length} | "
                f"Elapsed {elapsed_time:.1f}s | "
                f"Episodes {metrics['num_episodes']} | "
                f"Terminated {metrics['num_terminated']} | "
                f"Truncated {metrics['num_truncated']}"
            )

            if metrics["num_episodes"] > 0:
                print(
                    f"       Delta Mean: {metrics['delta_mean']:.6f} | "
                    f"Risk Decrease Rate: {metrics['risk_decrease_rate']:.4f} | "
                    f"Flow MAE: {metrics['flow_mae']:.6f} | "
                    f"Flow Direction Acc: {metrics['flow_direction_accuracy']:.4f} | "
                    f"Value MAE: {metrics['value_mae']:.6f} | "
                    f"Risk Miss Rate: {metrics['current_risk_miss_rate']:.4f}"
                )

        # Existing convention: video_length is also the rollout length.
        if timestep == args_cli.video_length:
            break

    # ============================= Final Result =============================
    metrics = evaluator.compute()
    trust_metrics = evaluator.compute_trust_horizon_metrics()

    print("\n================ Evaluation Result ================")
    print(f"Delta Mean              : {metrics['delta_mean']:.6f}")
    print(f"Risk Decrease Rate      : {metrics['risk_decrease_rate']:.6f}")
    print(f"Flow MAE                : {metrics['flow_mae']:.6f}")
    print(f"Flow Direction Accuracy : {metrics['flow_direction_accuracy']:.6f}")
    print(f"Value MAE               : {metrics['value_mae']:.6f}")
    print(f"Current Risk Miss Rate  : {metrics['current_risk_miss_rate']:.6f}")
    print(f"Num Episodes            : {metrics['num_episodes']}")
    print(f"Num Terminated          : {metrics['num_terminated']}")
    print(f"Num Truncated           : {metrics['num_truncated']}")
    for step in [0, 5, 10, 20, 50]:
        if step < len(trust_metrics["risk_miss_rate"]):
            print(
                f"[Trust] Step {step:3d} | "
                f"Miss Rate {trust_metrics['risk_miss_rate'][step]:.4f} | "
                f"Value MAE {trust_metrics['value_mae'][step]:.4f} | "
                f"Unsafe Samples {int(trust_metrics['unsafe_count'][step].item())}"
            )
    print("===================================================")

    if evaluator.num_episodes > 0:
        step_dt = float(env._unwrapped.step_dt)

        plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        evaluator.save_timeseries_plot(file_path=os.path.join(plot_dir, "value_trajectory.png"), step_dt=step_dt)
        evaluator.save_horizon_metrics_plot(file_path=os.path.join(plot_dir, "flow_trajectory.png"), step_dt=step_dt)
        evaluator.save_trust_horizon_plot(file_path=os.path.join(plot_dir, "trust_trajectory.png"), step_dt=step_dt)

        print(f"[INFO] Evaluation plots saved to: {log_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()