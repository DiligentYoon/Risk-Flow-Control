"""
Script to collect risk-classified initial-condition states using a frozen
nominal policy and a trained Reach-Avoid value function.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect risk-classified initial-condition states.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during collection.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments (overrides cfg default if given).")
parser.add_argument("--task", type=str, default="R1-collect", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to nominal policy checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, required=True, help="Path to trained Reach-Avoid value checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm of the nominal policy.")

parser.add_argument("--model",
                    type=str,
                    default="Shared",
                    choices=["MLP", "Shared"],
                    help="The NN model of the nominal policy.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import lib

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.model.model_factory import ModelFactory

from tasks.p02_safety_value.buffer.risk_buffer import RiskBuffer
from tasks.p02_safety_value.wrappers.safety_wrapper import SafetyEnvWrapper

algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None

# ============================ Helpers ============================

def extract_physical_snapshot(info) -> dict[str, torch.Tensor]:
    """Read root + joint state"""
    return {
        "root_pos_offset_w": info["root_pos_offset_w"].clone(),
        "root_quat_w":       info["root_quat_w"].clone(),
        "root_lin_vel_w":    info["root_lin_vel_w"].clone(),
        "root_ang_vel_w":    info["root_ang_vel_w"].clone(),
        "joint_pos":         info["joint_pos"].clone(),
        "joint_vel":         info["joint_vel"].clone(),
        "prev_action":       info["prev_action"].clone(),
    }

def update_risk_streak(risk_streak: torch.Tensor, risk_values: torch.Tensor, threshold: float) -> torch.Tensor:
    risky = risk_values > threshold
    risk_streak[:] = torch.where(risky, risk_streak + 1, torch.zeros_like(risk_streak))
    return risk_streak

def print_progress_box(fill_status, timestep, max_timestep, elapsed_sec, eta_sec):
    content_width = 64
    e_h = int(elapsed_sec // 3600); e_m = int((elapsed_sec % 3600) // 60); e_s = int(elapsed_sec % 60)
    c_h = int(eta_sec // 3600); c_m = int((eta_sec % 3600) // 60); c_s = int(eta_sec % 60)
    line_step = f"Step Progress {timestep} / {max_timestep}"
    line_time = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
    print(" ________________________________________________________________")
    print("|                                                                |")
    print(f"|{line_step.center(content_width)}|")
    print(f"|{line_time.center(content_width)}|")
    print("|________________________________________________________________|")
    print("|                                                                |")
    cur, cap = fill_status
    ratio = 100.0 * cur / max(cap, 1)
    tag = "FULL" if cur >= cap else f"{ratio:5.1f}%"
    print(f"[Collection] {cur}/{cap} ({tag}) | Step {timestep}/{max_timestep} | ETA {eta_sec:.0f}s")
    print("|________________________________________________________________|")


# ============================ Main ============================


def main():
    """Main collection routine."""

    # ============================= Config Parsing ===============================
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric)

    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
        pred_cfg = load_cfg_from_registry(args_cli.task, "predictor_cfg_entry_point")
        collection_cfg = pred_cfg["collection"]
    except ValueError as e:
        print(e)
        return

    # save_dir = next to predictor_checkpoint
    save_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.predictor_checkpoint)),
                            collection_cfg.get("save_subdir", "collected"),)

    # ============================ Env & Wrapper Spawn ================================
    seed = args_cli.seed if args_cli.seed is not None else pred_cfg.get("seed", 42)
    env_cfg.seed = seed
    cfg["agent"]["seed"] = seed
    pred_cfg["agent"]["seed"] = seed

    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

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
    predictor_checkpoint = os.path.abspath(args_cli.predictor_checkpoint)
    agent.load(nominal_checkpoint)
    pred_agent.load(predictor_checkpoint)

    agent.set_running_mode("eval")
    pred_agent.set_running_mode("eval")

    print(f"[INFO] Nominal checkpoint: {nominal_checkpoint}")
    print(f"[INFO] Predictor checkpoint: {predictor_checkpoint}")

    # ============= Risk-classified buffer ===============
    joint_dim = env._unwrapped._robot.num_joints
    risk_buffer = RiskBuffer(
        capacity=int(collection_cfg["capacity"]),
        joint_dim=joint_dim,
        device=env.device,
    )

    # ============= Collection loop ===============
    max_timestep = int(collection_cfg["max_timestep"])
    log_interval = 500
    persistent_steps = int(collection_cfg.get("persistent_risk_steps", 10))
    risk_threshold = float(collection_cfg.get("risk_threshold", 0.0))
    
    risk_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    collected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obs, states, safety_states, infos = env.reset()

    timestep = 0
    t_start = time.time()
    while simulation_app.is_running() and timestep < max_timestep:
        with torch.no_grad():
            pred_values = pred_agent.predict(safety_states).squeeze(-1)
            snapshot = extract_physical_snapshot(infos["collection"])

            risk_streak = update_risk_streak(risk_streak, pred_values, risk_threshold)
            persistent_mask = (risk_streak == persistent_steps) & ~collected # 최초감지 state에 대해 저장 & 에피소드당 최대 1개
            risk_buffer.add(snapshot=snapshot, risk_scores=pred_values, mask=persistent_mask)
            collected[persistent_mask] = True

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

            done = (terminated | truncated).flatten().bool()
            risk_streak[done] = 0
            collected[done] = False

        timestep += 1
        obs = next_obs
        states = next_states
        safety_states = next_safety_states
        safety_values = next_safety_values
        infos = next_infos

        if timestep % log_interval == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(timestep, 1) * max(max_timestep - timestep, 0)
            print_progress_box(risk_buffer.fill_status(), timestep, max_timestep, elapsed, eta)

        if risk_buffer.is_full():
            print("[INFO] All buckets reached capacity. Stopping.")
            break

    # ============= Save & summary ===============
    risk_buffer.save(save_dir)
    print(f"[INFO] Saved risk-classified buckets to {save_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
