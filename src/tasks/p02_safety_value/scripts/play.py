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

from tasks.p02_safety_value.wrappers.safety_wrapper import SafetyEnvWrapper, SafetyEnvRecordVideo


algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None


class RolloutEvaluator:
    def __init__(
        self,
        num_envs: int,
        max_segment_length: int,
        device: torch.device,
        threshold: float = 0.0,
        file_path: str = None,
    ) -> None:
        self.num_envs = num_envs
        self.max_segment_length = max_segment_length
        self.device = device
        self.threshold = threshold
        self.file_path = file_path

        self.env_ids = torch.arange(num_envs, device=device)
        self.g_buffer = torch.zeros((num_envs, max_segment_length), dtype=torch.float32, device=device)
        self.pred_buffer = torch.zeros((num_envs, max_segment_length), dtype=torch.float32, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

        self.abs_error_sum = 0.0
        self.squared_error_sum = 0.0

        self.num_segments = 0
        self.num_samples = 0
        self.num_episodes = 0
        self.num_terminated = 0
        self.num_truncated = 0
        self.short_segments_skipped = 0

        # Time-series history (single env : env_ids = 0)
        self.step_count = 0
        self.env_history = {
            "step": [],
            "g_value": [],
            "pred_value": [],
            "empirical_value": [],
            "push_event": [],
            "terminated": [],
            "truncated": [],
        }
        self.env_segment_indices = []

    @torch.no_grad()
    def append(
        self,
        g_values: torch.Tensor,
        pred_values: torch.Tensor,
        push_events: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        g_values = g_values.reshape(-1).to(self.device)
        pred_values = pred_values.reshape(-1).to(self.device)
        push_events = push_events.reshape(-1).bool().to(self.device)
        terminated = terminated.reshape(-1).bool().to(self.device)
        truncated = truncated.reshape(-1).bool().to(self.device)

        push_boundaries = push_events & (self.lengths > 0)
        self._finalize(push_boundaries)

        if torch.any(self.lengths >= self.max_segment_length):
            raise RuntimeError("Rollout evaluator segment buffer overflow.")

        self.g_buffer[self.env_ids, self.lengths] = g_values
        self.pred_buffer[self.env_ids, self.lengths] = pred_values
        self.lengths += 1

        # History
        history_index = len(self.env_history["step"])
        self.env_history["step"].append(self.step_count)
        self.env_history["g_value"].append(float(g_values[0].item()))
        self.env_history["pred_value"].append(float(pred_values[0].item()))
        self.env_history["empirical_value"].append(float("nan"))
        self.env_history["push_event"].append(bool(push_events[0].item()))
        self.env_history["terminated"].append(bool(terminated[0].item()))
        self.env_history["truncated"].append(bool(truncated[0].item()))
        self.env_segment_indices.append(history_index)
        self.step_count += 1

        done = terminated | truncated
        self.num_episodes += int(done.sum().item())
        self.num_terminated += int(terminated.sum().item())
        self.num_truncated += int(truncated.sum().item())

        self._finalize(done)

    @torch.no_grad()
    def _finalize(self, mask: torch.Tensor) -> None:
        env_ids = torch.nonzero(mask, as_tuple=False).flatten()

        for env_id in env_ids.tolist():
            length = int(self.lengths[env_id].item())

            if length < 2:
                if length > 0:
                    self.short_segments_skipped += 1
                self.lengths[env_id] = 0
                continue

            g_values = self.g_buffer[env_id, :length]
            pred_values = self.pred_buffer[env_id, :length]

            future_max_g = torch.flip(torch.cummax(torch.flip(g_values, dims=[0]), dim=0).values, dims=[0])

            # Fill empirical value for timeseries analysis
            if env_id == 0:
                future_values = future_max_g.detach().cpu().tolist()

                if len(self.env_segment_indices) != length:
                    raise RuntimeError(f"Env 0 history mismatch: indices={len(self.env_segment_indices)}, segment={length}")

                for history_index, value in zip(self.env_segment_indices, future_values):
                    self.env_history["empirical_value"][history_index] = value

                self.env_segment_indices.clear()

            empirical_values = future_max_g[:-1]
            pred_values = pred_values[:-1]

            real_risk = empirical_values > self.threshold
            pred_risk = pred_values > self.threshold

            self.tp += int((pred_risk & real_risk).sum().item())
            self.fn += int(((~pred_risk) & real_risk).sum().item())
            self.fp += int((pred_risk & (~real_risk)).sum().item())
            self.tn += int(((~pred_risk) & (~real_risk)).sum().item())

            errors = pred_values - empirical_values
            self.abs_error_sum += float(errors.abs().sum().item())
            self.squared_error_sum += float(errors.square().sum().item())

            self.num_samples += length - 1
            self.num_segments += 1
            self.lengths[env_id] = 0

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    def compute(self) -> dict[str, float]:
        detection_rate = self._safe_div(self.tp, self.tp + self.fn)
        false_alarm_rate = self._safe_div(self.fp, self.fp + self.tn)
        specificity = self._safe_div(self.tn, self.tn + self.fp)
        precision = self._safe_div(self.tp, self.tp + self.fp)
        accuracy = self._safe_div(self.tp + self.tn, self.num_samples)
        balanced_accuracy = 0.5 * (detection_rate + specificity) if not np.isnan(detection_rate) and not np.isnan(specificity) else float("nan")

        return {
            "detection_rate": detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "real_risk_rate": self._safe_div(self.tp + self.fn, self.num_samples),
            "pred_risk_rate": self._safe_div(self.tp + self.fp, self.num_samples),
            "empirical_mae": self._safe_div(self.abs_error_sum, self.num_samples),
            "empirical_mse": self._safe_div(self.squared_error_sum, self.num_samples),
            "num_segments": self.num_segments,
            "num_samples": self.num_samples,
            "num_episodes": self.num_episodes,
            "num_terminated": self.num_terminated,
            "num_truncated": self.num_truncated,
            "partial_segments": int((self.lengths > 0).sum().item()),
            "partial_samples": int(self.lengths.sum().item()),
            "short_segments_skipped": self.short_segments_skipped,
            "true_positive": self.tp,
            "false_negative": self.fn,
            "false_positive": self.fp,
            "true_negative": self.tn,
        }

    def save_timeseries_plot(self, step_dt: float) -> None:
        import matplotlib.pyplot as plt

        if len(self.env_history["step"]) == 0:
            return

        steps = np.asarray(self.env_history["step"])
        time_axis = steps * step_dt

        g_values = np.asarray(self.env_history["g_value"], dtype=np.float32)
        pred_values = np.asarray(self.env_history["pred_value"], dtype=np.float32)
        empirical_values = np.asarray(self.env_history["empirical_value"], dtype=np.float32)

        push_events = np.asarray(self.env_history["push_event"], dtype=bool)
        terminated = np.asarray(self.env_history["terminated"], dtype=bool)
        truncated = np.asarray(self.env_history["truncated"], dtype=bool)

        pred_risk = pred_values > self.threshold
        empirical_risk = empirical_values > self.threshold

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # ====================== Value ====================== #
        axes[0].plot(time_axis, g_values, label=r"$g(s_t)$")
        axes[0].plot(time_axis, pred_values, label=r"$V_\theta(s_t)$")
        axes[0].plot(time_axis, empirical_values, linestyle="--", label=r"$\max_{k \geq t} g(s_k)$")

        axes[0].axhline(self.threshold, linestyle=":", label="Risk boundary")
        axes[0].set_ylabel("Safety value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ====================== Risk classification ====================== #
        axes[1].step(time_axis, pred_risk.astype(float), where="post", label="Predicted risk")
        axes[1].step(time_axis, empirical_risk.astype(float), where="post", linestyle="--", label="Empirical future risk")

        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Safe", "Risk"])
        axes[1].legend()
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Risk class")
        axes[1].grid(True, alpha=0.3)

        # ====================== Events ====================== #
        for index in np.where(push_events)[0]:
            for ax in axes:
                ax.axvline(time_axis[index], linestyle=":", alpha=0.5)

        for index in np.where(terminated)[0]:
            for ax in axes:
                ax.axvline(time_axis[index], linestyle="--", linewidth=1.5)

        for index in np.where(truncated)[0]:
            for ax in axes:
                ax.axvline(time_axis[index], linestyle="-.", linewidth=1.5)

        fig.tight_layout()

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        fig.savefig(self.file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


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
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = SafetyEnvRecordVideo(env, **video_kwargs)
        print("[INFO] Recording evaluation video.")

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
                f"episodes={metrics['num_episodes']} | "
                f"segments={metrics['num_segments']} | "
                f"samples={metrics['num_samples']} | "
                f"DR={metrics['detection_rate']:.4f} | "
                f"FAR={metrics['false_alarm_rate']:.4f} | "
                f"BA={metrics['balanced_accuracy']:.4f} | "
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
    print(f"Completed episodes     : {metrics['num_episodes']}")
    print(f"Completed segments     : {metrics['num_segments']}")
    print(f"Evaluation samples     : {metrics['num_samples']}")
    print(f"Partial segments       : {metrics['partial_segments']}")
    print(f"Partial samples        : {metrics['partial_samples']}")
    print(f"Detection rate         : {metrics['detection_rate'] * 100:.2f}%")
    print(f"False alarm rate       : {metrics['false_alarm_rate'] * 100:.2f}%")
    print(f"Balanced accuracy      : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"Accuracy               : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision              : {metrics['precision'] * 100:.2f}%")
    print(f"Empirical MAE          : {metrics['empirical_mae']:.6f}")
    print(f"Empirical MSE          : {metrics['empirical_mse']:.6f}")
    print(f"Real future-risk ratio : {metrics['real_risk_rate'] * 100:.2f}%")
    print(f"Predicted-risk ratio   : {metrics['pred_risk_rate'] * 100:.2f}%")
    print(f"TP / FN / FP / TN      : {metrics['true_positive']} / {metrics['false_negative']} / {metrics['false_positive']} / {metrics['true_negative']}")
    print(f"[INFO] Results saved to: {result_path}")

    evaluator.save_timeseries_plot(step_dt=float(env._unwrapped.step_dt))
    print(f"[INFO] Rollout plot saved to: {evaluator.file_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()