import os

import numpy as np
import torch


class RolloutEvaluator:
    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        horizon: int,
        device: torch.device,
        discount_factor: float = 0.99,
        threshold: float = 0.0,
    ) -> None:
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.horizon = horizon
        self.device = device
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.direction_epsilon = 1e-6

        self.env_ids = torch.arange(num_envs, device=device)
        self.g_buffer = torch.zeros((num_envs, max_episode_length + 1), dtype=torch.float32, device=device)
        self.value_buffer = torch.zeros((num_envs, max_episode_length + 1), dtype=torch.float32, device=device)
        self.flow_buffer = torch.zeros((num_envs, max_episode_length, horizon), dtype=torch.float32, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.num_samples = 0
        self.num_episodes = 0
        self.num_terminated = 0
        self.num_truncated = 0
        self.episodes = []

        self.step_count = 0
        self.history = {
            "step": [],
            "g_value": [],
            "pred_value": [],
            "delta_value": [],
            "discounted_empirical_value": [],
            "terminated": [],
            "truncated": [],
        }
        self.history_indices = []
        self.history_episodes = []

    @torch.no_grad()
    def append(
        self,
        g_values: torch.Tensor,
        pred_values: torch.Tensor,
        pred_flows: torch.Tensor,
        final_g_values: torch.Tensor,
        final_pred_values: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        g_values = g_values.reshape(-1).to(self.device)
        pred_values = pred_values.reshape(-1).to(self.device)
        pred_flows = pred_flows.to(self.device)
        final_g_values = final_g_values.reshape(-1).to(self.device)
        final_pred_values = final_pred_values.reshape(-1).to(self.device)
        terminated = terminated.reshape(-1).bool().to(self.device)
        truncated = truncated.reshape(-1).bool().to(self.device)

        if torch.any(self.lengths >= self.max_episode_length):
            raise RuntimeError("Rollout evaluator episode buffer overflow.")
        if pred_flows.shape != (self.num_envs, self.horizon):
            raise RuntimeError(f"Invalid flow shape: {pred_flows.shape}, expected {(self.num_envs, self.horizon)}")

        indices = self.lengths
        self.g_buffer[self.env_ids, indices] = g_values
        self.value_buffer[self.env_ids, indices] = pred_values
        self.flow_buffer[self.env_ids, indices] = pred_flows
        self.g_buffer[self.env_ids, indices + 1] = final_g_values
        self.value_buffer[self.env_ids, indices + 1] = final_pred_values
        self.lengths += 1

        history_index = len(self.history["step"])
        self.history["step"].append(self.step_count)
        self.history["g_value"].append(float(g_values[0].item()))
        self.history["pred_value"].append(float(pred_values[0].item()))
        self.history["delta_value"].append(float("nan"))
        self.history["discounted_empirical_value"].append(float("nan"))
        self.history["terminated"].append(bool(terminated[0].item()))
        self.history["truncated"].append(bool(truncated[0].item()))
        self.history_indices.append(history_index)
        self.step_count += 1

        done = terminated | truncated
        self.num_episodes += int(done.sum().item())
        self.num_terminated += int(terminated.sum().item())
        self.num_truncated += int(truncated.sum().item())
        self._finalize(done, terminated, truncated)

    @torch.no_grad()
    def _finalize(
        self,
        mask: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        env_ids = torch.nonzero(mask, as_tuple=False).flatten()

        for env_id in env_ids.tolist():
            length = int(self.lengths[env_id].item())
            if length == 0:
                continue

            g_values = self.g_buffer[env_id, :length + 1]
            values = self.value_buffer[env_id, :length + 1]
            pred_flows = self.flow_buffer[env_id, :length]
            delta_values = values[1:] - values[:-1]

            discounted_empirical_values = torch.empty_like(g_values)
            discounted_empirical_values[-1] = g_values[-1]
            for t in range(length - 1, -1, -1):
                discounted_empirical_values[t] = (
                    (1.0 - self.discount_factor) * g_values[t]
                    + self.discount_factor * torch.maximum(g_values[t], discounted_empirical_values[t + 1])
                )

            gt_flows = torch.zeros((length, self.horizon), dtype=torch.float32, device=self.device)
            valid_flow_mask = torch.zeros((length, self.horizon), dtype=torch.bool, device=self.device)
            is_terminated = bool(terminated[env_id].item())

            for t in range(length):
                valid_horizon = min(self.horizon, length - t)
                gt_flows[t, :valid_horizon] = values[t + 1:t + 1 + valid_horizon] - values[t]
                valid_flow_mask[t, :valid_horizon] = True

                if is_terminated and valid_horizon < self.horizon:
                    gt_flows[t, valid_horizon:] = values[-1] - values[t]
                    valid_flow_mask[t, valid_horizon:] = True

            episode = {
                "env_id": env_id,
                "g_values": g_values.detach().cpu().clone(),
                "values": values.detach().cpu().clone(),
                "delta_values": delta_values.detach().cpu().clone(),
                "discounted_empirical_values": discounted_empirical_values.detach().cpu().clone(),
                "pred_flows": pred_flows.detach().cpu().clone(),
                "gt_flows": gt_flows.detach().cpu().clone(),
                "valid_flow_mask": valid_flow_mask.detach().cpu().clone(),
                "terminated": is_terminated,
                "truncated": bool(truncated[env_id].item()),
            }
            self.episodes.append(episode)

            if env_id == 0:
                self.history_episodes.append(episode)

                if len(self.history_indices) != length:
                    raise RuntimeError(
                        f"History mismatch: indices={len(self.history_indices)}, episode={length}"
                    )

                delta_cpu = delta_values.detach().cpu().tolist()
                empirical_cpu = discounted_empirical_values[:-1].detach().cpu().tolist()

                for history_index, delta_value, empirical_value in zip(
                    self.history_indices,
                    delta_cpu,
                    empirical_cpu,
                ):
                    self.history["delta_value"][history_index] = delta_value
                    self.history["discounted_empirical_value"][history_index] = empirical_value

                self.history_indices.clear()

            self.num_samples += length
            self.lengths[env_id] = 0

    def compute(self) -> dict[str, float]:
        if len(self.episodes) == 0:
            return {}

        delta_values = torch.cat([episode["delta_values"] for episode in self.episodes])
        values = torch.cat([episode["values"][:-1] for episode in self.episodes])
        empirical_values = torch.cat([episode["discounted_empirical_values"][:-1] for episode in self.episodes])
        g_values = torch.cat([episode["g_values"][:-1] for episode in self.episodes])

        pred_flows = torch.cat([episode["pred_flows"].reshape(-1) for episode in self.episodes])
        gt_flows = torch.cat([episode["gt_flows"].reshape(-1) for episode in self.episodes])
        valid_flow_mask = torch.cat([episode["valid_flow_mask"].reshape(-1) for episode in self.episodes])

        pred_flows = pred_flows[valid_flow_mask]
        gt_flows = gt_flows[valid_flow_mask]

        delta_mean = delta_values.mean().item()
        risk_decrease_rate = (delta_values < 0.0).float().mean().item()
        flow_mae = torch.abs(pred_flows - gt_flows).mean().item()

        direction_mask = torch.abs(gt_flows) > self.direction_epsilon
        if torch.any(direction_mask):
            flow_direction_accuracy = (torch.sign(pred_flows[direction_mask]) == torch.sign(gt_flows[direction_mask])).float().mean().item()
        else:
            flow_direction_accuracy = float("nan")

        value_mae = torch.abs(values - empirical_values).mean().item()

        unsafe_mask = g_values > self.threshold
        if torch.any(unsafe_mask):
            current_risk_miss_rate = (values[unsafe_mask] <= self.threshold).float().mean().item()
        else:
            current_risk_miss_rate = float("nan")

        return {
            "delta_mean": delta_mean,
            "risk_decrease_rate": risk_decrease_rate,
            "flow_mae": flow_mae,
            "flow_direction_accuracy": flow_direction_accuracy,
            "value_mae": value_mae,
            "current_risk_miss_rate": current_risk_miss_rate,
            "num_episodes": self.num_episodes,
            "num_terminated": self.num_terminated,
            "num_truncated": self.num_truncated,
        }

    def compute_horizon_metrics(self) -> dict[str, torch.Tensor]:
        mae_sum = torch.zeros(self.horizon, dtype=torch.float64)
        mae_count = torch.zeros(self.horizon, dtype=torch.float64)
        direction_correct = torch.zeros(self.horizon, dtype=torch.float64)
        direction_count = torch.zeros(self.horizon, dtype=torch.float64)

        for episode in self.episodes:
            pred = episode["pred_flows"]
            gt = episode["gt_flows"]
            valid = episode["valid_flow_mask"]

            error = torch.abs(pred - gt)
            mae_sum += (error * valid).sum(dim=0).double()
            mae_count += valid.sum(dim=0).double()

            direction_mask = valid & (torch.abs(gt) > self.direction_epsilon)
            direction_correct += (
                (torch.sign(pred) == torch.sign(gt)) & direction_mask
            ).sum(dim=0).double()
            direction_count += direction_mask.sum(dim=0).double()

        flow_mae = torch.full((self.horizon,), float("nan"), dtype=torch.float64)
        flow_direction_accuracy = torch.full((self.horizon,), float("nan"), dtype=torch.float64)

        valid_mae = mae_count > 0
        valid_direction = direction_count > 0
        flow_mae[valid_mae] = mae_sum[valid_mae] / mae_count[valid_mae]
        flow_direction_accuracy[valid_direction] = (
            direction_correct[valid_direction] / direction_count[valid_direction]
        )

        return {
            "flow_mae": flow_mae,
            "flow_direction_accuracy": flow_direction_accuracy,
            "valid_count": mae_count,
            "direction_count": direction_count,
        }

    def save_timeseries_plot(self, file_path: str, step_dt: float) -> None:
        import matplotlib.pyplot as plt

        if len(self.history["step"]) == 0:
            return

        steps = np.asarray(self.history["step"])
        time_axis = steps * step_dt

        g_values = np.asarray(self.history["g_value"], dtype=np.float32)
        values = np.asarray(self.history["pred_value"], dtype=np.float32)
        delta_values = np.asarray(self.history["delta_value"], dtype=np.float32)
        empirical_values = np.asarray(self.history["discounted_empirical_value"], dtype=np.float32)
        terminated = np.asarray(self.history["terminated"], dtype=bool)
        truncated = np.asarray(self.history["truncated"], dtype=bool)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        axes[0].plot(time_axis, g_values, label=r"$g(s_t)$")
        axes[0].plot(time_axis, values, label=r"$V_N(s_t)$")
        axes[0].plot(
            time_axis,
            empirical_values,
            linestyle="--",
            label=r"$V_{\mathrm{emp}}^\gamma(s_t)$",
        )
        axes[0].axhline(self.threshold, linestyle=":", label="Risk boundary")
        axes[0].set_ylabel("Risk value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_axis, delta_values, label=r"$\Delta_N(t)$")
        axes[1].axhline(0.0, linestyle=":")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel(r"$\Delta_N$")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        for index in np.where(terminated)[0]:
            for ax in axes:
                ax.axvline(time_axis[index], linestyle="--", linewidth=1.5)

        for index in np.where(truncated)[0]:
            for ax in axes:
                ax.axvline(time_axis[index], linestyle="-.", linewidth=1.5)

        fig.tight_layout()
        fig.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_horizon_metrics_plot(self, file_path: str, step_dt: float) -> None:
        import matplotlib.pyplot as plt

        metrics = self.compute_horizon_metrics()
        flow_mae = metrics["flow_mae"].numpy()
        direction_accuracy = metrics["flow_direction_accuracy"].numpy()
        horizon_time = np.arange(1, self.horizon + 1) * step_dt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(horizon_time, flow_mae)
        axes[0].set_ylabel("Flow MAE")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(horizon_time, direction_accuracy)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlabel("Prediction horizon [s]")
        axes[1].set_ylabel("Direction accuracy")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)