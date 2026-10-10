import os

import numpy as np
import torch


class RolloutEvaluator:
    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        device: torch.device,
        discount_factor: float = 0.99,
        threshold: float = 0.0,
    ) -> None:
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.direction_epsilon = 1e-6

        self.env_ids = torch.arange(num_envs, device=device)
        self.l_buffer = torch.zeros((num_envs, max_episode_length + 1), dtype=torch.float32, device=device)
        self.g_buffer = torch.zeros((num_envs, max_episode_length + 1), dtype=torch.float32, device=device)
        self.value_buffer = torch.zeros((num_envs, max_episode_length + 1), dtype=torch.float32, device=device)
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
            "l_value": [],
            "pred_value": [],
            "terminated": [],
            "truncated": [],
        }
        self.history_indices = []
        self.history_episodes = []

    @torch.no_grad()
    def append(
        self,
        g_values: torch.Tensor,
        l_values: torch.Tensor,
        pred_values: torch.Tensor,
        final_g_values: torch.Tensor,
        final_l_values: torch.Tensor,
        final_pred_values: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        g_values = g_values.reshape(-1).to(self.device)
        l_values = l_values.reshape(-1).to(self.device)
        pred_values = pred_values.reshape(-1).to(self.device)
        final_g_values = final_g_values.reshape(-1).to(self.device)
        final_l_values = final_l_values.reshape(-1).to(self.device)
        final_pred_values = final_pred_values.reshape(-1).to(self.device)
        terminated = terminated.reshape(-1).bool().to(self.device)
        truncated = truncated.reshape(-1).bool().to(self.device)

        if torch.any(self.lengths >= self.max_episode_length):
            raise RuntimeError("Rollout evaluator episode buffer overflow.")

        indices = self.lengths
        self.g_buffer[self.env_ids, indices] = g_values
        self.l_buffer[self.env_ids, indices] = l_values
        self.value_buffer[self.env_ids, indices] = pred_values
        self.g_buffer[self.env_ids, indices + 1] = final_g_values
        self.l_buffer[self.env_ids, indices + 1] = final_l_values
        self.value_buffer[self.env_ids, indices + 1] = final_pred_values
        self.lengths += 1

        history_index = len(self.history["step"])
        self.history["step"].append(self.step_count)
        self.history["g_value"].append(float(g_values[0].item()))
        self.history["l_value"].append(float(l_values[0].item()))
        self.history["pred_value"].append(float(pred_values[0].item()))
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
    def _finalize(self, mask: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        env_ids = torch.nonzero(mask, as_tuple=False).flatten()

        for env_id in env_ids.tolist():
            length = int(self.lengths[env_id].item())
            if length == 0:
                continue

            g_values = self.g_buffer[env_id, :length + 1]
            l_values = self.l_buffer[env_id, :length + 1]
            values = self.value_buffer[env_id, :length + 1]

            episode = {
                "env_id": env_id,
                "g_values": g_values.detach().cpu().clone(),
                "l_values": l_values.detach().cpu().clone(),
                "values": values.detach().cpu().clone(),
                "truncated": bool(truncated[env_id].item()),
            }
            self.episodes.append(episode)

            if env_id == 0:
                self.history_episodes.append(episode)
                if len(self.history_indices) != length:
                    raise RuntimeError(f"History mismatch: indices={len(self.history_indices)}, episode={length}")
                self.history_indices.clear()

            self.num_samples += length
            self.lengths[env_id] = 0

    def compute(self) -> dict[str, float]:
        if len(self.episodes) == 0:
            return {}

        values = torch.cat([episode["values"][:-1] for episode in self.episodes])
        g_values = torch.cat([episode["g_values"][:-1] for episode in self.episodes])
        l_values = torch.cat([episode["l_values"][:-1] for episode in self.episodes])

        return {
            "num_episodes": self.num_episodes,
            "num_terminated": self.num_terminated,
            "num_truncated": self.num_truncated,
        }

    def save_timeseries_plot(self, file_path: str, step_dt: float) -> None:
        import matplotlib.pyplot as plt

        if len(self.history["step"]) == 0:
            return

        steps = np.asarray(self.history["step"])
        time_axis = steps * step_dt

        g_values = np.asarray(self.history["g_value"], dtype=np.float32)
        l_values = np.asarray(self.history["l_value"], dtype=np.float32)
        values = np.asarray(self.history["pred_value"], dtype=np.float32)
        terminated = np.asarray(self.history["terminated"], dtype=bool)
        truncated = np.asarray(self.history["truncated"], dtype=bool)

        fig, axes = plt.subplots(1, 1, figsize=(16, 8), sharex=True)

        axes.plot(time_axis, g_values, label=r"$g(s_t)$")
        axes.plot(time_axis, l_values, label=r"$l(s_t)$")
        axes.plot(time_axis, values, label=r"$Q_N(s_t,a_t)$")
        axes.axhline(self.threshold, linestyle=":", label="Risk boundary")
        axes.set_ylabel("Risk value")
        axes.legend()
        axes.grid(True, alpha=0.3)

        for index in np.where(terminated)[0]:
            axes.axvline(time_axis[index], linestyle="--", linewidth=1.5)

        for index in np.where(truncated)[0]:
            axes.axvline(time_axis[index], linestyle="-.", linewidth=1.5)

        fig.tight_layout()
        fig.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)