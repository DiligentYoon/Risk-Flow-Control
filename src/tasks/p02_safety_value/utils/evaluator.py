import torch
import numpy as np
import os

class RolloutEvaluator:
    def __init__(
        self,
        num_envs: int,
        max_segment_length: int,
        device: torch.device,
        discount_factor: float = 0.99,
        threshold: float = 0.0,
        file_path: str = None,
    ) -> None:
        self.num_envs = num_envs
        self.max_segment_length = max_segment_length
        self.device = device
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.file_path = file_path

        self.env_ids = torch.arange(num_envs, device=device)
        self.g_buffer = torch.zeros((num_envs, max_segment_length), dtype=torch.float32, device=device)
        self.pred_buffer = torch.zeros((num_envs, max_segment_length), dtype=torch.float32, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_pred_risk = torch.zeros(num_envs, dtype=torch.bool, device=device) 

        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

        self.segment_tp = 0
        self.segment_fn = 0
        self.segment_fp = 0
        self.segment_tn = 0
        self.risk_persistence = 10
        self.safe_persistence = self.risk_persistence

        self.episode_alarm_count = 0
        self.episode_alarm_terminated = 0
        self.episode_alarm_false = 0

        self.num_segments = 0
        self.num_samples = 0
        self.num_episodes = 0
        self.num_terminated = 0
        self.num_truncated = 0
        self.short_segments_skipped = 0

        self.proactive_recall_sum = 0.0
        self.proactive_recall_count = 0

        # Time-series history (single env : env_ids = 0)
        self.step_count = 0
        self.env_history = {
            "step": [],
            "g_value": [],
            "pred_value": [],
            "empirical_value": [],
            "discounted_empirical_value": [],
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
        self.env_history["discounted_empirical_value"].append(float("nan"))
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

        done_pred_risk = self.episode_pred_risk & done
        self.episode_alarm_count += int(done_pred_risk.sum().item())
        self.episode_alarm_terminated += int((done_pred_risk & terminated).sum().item())
        self.episode_alarm_false += int((done_pred_risk & (~terminated)).sum().item())
        self.episode_pred_risk[done] = False

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

            # empirical value
            future_max_g = torch.flip(torch.cummax(torch.flip(g_values, dims=[0]), dim=0).values, dims=[0])

            # time-discounted empirical value
            discounted_empirical_value = torch.empty_like(g_values)
            discounted_empirical_value[-1] = g_values[-1]
            for t in range(g_values.numel()-2, -1, -1):
                discounted_empirical_value[t] = (1.0 - self.discount_factor) * g_values[t] + self.discount_factor * torch.maximum(g_values[t], discounted_empirical_value[t+1])

            # Fill empirical value for timeseries analysis
            if env_id == 0:
                future_values = future_max_g.detach().cpu().tolist()
                discounted_values = discounted_empirical_value.detach().cpu().tolist()
                if len(self.env_segment_indices) != length:
                    raise RuntimeError(f"Env 0 history mismatch: indices={len(self.env_segment_indices)}, segment={length}")
                
                for history_index, value, discounted_value in zip(self.env_segment_indices, future_values, discounted_values):
                    self.env_history["empirical_value"][history_index] = value
                    self.env_history["discounted_empirical_value"][history_index] = discounted_value

                self.env_segment_indices.clear()

            # Risk coverage rate metric
            empirical_values = future_max_g[:-1]
            pred_values = pred_values[:-1]

            real_risk = empirical_values > self.threshold
            pred_risk = pred_values > self.threshold

            self.tp += int((pred_risk & real_risk).sum().item())
            self.fn += int(((~pred_risk) & real_risk).sum().item())
            self.fp += int((pred_risk & (~real_risk)).sum().item())
            self.tn += int(((~pred_risk) & (~real_risk)).sum().item())

            self.num_samples += length - 1
            self.num_segments += 1
            self.lengths[env_id] = 0

            # Safe decision metric
            real_segment_risk = bool(torch.any(g_values > self.threshold).item())
            step_risk = pred_values > self.threshold
            pred_segment_risk = self._has_consecutive_true(step_risk, self.risk_persistence)

            if pred_segment_risk:
                self.episode_pred_risk[env_id] = True
            if real_segment_risk and pred_segment_risk:
                self.segment_tp += 1
            elif real_segment_risk and not pred_segment_risk:
                self.segment_fn += 1
            elif not real_segment_risk and pred_segment_risk:
                self.segment_fp += 1
            else:
                self.segment_tn += 1

            # Proactive recall metric
            unsafe_indices = torch.nonzero(g_values > self.threshold, as_tuple=False).flatten()
            if unsafe_indices.numel() > 0:
                unsafe_idx = int(unsafe_indices[0].item())

                if unsafe_idx > 0:
                    pred_idx = self._first_consecutive_true(step_risk, self.risk_persistence)

                    if pred_idx is None:
                        proactive_recall = 0.0
                    else:
                        proactive_recall = max(0.0, (unsafe_idx - pred_idx) / unsafe_idx)

                    self.proactive_recall_sum += proactive_recall
                    self.proactive_recall_count += 1

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    @staticmethod
    def _has_consecutive_true(values: torch.Tensor, count: int) -> bool:
        if values.numel() < count:
            return False
        run = 0
        for value in values.tolist():
            if value:
                run += 1
                if run >= count:
                    return True
            else:
                run = 0
        return False

    @staticmethod
    def _first_consecutive_true(values: torch.Tensor, count: int) -> int | None:
        run = 0
        for i, value in enumerate(values.tolist()):
            if value:
                run += 1
                if run >= count:
                    return i
            else:
                run = 0
        return None

    def compute(self) -> dict[str, float]:
        risk_coverage_rate = self._safe_div(self.tp, self.tp + self.fn)
        detection_rate = self._safe_div(self.segment_tp, self.segment_tp + self.segment_fn)
        false_alarm_rate = self._safe_div(self.segment_fp, self.segment_fp + self.segment_tn)
        proactive_recall = self._safe_div(self.proactive_recall_sum, self.proactive_recall_count)
        accuracy = self._safe_div(self.segment_tp + self.segment_tn, self.num_segments)
        episode_false_alarm_rate = self._safe_div(self.episode_alarm_false, self.num_truncated)
        episode_detection_rate = self._safe_div(self.episode_alarm_terminated, self.num_terminated)

        return {
            "risk_coverage_rate": risk_coverage_rate,
            "risk_detection_rate": detection_rate,
            "risk_false_alarm_rate": false_alarm_rate,            
            "termination_detection_rate": episode_detection_rate,
            "termination_false_alarm_rate": episode_false_alarm_rate,
            "proactive_recall": proactive_recall,
            "accuracy": accuracy,
            "real_risk_rate": self._safe_div(self.tp + self.fn, self.num_samples),
            "pred_risk_rate": self._safe_div(self.tp + self.fp, self.num_samples),
            "num_segments": self.num_segments,
            "num_terminated": self.num_terminated,
            "num_truncated": self.num_truncated,
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
        discounted_empirical_values = np.asarray(self.env_history["discounted_empirical_value"], dtype=np.float32)

        push_events = np.asarray(self.env_history["push_event"], dtype=bool)
        terminated = np.asarray(self.env_history["terminated"], dtype=bool)
        truncated = np.asarray(self.env_history["truncated"], dtype=bool)

        pred_risk = pred_values > self.threshold
        empirical_risk = empirical_values > self.threshold

        risk_step = 0
        safe_step = 0
        decision = np.zeros_like(pred_risk, dtype=np.bool_)
        for i in range(len(pred_risk)):
            if push_events[i]:
                risk_step = 0
            if pred_risk[i]:
                risk_step += 1
                safe_step = 0
                decision[i] = risk_step >= self.risk_persistence
            else:
                risk_step = 0
                safe_step += 1
                if i > 0:
                    if terminated[i-1] or truncated[i-1]:
                        continue
                    else:
                        if decision[i-1]:
                            decision[i] = safe_step < self.safe_persistence

            if terminated[i] or truncated[i]:
                risk_step = 0

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # ====================== Value ====================== #
        axes[0].plot(time_axis, g_values, label=r"$g(s_t)$")
        axes[0].plot(time_axis, pred_values, label=r"$V_\theta(s_t)$")
        axes[0].plot(time_axis, discounted_empirical_values, linestyle="--", label=r"$\hat{V}_{\mathrm{emp}}^\gamma(s_t)$")

        axes[0].axhline(self.threshold, linestyle=":", label="Risk boundary")
        axes[0].set_ylabel("Safety value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ====================== Risk classification ====================== #
        axes[1].step(time_axis, empirical_risk.astype(float), where="post", linestyle="--", label="Empirical future risk")
        axes[1].step(time_axis, decision.astype(float), where="post", label="Predicted risk")

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