from __future__ import annotations

from typing import Dict, Union

import itertools
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent


class Safety(Agent):
    def __init__(
        self,
        model: Dict[str, nn.Module],
        device: Union[str, torch.device],
        cfg: Dict,
    ) -> None:
        super().__init__(cfg, model, device)

        # Hyperparameters
        self.learning_epochs = self.cfg["learning_epochs"]
        self.batch_size = self.cfg["batch_size"]
        self.learning_rate = self.cfg["learning_rate"]
        self.discount_factor = self.cfg["discount_factor"]
        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.target_tau = self.cfg.get("tau", 0.005)

        # Network
        if self.cfg["double_network"]:
            self.critic_1 = self.model.get("critic_1", None)
            self.critic_2 = self.model.get("critic_2", None)
            if self.critic_1 is None or self.critic_2 is None:
                raise ValueError("Safety agent requires critic_1 and critic_2.")
            self.critic_1 = self.critic_1.to(self.device)
            self.critic_2 = self.critic_2.to(self.device)
            self.checkpoint_modules["critic_1"] = self.critic_1
            self.checkpoint_modules["critic_2"] = self.critic_2

            self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
            self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
            self.target_critic_1.requires_grad_(False)
            self.target_critic_2.requires_grad_(False)
            self.target_critic_1.eval()
            self.target_critic_2.eval()
            self.checkpoint_modules["target_critic_1"] = self.target_critic_1
            self.checkpoint_modules["target_critic_2"] = self.target_critic_2

            self.optimizer = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self.learning_rate)
            self.checkpoint_modules["optimizer"] = self.optimizer
        else:
            self.critic_1 = self.model.get("critic_1", None)
            if self.critic_1 is None:
                raise ValueError("Safety agent requires critic_1 and critic_2.")
            self.critic_1 = self.critic_1.to(self.device)
            self.checkpoint_modules["critic_1"] = self.critic_1

            self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
            self.target_critic_1.requires_grad_(False)
            self.target_critic_1.eval()
            self.checkpoint_modules["target_critic_1"] = self.target_critic_1

            self.optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.learning_rate)
            self.checkpoint_modules["optimizer"] = self.optimizer

        self.set_running_mode("eval")

    @torch.no_grad()
    def predict(self, states: torch.Tensor) -> torch.Tensor:
        value_1, _, _ = self.critic_1(states, update_rms=False)
        # value_2, _, _ = self.critic_2(states, update_rms=False)
        # return torch.minimum(value_1, value_2)
        return value_1

    def _compute_target(self, 
                        next_states: torch.Tensor, 
                        g_values: torch.Tensor, 
                        next_g_values: torch.Tensor, 
                        segmend_end: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_value_1, _, _ = self.target_critic_1(next_states, update_rms=False)
            # next_value_2, _, _ = self.target_critic_2(next_states, update_rms=False)

            # next_values = torch.minimum(next_value_1, next_value_2)
            next_values = next_value_1
            next_values = torch.where(segmend_end, next_g_values, next_values)

            targets = ((1.0 - self.discount_factor) * g_values + self.discount_factor * torch.maximum(g_values, next_values))

        return targets

    def _update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.lerp_(param.data, self.target_tau)

            # for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            #     target_param.data.lerp_(param.data, self.target_tau)

    def _sync_target_rms(self) -> None:
        self.target_critic_1.critic_standardizer.load_state_dict(
            self.critic_1.critic_standardizer.state_dict()
        )
        # self.target_critic_2.critic_standardizer.load_state_dict(
        #     self.critic_2.critic_standardizer.state_dict()
        # )

    def update(self, batch) -> Dict[str, float]:
        self.set_running_mode("train")

        states, next_states, g_values, next_g_values, _, segment_end = batch
        states = states.to(self.device, non_blocking=True)
        next_states = next_states.to(self.device, non_blocking=True)
        g_values = g_values.to(self.device, non_blocking=True)
        next_g_values = next_g_values.to(self.device, non_blocking=True)
        segment_end = segment_end.to(self.device, non_blocking=True)

        if g_values.ndim == 1:
            g_values = g_values.unsqueeze(-1)
        if next_g_values.ndim == 1:
            next_g_values = next_g_values.unsqueeze(-1)
        if segment_end.ndim == 1:
            segment_end = segment_end.unsqueeze(-1)

        cumulative_loss = 0.0
        cumulative_loss_1 = 0.0
        cumulative_loss_2 = 0.0
        cumulative_value = 0.0
        cumulative_target = 0.0
        cumulative_gap = 0.0

        for epoch in range(self.learning_epochs):
            update_rms = epoch == 0

            value_1, _, _ = self.critic_1(states, update_rms=update_rms)
            # value_2, _, _ = self.critic_2(states, update_rms=update_rms)

            if epoch == 0:
                self._sync_target_rms()

            target_values = self._compute_target(next_states, g_values, next_g_values, segment_end)

            value_loss_1 = F.mse_loss(value_1, target_values)
            # value_loss_2 = F.mse_loss(value_2, target_values)
            # value_loss = 0.5 * (value_loss_1 + value_loss_2)
            value_loss = value_loss_1

            self.optimizer.zero_grad()
            value_loss.backward()

            if self.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_norm_clip)
                # nn.utils.clip_grad_norm_(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self.grad_norm_clip)

            self.optimizer.step()
            self._update_target()

            # predicted_values = torch.minimum(value_1, value_2)
            predicted_values = value_1

            cumulative_loss += value_loss.item()
            cumulative_loss_1 += value_loss_1.item()
            cumulative_loss_2 += 0.0
            # cumulative_loss_2 += value_loss_2.item()
            cumulative_value += predicted_values.detach().mean().item()
            cumulative_target += target_values.detach().mean().item()
            cumulative_gap += 0.0
            # cumulative_gap += (value_1.detach() - value_2.detach()).abs().mean().item()

        self._sync_target_rms()
        self.set_running_mode("eval")

        num_updates = max(self.learning_epochs, 1)

        return {
            "loss": cumulative_loss / num_updates,
            "loss_1": cumulative_loss_1 / num_updates,
            "loss_2": cumulative_loss_2 / num_updates,
            "value_mean": cumulative_value / num_updates,
            "target_mean": cumulative_target / num_updates,
            "critic_gap": cumulative_gap / num_updates,
        }

    @torch.no_grad()
    def evaluate(self, data_loader, threshold: float = 0.0) -> Dict[str, float]:
        self.set_running_mode("eval")
        self.target_critic_1.eval()
        # self.target_critic_2.eval()

        td_loss_sum = 0.0
        critic_gap_sum = 0.0
        total = 0

        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0

        diagnostics = {
                    "g": [],
                    "future": [],
                    "target": [],
                    "v1": [],
                    "v2": [],
                    "vmin": [],
                }

        for states, next_states, g_values, next_g_values, future_max_g, segment_end in data_loader:
            states = states.to(self.device, non_blocking=True)
            next_states = next_states.to(self.device, non_blocking=True)
            g_values = g_values.to(self.device, non_blocking=True)
            next_g_values = next_g_values.to(self.device, non_blocking=True)
            future_max_g = future_max_g.to(self.device, non_blocking=True)
            segment_end = segment_end.to(self.device, non_blocking=True)

            if g_values.ndim == 1:
                g_values = g_values.unsqueeze(-1)
            if next_g_values.ndim == 1:
                next_g_values = next_g_values.unsqueeze(-1)
            if future_max_g.ndim == 1:
                future_max_g = future_max_g.unsqueeze(-1)
            if segment_end.ndim == 1:
                segment_end = segment_end.unsqueeze(-1)

            value_1, _, _ = self.critic_1(states, update_rms=False)
            # value_2, _, _ = self.critic_2(states, update_rms=False)
            # predicted_values = torch.minimum(value_1, value_2)
            predicted_values = value_1

            target_values = self._compute_target(next_states, g_values, next_g_values, segment_end)

            loss_1 = F.mse_loss(value_1, target_values, reduction="sum")
            # loss_2 = F.mse_loss(value_2, target_values, reduction="sum")

            td_loss_sum += loss_1.item()
            critic_gap_sum += value_1.abs().sum().item()
            # td_loss_sum += 0.5 * (loss_1.item() + loss_2.item())
            # critic_gap_sum += (value_1 - value_2).abs().sum().item()
            total += states.shape[0]

            pred_risk = predicted_values > threshold
            real_risk = future_max_g > threshold

            # Collect only empirical risky samples for checking training stability
            tensors = {
                "g": g_values,
                "future": future_max_g,
                "target": target_values,
                "v1": value_1,
                "v2": value_1,
                "vmin": predicted_values,
            }
            for name, tensor in tensors.items():
                diagnostics[name].append(tensor[real_risk].detach().flatten().cpu())

            true_positive += (pred_risk & real_risk).sum().item()
            false_negative += ((~pred_risk) & real_risk).sum().item()
            false_positive += (pred_risk & (~real_risk)).sum().item()
            true_negative += ((~pred_risk) & (~real_risk)).sum().item()

        detection_rate = true_positive / max(true_positive + false_negative, 1)
        false_alarm_rate = false_positive / max(false_positive + true_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        accuracy = (true_positive + true_negative) / max(total, 1)
        balanced_accuracy = 0.5 * (detection_rate + specificity)
        real_risk_rate = (true_positive + false_negative) / max(total, 1)
        pred_risk_rate = (true_positive + false_positive) / max(total, 1)

        # Logging
        print("\n[RISK VALUE STABILITY]")
        for name, values in diagnostics.items():
            x = torch.cat(values)
            print(
                f"{name:<7}: "
                f"mean={x.mean().item():8.3f} | "
                f"p99={torch.quantile(x, 0.99).item():8.3f} | "
                f"max={x.max().item():8.3f}"
            )

        return {
            "td_loss": td_loss_sum / max(total, 1),
            "critic_gap": critic_gap_sum / max(total, 1),
            "detection_rate": detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "balanced_accuracy": balanced_accuracy,
            "accuracy": accuracy,
            "real_risk_rate": real_risk_rate,
            "pred_risk_rate": pred_risk_rate,
        }