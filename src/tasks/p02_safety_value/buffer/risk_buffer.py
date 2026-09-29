from __future__ import annotations

import os
from typing import Optional, Union

import torch


class RiskBuffer:
    """Replay buffer for initial-condition collection."""

    def __init__(self, capacity: int, joint_dim: int, device: Optional[Union[str, torch.device]] = None) -> None:
        self.capacity = int(capacity)
        self.joint_dim = int(joint_dim)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Schema: key -> last-dim size
        self._key_dims = {
            "root_pos_offset_w": 3,
            "root_quat_w": 4,
            "root_lin_vel_w": 3,
            "root_ang_vel_w": 3,
            "joint_pos": self.joint_dim,
            "joint_vel": self.joint_dim,
            "prev_action": self.joint_dim,
            "risk_score": 1,
        }

        self.tensors: dict[str, torch.Tensor] = {}
        for k, d in self._key_dims.items():
            self.tensors[k] = torch.zeros((self.capacity, d), dtype=torch.float32, device=self.device)

        self.write_idx = 0

    # -------- writes --------

    def add(self, snapshot: dict[str, torch.Tensor], risk_scores: torch.Tensor, mask: torch.Tensor) -> int:
        """Insert one multi-env step."""
        scores = risk_scores.flatten()
        mask = mask.flatten().bool()

        free = self.capacity - self.write_idx
        if free <= 0:
            return 0

        count = int(mask.sum().item())
        n = min(count, free)
        if n == 0:
            return 0

        env_idx = torch.nonzero(mask, as_tuple=False).flatten()[:n]
        start = self.write_idx
        end = start + n

        self.tensors["root_pos_offset_w"][start:end].copy_(snapshot["root_pos_offset_w"][env_idx])
        self.tensors["root_quat_w"][start:end].copy_(snapshot["root_quat_w"][env_idx])
        self.tensors["root_lin_vel_w"][start:end].copy_(snapshot["root_lin_vel_w"][env_idx])
        self.tensors["root_ang_vel_w"][start:end].copy_(snapshot["root_ang_vel_w"][env_idx])
        self.tensors["joint_pos"][start:end].copy_(snapshot["joint_pos"][env_idx])
        self.tensors["joint_vel"][start:end].copy_(snapshot["joint_vel"][env_idx])
        self.tensors["prev_action"][start:end].copy_(snapshot["prev_action"][env_idx])
        self.tensors["risk_score"][start:end, 0].copy_(scores[env_idx])

        self.write_idx = end

        return n

    # -------- status --------

    def is_full(self) -> bool:
        """True if every bucket has reached capacity."""
        return self.write_idx >= self.capacity

    def fill_status(self) -> tuple[int, int]:
        """Return ``{bucket: (current, capacity)}`` for each bucket."""
        return (self.write_idx, self.capacity)

    # -------- save --------

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        n = self.write_idx
        payload = {k: v[:n].detach().cpu().contiguous() for k, v in self.tensors.items()}
        torch.save(payload, os.path.join(save_dir, f"risk_init_dataset.pt"))