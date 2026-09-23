from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils

from isaaclab.managers import SceneEntityCfg

from ....safe_value_env import SafeValueEnv

def push_and_log(
    env: SafeValueEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply a randomized root-velocity disturbance and mark the push event."""
    asset = env.scene[asset_cfg.name]

    before = asset.data.root_vel_w[env_ids].clone()
    # sample random velocities in body frame
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    delta_b = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], before.shape, device=asset.device)

    # frame transformation (Body -> World)
    root_quat = asset.data.root_quat_w[env_ids]
    delta_w_lin = math_utils.quat_apply(root_quat, delta_b[:, :3])
    delta_w_ang = math_utils.quat_apply(root_quat, delta_b[:, 3:])

    # set the velocities into the physics simulation
    vel_w = before + torch.cat([delta_w_lin, delta_w_ang], dim=-1)
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    env.push_event_buf[env_ids] = True