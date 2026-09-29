from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from lib.env.env import Env

class reset_state_from_dataset(ManagerTermBase):
    """
    Reset both root state and joint state by sampling from a precomputed 
    risk-classified initial-condition dataset.
    
    """
    REQUIRED_KEYS: tuple[str, ...] = (
        "root_quat_w",
        "root_lin_vel_w", 
        "root_ang_vel_w",
        "joint_pos", 
        "joint_vel",
        "prev_action",
    )

    def __init__(self, cfg: EventTermCfg, env: Env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._device = self.asset.device
        self._loaded: bool = False
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_length = 0

    def _load_dataset(self, file_path: str) -> None:
        if file_path is not None:
            payload = torch.load(file_path, map_location=self._device)
            self._buffer = {k: payload[k].to(self._device) for k in self.REQUIRED_KEYS}
            self._buffer_length = payload["root_quat_w"].shape[0]


    def __call__(self,
                 env: Env,
                 env_ids: torch.Tensor,
                 dataset_path: str = None,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        if not self._loaded:
            self._load_dataset(dataset_path)
            self._loaded = True

        n = int(env_ids.numel())
        num_joints = self.asset.num_joints

        root_quat       = torch.zeros((n, 4), device=self._device)
        root_lin_vel    = torch.zeros((n, 3), device=self._device)
        root_ang_vel    = torch.zeros((n, 3), device=self._device)
        joint_pos       = torch.zeros((n, num_joints), device=self._device)
        joint_vel       = torch.zeros((n, num_joints), device=self._device)
        prev_action     = torch.zeros((n, num_joints), device=self._device)

        if dataset_path is not None:
            row = torch.randint(0, self._buffer_length, (n,), device=self._device)
            root_quat       = self._buffer["root_quat_w"][row]
            root_lin_vel    = self._buffer["root_lin_vel_w"][row]
            root_ang_vel    = self._buffer["root_ang_vel_w"][row]
            joint_pos       = self._buffer["joint_pos"][row]
            joint_vel       = self._buffer["joint_vel"][row]
            prev_action     = self._buffer["prev_action"][row]
        else:
            joint_pos = self.asset.data.default_joint_pos[env_ids]

        root_pos = env.scene.env_origins[env_ids]

        self.asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
        self.asset.write_root_velocity_to_sim(torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)
        self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # Stage prev_action for env._reset_idx to consume.
        if env.cfg.num_agents > 1:
            env.prev_actions["arm"][env_ids] = prev_action[:, self.total_arm_joint_ids]
            env.prev_actions["leg"][env_ids] = prev_action[:, self.total_leg_joint_ids]
        else:
            env.prev_actions[env_ids] = prev_action