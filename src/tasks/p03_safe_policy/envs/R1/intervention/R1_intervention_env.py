from __future__ import annotations

import torch
import copy

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

from ..R1_base_env import R1BaseEnv
from .R1_intervention_env_cfg import R1InterventionEnvCfg, R1InterventionPlayEnvCfg


class R1InterventionEnv(R1BaseEnv):
    cfg: R1InterventionEnvCfg | R1InterventionPlayEnvCfg

    def __init__(self, cfg: R1InterventionEnvCfg | R1InterventionPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        total_body_ids, _ = self.contact_sensors.find_bodies(".*")
        self.allowed_collision_link_ids, _ = self.contact_sensors.find_bodies(self.cfg.allowed_collision_bodies)
        self.denied_collision_link_ids = [body_id for body_id in total_body_ids if body_id not in self.allowed_collision_link_ids]

        if self.cfg.num_agents > 1:
            self.cfg.action_scale_factor["arm"][1] = self.total_arm_joint_ids
            self.cfg.action_scale_factor["leg"][1] = self.total_leg_joint_ids

        self.root_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_rot_w = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.root_lin_vel_b = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_ang_vel_b = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_heading = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.projected_gravity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)

        if self.cfg.num_agents > 1:
            self.prev_actions = {
                                    "leg": torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), device=self.device),
                                    "arm": torch.zeros((self.num_envs, len(self.total_arm_joint_ids)), device=self.device),
                                }
        else:
            self.prev_actions = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    def _setup_scene(self):
        super()._setup_scene()

        self.scene.clone_environments(copy_from_source=False)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.cfg.num_agents > 1:
            observations = {
                "arm": torch.cat([
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos[:, self.total_arm_joint_ids],
                    self._robot.data.joint_vel[:, self.total_arm_joint_ids],
                    self.prev_actions["arm"],
                ], dim=-1),
                "leg": torch.cat([
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos[:, self.total_leg_joint_ids],
                    self._robot.data.joint_vel[:, self.total_leg_joint_ids],
                    self.prev_actions["leg"],
                ], dim=-1),
            }
        else:
            observations = torch.cat([
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos,
                self._robot.data.joint_vel,
                self.prev_actions,
            ], dim=-1)

        return observations

    def _get_states(self) -> dict[str, torch.Tensor]:
        if self.cfg.num_agents > 1:
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids

            shared_states = torch.cat([
                self._robot.data.root_pos_w[:, 2:3],
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, total_joint_ids],
                self._robot.data.joint_vel[:, total_joint_ids],
                self.prev_actions["leg"],
                self.prev_actions["arm"],
            ], dim=-1)

            states = {
                "arm": shared_states,
                "leg": shared_states,
            }
        else:
            states = torch.cat([
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos,
                self._robot.data.joint_vel,
                self.prev_actions,
            ], dim=-1)

        return states

    def _get_safety_states(self):
        return torch.cat([self._robot.data.root_lin_vel_b,
                          self._robot.data.root_ang_vel_b,
                          self._robot.data.projected_gravity_b,
                          self._robot.data.joint_pos,
                          self._robot.data.joint_vel], dim=-1)

    def _get_safety_values(self):
        # safety value
        base_tilt = (torch.atan2(torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=-1), 
                                 -self._robot.data.projected_gravity_b[:, 2]) - self.cfg.phi_max) / self.cfg.phi_max
        base_height = (self.cfg.termination_height - self._robot.data.root_pos_w[:, 2]) / self.cfg.termination_height

        return torch.max(base_tilt, base_height)

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.num_agents > 1:
            rewards = torch.zeros((self.num_envs, self.cfg.num_agents), dtype=torch.float32, device=self.device)
            self.prev_actions = {k: v.clone() for k, v in self.actions.items()}
        else:
            rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            self.prev_actions = self.actions.clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        critical_contact_forces = torch.norm(self.contact_sensors.data.net_forces_w_history[:, :, self.denied_collision_link_ids], dim=-1)
        died_fall = self.root_pos_w[:, 2] <= self.cfg.termination_height
        died_collision = torch.any(torch.any(critical_contact_forces > 1.0, dim=-1), dim=-1)

        died = died_fall & died_collision

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._compute_intermediate_values(env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES

        self.root_pos_w[i] = self._robot.data.root_pos_w[i]
        self.root_rot_w[i] = self._robot.data.root_quat_w[i]
        self.root_lin_vel_b[i] = self._robot.data.root_lin_vel_b[i]
        self.root_ang_vel_b[i] = self._robot.data.root_ang_vel_b[i]

        forward_root_w = quat_apply(self._robot.data.root_quat_w[i], self.forward_vec[i])
        self.root_heading[i] = torch.atan2(forward_root_w[:, 1], forward_root_w[:, 0]).unsqueeze(-1)
        self.projected_gravity[i] = self._robot.data.projected_gravity_b[i]

        self.joint_pos[i] = self._robot.data.joint_pos[i]
        self.joint_vel[i] = self._robot.data.joint_vel[i]

    def _update_viz_data(self):
        return self.extras