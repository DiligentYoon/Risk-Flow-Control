from __future__ import annotations

import torch
import copy

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

from lib.domain_randomizer.commander import UniformNonHolonomicCommand

from ..R1_base_env import R1BaseEnv
from .R1_fall_env_cfg import R1FallEnvCfg, R1FallPlayEnvCfg


class R1FallEnv(R1BaseEnv):
    cfg: R1FallEnvCfg | R1FallPlayEnvCfg

    def __init__(self, cfg: R1FallEnvCfg | R1FallPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        total_body_ids, _ = self.contact_sensors.find_bodies(".*")
        self.allowed_collision_link_ids, _ = self.contact_sensors.find_bodies(self.cfg.allowed_collision_bodies)
        self.denied_collision_link_ids = [body_id for body_id in total_body_ids if body_id not in self.allowed_collision_link_ids]

        self.arm_deviation_joint_ids, _ = self._robot.find_joints([
            r".*_shoulder_(roll|pitch|yaw)_joint",
            r".*_elbow_joint",
            r".*_wrist_roll_joint",
            r"head_(pitch|yaw)_joint",
        ])

        self.swing_joint_ids, _ = self._robot.find_joints([r".*_shoulder_pitch_joint"])
        self.commands = UniformNonHolonomicCommand(self.cfg.commands, self._robot, self.device)
        self.mapping_sort_ids = torch.argsort(torch.tensor(self.total_arm_joint_ids + self.total_leg_joint_ids, device=self.device))

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
        self.foot_rot_w = torch.zeros((self.num_envs, 2, 4), dtype=torch.float, device=self.device)

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.update_phase_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.command_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.update_command_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.step_period = int(self.cfg.time_period / self.step_dt) * torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.full_step_period = int(2 * self.cfg.time_period / self.step_dt) * torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        self.contact_schedule = torch.zeros(self.num_envs, device=self.device)
        self.phase_sin = torch.zeros(self.num_envs, device=self.device)
        self.phase_cos = torch.zeros(self.num_envs, device=self.device)

        self.foot_on_swing = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.support_foot_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

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
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)

            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return

        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.6

        vel_des_arrow_scale, vel_des_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(
            scale=self.goal_vel_visualizer.cfg.markers["arrow"].scale,
            xy_velocity=self.commands.command_b,
        )

        vel_arrow_scale, vel_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(
            scale=self.current_vel_visualizer.cfg.markers["arrow"].scale,
            xy_velocity=self._robot.data.root_lin_vel_b[:, :2],
        )

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _setup_scene(self):
        super()._setup_scene()

        self.scene.clone_environments(copy_from_source=False)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.cfg.commands.num_envs = self.scene.num_envs
        self.cfg.commands.step_dt = self.step_dt

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.cfg.num_agents > 1:
            observations = {
                "arm": torch.cat([
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self.commands.command_b,
                    self.phase_sin.unsqueeze(-1),
                    self.phase_cos.unsqueeze(-1),
                    self._robot.data.joint_pos[:, self.total_arm_joint_ids],
                    self._robot.data.joint_vel[:, self.total_arm_joint_ids],
                    self.prev_actions["arm"],
                ], dim=-1),
                "leg": torch.cat([
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self.commands.command_b,
                    self.phase_sin.unsqueeze(-1),
                    self.phase_cos.unsqueeze(-1),
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
                self.commands.command_b,
                self.phase_sin.unsqueeze(-1),
                self.phase_cos.unsqueeze(-1),
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
                self.commands.command_b,
                self.phase_sin.unsqueeze(-1),
                self.phase_cos.unsqueeze(-1),
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
            states = None

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

        self.phase[env_ids] = 0
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = False
        self.command_count[env_ids] = 0
        self.update_command_ids[env_ids] = False

        self.foot_on_swing[env_ids] = 0
        self.foot_on_swing[env_ids, 0] = 1

        if self.cfg.num_agents > 1:
            self.prev_actions["leg"][env_ids] = 0.0
            self.prev_actions["arm"][env_ids] = 0.0
        else:
            self.prev_actions[env_ids] = 0.0

        self.commands.reset(env_ids)
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

        self.foot_rot_w[i] = self._robot.data.body_link_quat_w[i][:, self.ankle_x_link_ids]

        if env_ids is not None:
            self.support_foot_rot[i] = self.foot_rot_w[i, 1, :4]
        else:
            self.phase += 1 / self.full_step_period
            self.phase_count += 1
            self.update_phase_ids = self.phase_count >= self.full_step_period

            self.command_count += 1
            self.update_command_ids = self.command_count >= self.step_period

            phase_update_mask = self.update_phase_ids.clone()
            command_update_mask = self.update_command_ids.clone()

            self.phase[phase_update_mask] = 0
            self.phase_count[phase_update_mask] = 0
            self.command_count[command_update_mask] = 0

            if torch.any(self.update_command_ids):
                self.foot_on_swing[self.update_command_ids] = ~self.foot_on_swing[self.update_command_ids]

                left_support_mask = self.foot_on_swing[:, 0] == 0
                right_support_mask = self.foot_on_swing[:, 1] == 0

                left_combined_mask = self.update_command_ids & left_support_mask
                right_combined_mask = self.update_command_ids & right_support_mask

                self.support_foot_rot[left_combined_mask] = self.foot_rot_w[left_combined_mask, 0, :4]
                self.support_foot_rot[right_combined_mask] = self.foot_rot_w[right_combined_mask, 1, :4]

        self.contact_schedule[i] = smooth_sqr_wave(self.phase[i])

        self.phase_sin[i] = torch.sin(2 * torch.pi * self.phase[i])
        self.phase_cos[i] = torch.cos(2 * torch.pi * self.phase[i])

    def _update_viz_data(self):
        return self.extras

@torch.jit.script
def smooth_sqr_wave(phase):
    p = 2.0 * torch.pi * phase
    eps = 0.2
    return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2.0 + eps**2.0)