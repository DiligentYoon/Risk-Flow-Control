from __future__ import annotations

import torch
import copy

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, euler_xyz_from_quat, quat_apply, matrix_from_quat

from lib.domain_randomizer.commander import UniformNonHolonomicCommand

from ..R1_base_env import R1BaseEnv
from .R1_loco_env_cfg import R1LocoEnvCfg, R1LocoPlayEnvCfg


class R1LocoEnv(R1BaseEnv):
    cfg: R1LocoEnvCfg | R1LocoPlayEnvCfg

    def __init__(self, cfg: R1LocoEnvCfg | R1LocoPlayEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.root_lin_vel_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_lin_vel_b = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_ang_vel_b = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_heading = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.projected_gravity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.command_inputs_b = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.command_heading = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.contact_time = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.in_contact = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
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

        self.out_of_limits_joint = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.out_of_limits_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.joint_deviations = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)

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
                    self.root_lin_vel_b,
                    self.root_ang_vel_b,
                    self.projected_gravity,
                    self.command_inputs_b,
                    self.phase_sin.unsqueeze(-1),
                    self.phase_cos.unsqueeze(-1),
                    self.joint_pos[:, self.total_arm_joint_ids],
                    self.joint_vel[:, self.total_arm_joint_ids],
                    self.prev_actions["arm"],
                ], dim=-1),
                "leg": torch.cat([
                    self.root_lin_vel_b,
                    self.root_ang_vel_b,
                    self.projected_gravity,
                    self.command_inputs_b,
                    self.phase_sin.unsqueeze(-1),
                    self.phase_cos.unsqueeze(-1),
                    self.joint_pos[:, self.total_leg_joint_ids],
                    self.joint_vel[:, self.total_leg_joint_ids],
                    self.prev_actions["leg"],
                ], dim=-1),
            }
        else:
            observations = torch.cat([
                self.root_lin_vel_b,
                self.root_ang_vel_b,
                self.projected_gravity,
                self.command_inputs_b,
                self.phase_sin.unsqueeze(-1),
                self.phase_cos.unsqueeze(-1),
                self.joint_pos,
                self.joint_vel,
                self.prev_actions,
            ], dim=-1)

        return observations

    def _get_states(self) -> dict[str, torch.Tensor]:
        if self.cfg.num_agents > 1:
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids

            shared_states = torch.cat([
                self.root_pos_w[:, 2:3],
                self.root_lin_vel_b,
                self.root_ang_vel_b,
                self.projected_gravity,
                self.command_inputs_b,
                self.phase_sin.unsqueeze(-1),
                self.phase_cos.unsqueeze(-1),
                self.joint_pos[:, total_joint_ids],
                self.joint_vel[:, total_joint_ids],
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

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self.command_inputs_b[:, :2] - self.root_lin_vel_b[:, :2]), dim=-1)
        ang_vel_error = torch.sum(torch.abs(self.command_inputs_b[:, 2] - self.root_ang_vel_b[:, 2]))
        heading_error = torch.square(wrap_to_pi(self.command_heading[:, 0] - self.root_heading[:, 0]))
        height_error = torch.square(self.root_pos_w[:, 2] - self.cfg.target_height)

        lin_vel_rewards = torch.exp(-lin_vel_error / 0.2)
        ang_vel_rewards = torch.exp(-ang_vel_error / 0.2)
        heading_rewards = torch.exp(-heading_error / 0.1)
        height_rewards = torch.exp(-height_error / 0.1)

        tilting = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        flat_rewards = torch.exp(-tilting / 0.1)

        diff = self.in_contact[:, 1].float() - self.in_contact[:, 0].float()
        gait_reward = diff * self.contact_schedule

        terminate_penalty = -self.reset_terminated.float()

        support_x, support_y, _ = euler_xyz_from_quat(self.support_foot_rot)
        support_xy = torch.stack([support_x, support_y], dim=-1)
        support_xy = abs(wrap_to_pi(support_xy))
        support_xy_penalty = -torch.sum(support_xy, dim=-1)

        joint_deviation_penalty_hip_xz = -torch.sum(torch.abs(self.joint_deviations[:, self.hip_xz_joint_ids]), dim=-1)
        joint_deviation_penalty_arm = -torch.sum(torch.abs(self.joint_deviations[:, self.total_arm_joint_ids]), dim=1)

        ang_vel_xy_penalty = -torch.sum(torch.square(self.root_ang_vel_b[:, :2]), dim=1)
        lin_vel_z_penalty = -torch.square(self.root_lin_vel_w[:, 2])

        joint_limit_penalty_leg = -torch.sum(self.out_of_limits_joint[:, self.total_leg_joint_ids], dim=1)
        joint_torque_limit_penalty_leg = -torch.sum(self.out_of_limits_torque[:, self.total_leg_joint_ids], dim=1)
        joint_torque_penalty_leg = -torch.sum(torch.square(self._robot.data.applied_torque[:, self.total_leg_joint_ids]), dim=1)
        joint_vel_penalty_leg = -torch.sum(torch.square(self.joint_vel[:, self.total_leg_joint_ids]), dim=1)

        if self.cfg.num_agents > 1:
            action_rate_penalty_leg = -torch.sum(torch.square(self.actions["leg"] - self.prev_actions["leg"]), dim=1)
        else:
            action_rate_penalty_leg = -torch.sum(torch.square(self.actions[:, self.total_leg_joint_ids] - self.prev_actions[:, self.total_leg_joint_ids]), dim=1)

        joint_limit_penalty_arm = -torch.sum(self.out_of_limits_joint[:, self.total_arm_joint_ids], dim=1)
        joint_torque_limit_penalty_arm = -torch.sum(self.out_of_limits_torque[:, self.total_arm_joint_ids], dim=1)
        joint_torque_penalty_arm = -torch.sum(torch.square(self._robot.data.applied_torque[:, self.total_arm_joint_ids]), dim=1)
        joint_vel_penalty_arm = -torch.sum(torch.square(self.joint_vel[:, self.total_arm_joint_ids]), dim=1)

        if self.cfg.num_agents > 1:
            action_rate_penalty_arm = -torch.sum(torch.square(self.actions["arm"] - self.prev_actions["arm"]), dim=1)
        else:
            action_rate_penalty_arm = -torch.sum(torch.square(self.actions[:, self.total_arm_joint_ids] - self.prev_actions[:, self.total_arm_joint_ids]), dim=1)

        common_rewards = (
            self.cfg.r_flat * flat_rewards
            + self.cfg.r_track_ang_vel * ang_vel_rewards
            + self.cfg.r_track_height * height_rewards
            + self.cfg.p_ang_vel_xy * ang_vel_xy_penalty
            + self.cfg.p_lin_vel_z * lin_vel_z_penalty
            + self.cfg.r_track_heading * heading_rewards
            + self.cfg.p_termination * terminate_penalty
        )

        arm_specific_rewards = (
            self.cfg.p_deviation_arm * joint_deviation_penalty_arm
            + self.cfg.p_limits * joint_limit_penalty_arm
            + self.cfg.p_joint_torque_limit * joint_torque_limit_penalty_arm
            + self.cfg.p_joint_torque * joint_torque_penalty_arm
            + self.cfg.p_joint_vel * joint_vel_penalty_arm
            + self.cfg.p_action_rate * action_rate_penalty_arm
        )

        leg_specific_rewards = (
            self.cfg.r_track_lin_vel * lin_vel_rewards
            + self.cfg.r_feet_gait * gait_reward
            + self.cfg.p_support_xy * support_xy_penalty
            + self.cfg.p_deviation_hip * joint_deviation_penalty_hip_xz
            + self.cfg.p_limits * joint_limit_penalty_leg
            + self.cfg.p_joint_torque_limit * joint_torque_limit_penalty_leg
            + self.cfg.p_joint_torque * joint_torque_penalty_leg
            + self.cfg.p_joint_vel * joint_vel_penalty_leg
            + self.cfg.p_action_rate * action_rate_penalty_leg
        )

        if self.cfg.num_agents > 1:
            arm_rewards = common_rewards + arm_specific_rewards
            leg_rewards = common_rewards + leg_specific_rewards
            rewards = torch.stack([arm_rewards, leg_rewards], dim=-1)
            self.prev_actions = {k: v.clone() for k, v in self.actions.items()}
        else:
            rewards = common_rewards + arm_specific_rewards + leg_specific_rewards
            self.prev_actions = self.actions.clone()

        self.extras["reward"] = {
            "Task Reward / Common_Angular_Velocity": ang_vel_rewards,
            "Task Reward / Common_Flat": flat_rewards,
            "Task Reward / Common_Heading": heading_rewards,
            "Task Reward / Leg_Gait": gait_reward,
            "Task Reward / Leg_Linear_Velocity": lin_vel_rewards,
            "Task Penalty / Common_Ang_Vel_XY": ang_vel_xy_penalty,
            "Task Penalty / Common_Lin_Vel_Z": lin_vel_z_penalty,
            "Task Penalty / Arm_Deviation": joint_deviation_penalty_arm,
            "Task Penalty / Arm_Joint_Limit": joint_limit_penalty_arm,
            "Task Penalty / Arm_Torque_Limit": joint_torque_limit_penalty_arm,
            "Task Penalty / Arm_Torque": joint_torque_penalty_arm,
            "Task Penalty / Arm_Vel": joint_vel_penalty_arm,
            "Task Penalty / Arm_Action_Rate": action_rate_penalty_arm,
            "Task Penalty / Leg_Support_XY": support_xy_penalty,
            "Task Penalty / Leg_Hip_XZ_Deviation": joint_deviation_penalty_hip_xz,
            "Task Penalty / Leg_Joint_Limit": joint_limit_penalty_leg,
            "Task Penalty / Leg_Torque_Limit": joint_torque_limit_penalty_leg,
            "Task Penalty / Leg_Torque": joint_torque_penalty_leg,
            "Task Penalty / Leg_Vel": joint_vel_penalty_leg,
            "Task Penalty / Leg_Action_Rate": action_rate_penalty_leg,
        }

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        critical_contact_forces = torch.norm(self.contact_sensors.data.net_forces_w_history[:, :, self.denied_collision_link_ids], dim=-1)
        died_fall = self.root_pos_w[:, 2] <= self.cfg.termination_height
        died_collision = torch.any(torch.any(critical_contact_forces > 1.0, dim=-1), dim=-1)
        died_ang = (torch.norm(self.root_ang_vel_b[:, :3], dim=-1) >= self.cfg.termination_ang_vel)

        died = (died_fall & died_collision) | died_ang

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
        self.root_lin_vel_w[i] = self._robot.data.root_lin_vel_w[i]
        self.root_lin_vel_b[i] = self._robot.data.root_lin_vel_b[i]
        self.root_ang_vel_b[i] = self._robot.data.root_ang_vel_b[i]

        forward_root_w = quat_apply(self._robot.data.root_quat_w[i], self.forward_vec[i])
        self.root_heading[i] = torch.atan2(forward_root_w[:, 1], forward_root_w[:, 0]).unsqueeze(-1)
        self.projected_gravity[i] = self._robot.data.projected_gravity_b[i]

        self.joint_pos[i] = self._robot.data.joint_pos[i]
        self.joint_vel[i] = self._robot.data.joint_vel[i]

        self.command_inputs_b[i] = self.commands.command_b[i]
        self.command_heading[i] = self.commands.heading[i]

        self.contact_time[i] = self.contact_sensors.data.current_contact_time[i][:, self.ankle_contact_roll_link_ids]
        self.in_contact[i] = self.contact_time[i] > 0.0

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

        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)

        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self._robot.data.joint_effort_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.joint_deviations[i] = self.joint_pos[i] - self._robot.data.default_joint_pos[i]

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * torch.pi
    angles -= 2 * torch.pi * (angles > torch.pi)
    return angles


@torch.jit.script
def smooth_sqr_wave(phase):
    p = 2.0 * torch.pi * phase
    eps = 0.2
    return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2.0 + eps**2.0)