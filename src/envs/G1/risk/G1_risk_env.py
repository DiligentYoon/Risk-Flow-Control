# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply

from envs.G1.base.G1_base_env import G1BaseEnv
from envs.G1.risk.G1_risk_env_cfg import G1RiskEnvCfg


class G1RiskEnv(G1BaseEnv):
    """Intervention environment for the Risk-Flow actor-critic.

    Every episode starts from a risk-classified state drawn by the dataset reset event and runs for
    exactly H steps unless the robot falls.
    """

    cfg: G1RiskEnvCfg

    def __init__(self, cfg: G1RiskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Collision links. Anything that is not an allowed contact body counts towards a failure,
        # mirroring the termination that the frozen value function was trained against.
        total_body_ids, _ = self.contact_sensors.find_bodies(".*")
        self.allowed_collision_link_ids, _ = self.contact_sensors.find_bodies(self.cfg.allowed_collision_bodies)
        self.denied_collision_link_ids = [b for b in total_body_ids if b not in self.allowed_collision_link_ids]
        self.arm_collision_link_ids, _ = self.contact_sensors.find_bodies([r"waist_.*_link",
                                                                           r"torso_link",
                                                                           r".*_shoulder_.*_link",
                                                                           r".*_elbow_link",
                                                                           r".*_wrist_(roll|pitch)_link"])

        # Foot link
        self.foot_link_ids, _ = self._robot.find_bodies([r".*_ankle_.*_link"])

        # Torso joint
        self.torso_joint_ids, _ = self._robot.find_joints([r"waist_yaw_joint",])

        # Mass, used for the CoM height that gates the fall termination
        self.robot_mass = self._robot.data.default_mass.to(self.device)
        self.total_mass = self._robot.data.default_mass.sum(dim=-1).to(self.device)

        # Intermediate values
        self.root_pos_w          = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_rot_w          = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.root_lin_vel_w      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_ang_vel_w      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_lin_vel_b      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_ang_vel_b      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.CoM                 = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_heading        = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.projected_gravity   = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.joint_pos           = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.joint_vel           = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.contact_force       = torch.zeros((self.num_envs, self.contact_sensors.num_bodies), dtype=torch.float, device=self.device)

        # Contact forces on the bodies that are not allowed to touch anything
        self.illegal_force     = torch.zeros((self.num_envs, len(self.denied_collision_link_ids), 3), dtype=torch.float, device=self.device)
        self.illegal_arm_force = torch.zeros((self.num_envs, len(self.arm_collision_link_ids), 3), dtype=torch.float, device=self.device)

        # Geometry vector
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Regularization
        self.out_of_limits_joint    = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.out_of_limits_torque   = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.deviation_arms         = torch.zeros((self.num_envs, len(self.total_arm_joint_ids)), dtype=torch.float, device=self.device)
        self.deviation_legs         = torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), dtype=torch.float, device=self.device)
        self.deviation_torso        = torch.zeros((self.num_envs, len(self.torso_joint_ids)), dtype=torch.float, device=self.device)

        # Prev value
        if self.cfg.num_agents > 1:
            # Multi Agent
            self.prev_actions = {"leg": torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), device=self.device),
                                 "arm": torch.zeros((self.num_envs, len(self.total_arm_joint_ids)), device=self.device)}
        else:
            # Single Agent
            self.prev_actions = torch.zeros((self.num_envs, len(self._joint_dof_ids)), device=self.device)

        # Visualization
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "torso_rotation_visualizer"):
                self.torso_rotation_visualizer = VisualizationMarkers(self.cfg.torso_rotation_visualizer_cfg)
            self.torso_rotation_visualizer.set_visibility(True)
        else:
            if hasattr(self, "torso_rotation_visualizer"):
                self.torso_rotation_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        torso_pos = self._robot.data.body_link_pos_w[:, self.torso_link_ids].reshape(-1, 3)
        torso_rot = self._robot.data.body_link_quat_w[:, self.torso_link_ids].reshape(-1, 4)
        self.torso_rotation_visualizer.visualize(translations=torso_pos, orientations=torso_rot)


    def _setup_scene(self):
        super()._setup_scene()
        # Without the clone the articulation view matches a single prim
        # while the scene claims num_envs, and the first indexed write trips an out-of-bounds assert.
        self.scene.clone_environments(copy_from_source=False)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _get_observations(self) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.cfg.num_agents > 1:
            # Multi Agent
            observations = {
                "arm": torch.cat(
                    [
                        self.root_lin_vel_b,                                # [E, 3]
                        self.root_ang_vel_b,                                # [E, 3]
                        self.projected_gravity,                             # [E, 3]
                        self.joint_pos[:, self.total_arm_joint_ids],        # [E, 17]
                        self.joint_vel[:, self.total_arm_joint_ids],        # [E, 17]
                        self.prev_actions["arm"],                           # [E, 17]
                    ],
                    dim=-1
                ),
                "leg": torch.cat(
                    [
                        self.root_lin_vel_b,                                # [E, 3]
                        self.root_ang_vel_b,                                # [E, 3]
                        self.projected_gravity,                             # [E, 3]
                        self.joint_pos[:, self.total_leg_joint_ids],        # [E, 12]
                        self.joint_vel[:, self.total_leg_joint_ids],        # [E, 12]
                        self.prev_actions["leg"],                           # [E, 12]
                    ],
                    dim=-1
                )
            }
        else:
            # Single Agent.
            observations = torch.cat(
                [
                    self.root_lin_vel_b,                                    # [E, 3]
                    self.root_ang_vel_b,                                    # [E, 3]
                    self.projected_gravity,                                 # [E, 3]
                    self.joint_pos - self._robot.data.default_joint_pos,    # [E, 29]
                    self.joint_vel,                                         # [E, 29]
                    self.prev_actions,                                      # [E, 29]
                ], dim=-1)

        return observations


    def _get_states(self) -> dict[str, torch.Tensor] | torch.Tensor | None:
        if self.cfg.num_agents > 1:
            # Multi Agent
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
            shared_states = torch.cat(
                [
                    self.root_pos_w[:, 2:3],                            # [E, 1]
                    self.root_lin_vel_b,                                # [E, 3]
                    self.root_ang_vel_b,                                # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.joint_pos[:, total_joint_ids],                 # [E, 29]
                    self.joint_vel[:, total_joint_ids],                 # [E, 29]
                    self.prev_actions["arm"],                           # [E, 17]
                    self.prev_actions["leg"],                           # [E, 12]
                ], dim=-1)

            states = {
                "arm": shared_states,
                "leg": shared_states
            }
        else:
            states = None

        return states


    def get_torque_model(self) -> dict:
        """Constants of the analytic PD torque surrogate.

        The action is a joint-position offset and the low-level controller is PD, so the applied
        torque is an analytic function of the action::

            tau(s, a) = Kp * (action_scale * a - (q - q_default)) - Kd * q_dot

        Both ``q - q_default`` and ``q_dot`` are already channels of the constraint state, so the
        actor's control-cost term needs nothing beyond the gains and the offsets published here.
        This environment owns that layout, so it publishes it rather than letting the agent
        hard-code indices into a tensor it does not define.

        The gains are read from environment 0. Domain randomization can spread them across
        environments, and a replayed batch mixes environments, so per-sample gains are not
        recoverable -- the nominal ones are the right constant for a regularizer.
        """
        return {
            "action_scale": self.cfg.action_scale_factor,
            "stiffness": self._robot.data.joint_stiffness[0, self._joint_dof_ids].clone(),
            "damping": self._robot.data.joint_damping[0, self._joint_dof_ids].clone(),
            "joint_pos_id": self.cfg.constraint_joint_pos_start,
        }

    def _get_constraint_states(self) -> torch.Tensor:
        """Input of the frozen safety value function V_N."""
        return torch.cat(
            [
                self.root_lin_vel_b,                                    # [E, 3]
                self.root_ang_vel_b,                                    # [E, 3]
                self.projected_gravity,                                 # [E, 3]
                self.joint_pos - self._robot.data.default_joint_pos,    # [E, 29]
                self.joint_vel,                                         # [E, 29]
            ], dim=-1)            


    def _get_rewards(self) -> torch.Tensor:
        # The algorithm consumes no environment reward: the learning signal is the increment of
        # V_N over the constraint states. This stays a zero channel.
        if self.cfg.num_agents > 1:
            # Multi Agent
            rewards = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
            # Dictionary key order (alphabetical order in dictionary)
            self.prev_actions = {k: v.clone() for k, v in self.actions.items()}
        else:
            # Single Agent
            rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.prev_actions = self.actions.clone()

        return rewards


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # The time-out is the recovery deadline H, so it fires on every episode that did not fail.
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_fall = self.CoM[:, 2] <= self.cfg.termination_height
        died_collision = torch.any(torch.norm(self.illegal_force, dim=-1) > 1.0, dim=1)
        # died_arm_collision = torch.any(torch.norm(self.illegal_arm_force, dim=-1) > 1.0, dim=1)

        died = died_collision & base_fall
        # died = (died_collision & base_fall) | died_arm_collision

        # The two reasons stay mutually exclusive: a fall is absorbing, a time-out bootstraps.
        time_out = time_out & ~died

        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Randomization by Event-based randomizer
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Prev action conditioning by dataset
        staged = getattr(self, "_dataset_reset_prev_action", None)
        if staged is not None:
            if self.cfg.num_agents > 1:
                # Multi agent
                self.prev_actions["arm"][env_ids] = staged[env_ids][:, self.total_arm_joint_ids]
                self.prev_actions["leg"][env_ids] = staged[env_ids][:, self.total_leg_joint_ids]
            else:
                # Single agent
                self.prev_actions[env_ids] = staged[env_ids]
        else:
            # Fallback: dataset reset event not registered for this env.
            if self.cfg.num_agents > 1:
                self.prev_actions["arm"][env_ids] = 0.0
                self.prev_actions["leg"][env_ids] = 0.0
            else:
                self.prev_actions[env_ids] = 0.0

        self._compute_intermediate_values(env_ids)


    def _update_viz_data(self):
        """Fill the simulator-side channels of ``viz_data``.

        The algorithm-side channels (``risk_value``, ``risk_flow``, ``terminal_risk``) are left at
        their declared defaults: they are functions of the frozen ``V_N`` and of the critic, which
        the environment does not hold, so ``play.py`` writes them in before appending a frame.

        The tensors are cloned. Everything here is a persistent per-env buffer that the next step
        rewrites in place, and the plotter appends the frame *after* that step has run.
        """
        max_torque = torch.max(torch.abs(self._robot.data.applied_torque), dim=-1).values      # [E,]
        action_magnitude = torch.mean(torch.abs(self.prev_actions), dim=-1)                    # [E,]

        extras = copy.deepcopy(self.extras)
        extras["viz_data"]["action_magnitude"] = action_magnitude.clone()
        extras["viz_data"]["max_torque"] = max_torque.clone()
        extras["viz_data"]["CoM_height"] = self.CoM[:, 2].clone()

        return extras


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        # Root Pose & Velocity
        self.root_pos_w[i], self.root_rot_w[i] = self._robot.data.root_pos_w[i], self._robot.data.root_quat_w[i]
        self.root_lin_vel_w[i], self.root_ang_vel_w[i] = self._robot.data.root_lin_vel_w[i], self._robot.data.root_ang_vel_w[i]
        self.root_lin_vel_b[i], self.root_ang_vel_b[i] = self._robot.data.root_lin_vel_b[i], self._robot.data.root_ang_vel_b[i]
        # Center of Mass (CoM)
        self.CoM[i] = (self._robot.data.body_link_pos_w[i] * self.robot_mass[i].unsqueeze(-1)).sum(dim=1) / self.total_mass[i].unsqueeze(-1)
        # Heading
        forward_root_w = quat_apply(self._robot.data.root_quat_w[i], self.forward_vec[i])
        self.root_heading[i] = torch.atan2(forward_root_w[:, 1], forward_root_w[:, 0]).unsqueeze(-1)
        # Attitude
        self.projected_gravity[i] = self._robot.data.projected_gravity_b[i]
        # Joint Angle & Velocity
        self.joint_pos[i], self.joint_vel[i] = self._robot.data.joint_pos[i], self._robot.data.joint_vel[i]
        # Contact forces
        self.illegal_force[i] = self.contact_sensors.data.net_forces_w[i][:, self.denied_collision_link_ids]
        self.illegal_arm_force[i] = self.contact_sensors.data.net_forces_w[i][:, self.arm_collision_link_ids]
        self.contact_force[i] = torch.norm(self.contact_sensors.data.net_forces_w[i], dim=-1)
        # Regularization Parameter
        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)
        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self._robot.data.joint_effort_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.deviation_arms[i]       = self.joint_pos[i][:, self.total_arm_joint_ids] - self._robot.data.default_joint_pos[i][:, self.total_arm_joint_ids]
        self.deviation_legs[i]       = self.joint_pos[i][:, self.total_leg_joint_ids] - self._robot.data.default_joint_pos[i][:, self.total_leg_joint_ids]
        self.deviation_torso[i]      = self.joint_pos[i][:, self.torso_joint_ids] - self._robot.data.default_joint_pos[i][:, self.torso_joint_ids]
