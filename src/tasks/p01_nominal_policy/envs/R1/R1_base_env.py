from __future__ import annotations

from abc import abstractmethod

import torch

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

from lib.env.env import Env

from .R1_base_env_cfg import R1BaseEnvCfg


class R1BaseEnv(Env):

    # Load config file
    cfg: R1BaseEnvCfg

    def __init__(self, cfg: R1BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Environment IDs
        self.total_env_ids = torch.arange(self.num_envs, device=self.device)

        # Joint Limits
        self.joint_pos_limits = self._robot.data.joint_pos_limits
        self.joint_vel_limits = self._robot.data.joint_vel_limits
        self.soft_joint_torque_limits = (self.cfg.soft_torque_limit * self._robot.data.joint_effort_limits)

        # All actuated joints
        self._joint_dof_ids, self._joint_dof_names = (self._robot.find_joints(".*"))

        # Leg joints
        self.total_leg_joint_ids, self.total_leg_joint_names = (
            self._robot.find_joints(
                [
                    r".*_hip_(pitch|roll|yaw)_joint",
                    r".*_knee_joint",
                    r".*_ankle_(pitch|roll)_joint",
                ]
            )
        )

        # Upper-body joints
        self.total_arm_joint_ids, _ = (
            self._robot.find_joints(
                [
                    r"waist_(roll|yaw)_joint",
                    r".*_shoulder_(pitch|roll|yaw)_joint",
                    r".*_elbow_joint",
                    r".*_wrist_roll_joint",
                    r"head_(pitch|yaw)_joint",
                ]
            )
        )

        # Specific joint groups

        # Upper-body without pitch
        self.deviation_arm_joint_ids, _ = (
            self._robot.find_joints(
                [
                    r"waist_(roll|yaw)_joint",
                    r".*_shoulder_(roll|yaw)_joint",
                    # r".*_elbow_joint",
                    r".*_wrist_roll_joint",
                    r"head_(pitch|yaw)_joint",
                ]
            )
        )

        self.swing_arm_joint_ids, _ = self._robot.find_joints(
            [
                r".*_shoulder_pitch_joint",
                r".*_elbow_joint",
            ]
        )

        # Hip yaw / roll
        self.hip_xz_joint_ids, _ = self._robot.find_joints(
            [
                r".*_hip_yaw_joint",
                r".*_hip_roll_joint",
            ]
        )

        # Knee
        self.knee_joint_ids, _ = self._robot.find_joints(
            [
                r".*_knee_joint",
            ]
        )

        # Ankle pitch / roll
        self.ankle_xy_joint_ids, _ = self._robot.find_joints(
            [
                r".*_ankle_pitch_joint",
                r".*_ankle_roll_joint",
            ]
        )

        # Waist joints
        self.torso_joint_ids, _ = self._robot.find_joints(
            [
                r"waist_(roll|yaw)_joint",
            ]
        )

        # Head joints
        self.head_joint_ids, _ = self._robot.find_joints(
            [
                r"head_(pitch|yaw)_joint",
            ]
        )

        # Link IDs
        self.torso_link_ids, _ = self._robot.find_bodies(
            [
                r"waist_yaw_link",
            ]
        )

        # Feet
        self.ankle_x_link_ids, _ = self._robot.find_bodies(
            [
                r".*_ankle_roll_link",
            ]
        )

        # Contact link IDs

        # Critical upper-body contact
        self.critical_contact_link_ids, _ = (
            self.contact_sensors.find_bodies(
                [
                    r"waist_yaw_link",
                ]
            )
        )

        # Foot contact
        self.ankle_contact_roll_link_ids, _ = (
            self.contact_sensors.find_bodies(
                [
                    r".*_ankle_roll_link",
                ]
            )
        )

        # Head contact
        self.head_contact_link_ids, _ = (
            self.contact_sensors.find_bodies(
                [
                    r"head_(pitch|yaw)_link",
                ]
            )
        )

        self.leg_joint_limits = self.joint_pos_limits[:, self.total_leg_joint_ids]
        self.arm_joint_limits = self.joint_pos_limits[:, self.total_arm_joint_ids]

    # Scene
    def _setup_scene(self):
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # sensor
        self.scene.sensors["contact_forces"] = ContactSensor(self.cfg.contact_forces)
        self.contact_sensors = self.scene.sensors["contact_forces"]
        self.contact_sensors.update_period = self.cfg.sim_dt

    # Reset Env
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        self.actions = actions
        if self.cfg.num_agents > 1:
            # Multi Agent
            self.processed_actions = {
                "arm": actions["arm"] * self.cfg.action_scale_factor["arm"][0],
                "leg": actions["leg"] * self.cfg.action_scale_factor["leg"][0]
            }
        else:
            # Single Agent
            self.processed_actions = actions * self.cfg.action_scale_factor

    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the scaled actions to the simulator.
        It is called at each physics time-step.
        """
        if self.cfg.num_agents > 1:
            # Multi Agent
            arm_actions = self.processed_actions["arm"]
            leg_actions = self.processed_actions["leg"]

            self._robot.set_joint_position_target(
                target=self._robot.data.default_joint_pos[:, self.total_arm_joint_ids] + arm_actions,
                joint_ids=self.total_arm_joint_ids
            )

            self._robot.set_joint_position_target(
                target=self._robot.data.default_joint_pos[:, self.total_leg_joint_ids] + leg_actions,
                joint_ids=self.total_leg_joint_ids
            )
        else:
            # Single Agent
            self._robot.set_joint_position_target(
                target=self._robot.data.default_joint_pos[:, self._joint_dof_ids] + self.processed_actions,
                joint_ids=self._joint_dof_ids
            )

    # =========================================================================
    # RL abstract methods
    # =========================================================================
    @abstractmethod
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.
        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_states(self) -> torch.Tensor | None:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.
        """
        if self.state_space is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}: state_space is set ({self.state_space}), "
                "so '_get_states' must be implemented to return privileged critic states.")
        else:
            return None  # noqa: R501

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self):
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")
    
    @abstractmethod
    def _compute_intermediate_values(self):
        """Compute planning states for convenient observation setting and reward calculating."""

        raise NotImplementedError(f"Please implement the '_compute_intermediate_values' method for {self.__class__.__name__}.")