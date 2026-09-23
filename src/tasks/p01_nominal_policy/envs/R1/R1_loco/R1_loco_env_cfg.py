from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.utils.plot_utils import PNGSavePlotter

from ..R1_base_env_cfg import R1BaseEnvCfg

@configclass
class R1LocoEnvCfg(R1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005
    decimation = 4

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 14, "leg": 12}
    observation_space = {"arm": 56, "leg": 50}
    state_space = {"arm": 93, "leg": 93}
    num_agents = 2
    action_scale_factor = {"arm": [0.5, ()],
                           "leg": [0.5, ()]}

    ## ========== Single Agent Setting ========== ##
    # action_space = 26
    # observation_space = 92
    # num_agents = 1
    # action_scale_factor = 0.5

    ## ==================== Reward Shaping ==================== ##
    r_track_lin_vel: float = 8.0
    r_track_ang_vel: float = 8.0
    r_track_heading: float = 0.0
    r_track_height: float = 2.0
    r_feet_gait: float = 10.0
    r_flat: float = 2.0
    
    p_support_xy: float = 1.0
    p_lin_vel_z: float = 2.0
    p_ang_vel_xy: float = 0.2
    p_joint_torque: float = 1.0e-7
    p_joint_torque_limit: float = 1.0e-5
    p_joint_vel: float = 1.0e-4

    p_limits: float = 10.0
    p_deviation_swing: float = 2.0
    p_deviation_hip: float = 2.0
    p_deviation_arm: float = 2.0
    p_action_rate: float = 1.0e-3

    p_termination: float = 200
    termination_height: float = 0.3
    termination_gravity: float = 0.8
    termination_ang_vel: float = 20.0

    target_height = 0.73

    # ===== Gait guidance ===== #
    time_period = 0.35

    ## ============== Self collision =============== ##
    allowed_collision_bodies = [
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ]

    # Commander
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )

    # visualization
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_ang_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angular_velocity_goal"
    )

    current_ang_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angular_velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
    goal_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
    current_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)

    def __post_init__(self):
        super().__post_init__()

        self.robot.spawn.articulation_props.enabled_self_collisions = True

        self.events.push_robot.interval_range_s = (4.0, 5.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-1.0, 1.0),
            "pitch": (-1.0, 1.0),
        }


@configclass
class R1LocoPlayEnvCfg(R1LocoEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0),
        )

        self.scene.num_envs = 1

        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-1.0, 1.0),
            "pitch": (-1.0, 1.0),
        }