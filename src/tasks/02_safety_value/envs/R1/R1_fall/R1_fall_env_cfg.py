
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.utils.plot_utils import CapturabilityPlotter, PNGSavePlotter

from ..R1_base_env_cfg import R1BaseEnvCfg

# Environment for training Reach-avoid network
@configclass
class R1FallEnvCfg(R1BaseEnvCfg):
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
    # observation_space = 95
    # num_agents = 1
    # action_scale_factor = 0.5

    # ===== Gait guidance ===== #
    time_period = 0.35

    # === safety value config === #
    safety_state_space = 67

    # === RA Setting === #
    height_max = 0.35
    phi_max = 3.14/4

    ## ============== Self collision =============== ##
    allowed_collision_bodies = [
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ]

    # commander
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
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
        self.episode_length_s = 30.0

        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.5, 1.5),
            "y": (-1.5, 1.5),
            "roll": (-3.0, 3.0),
            "pitch": (-3.0, 3.0),
        }
        self.events.push_robot.interval_range_s = (2.0, 3.0)

# Environment for eveluating Reach-avoid network
@configclass
class R1FallPlayEnvCfg(R1FallEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # curriculum
        self.curriculum = None

        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        # ==== Viz data ==== #
        self.plotter: PNGSavePlotter = PNGSavePlotter
        self.viz_data = {"risk_value": 0}