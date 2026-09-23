
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.utils.plot_utils import CapturabilityPlotter, PNGSavePlotter

from ..R1_base_env_cfg import R1BaseEnvCfg
from .mdp.randomizer import push_and_log

# Environment for training Reach-avoid network
@configclass
class R1FallEnvCfg(R1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 12.0
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

    # ===== Gait guidance ===== #
    time_period = 0.35

    # === safety value config === #
    safety_state_space = 61

    termination_height  = 0.35
    termination_ang_vel = 20.0
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
        resampling_time_range=(100.0, 100.0),
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

        self.events.push_robot = EventTerm(
            func=push_and_log,
            mode="interval",
            interval_range_s=(3.0, 3.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "velocity_range": {
                    "x": (-3.0, 3.0),
                    "y": (-3.0, 3.0),
                    "roll": (-2.0, 2.0),
                    "pitch": (-2.0, 2.0),
                }
            }
        )

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
        self.viz_data = {"real_risk_value": 0,
                         "pred_risk_value": 0,
                         "prediction_error": 0}