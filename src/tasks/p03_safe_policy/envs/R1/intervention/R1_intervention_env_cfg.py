
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.utils.plot_utils import PNGSavePlotter

from ..R1_base_env_cfg import R1BaseEnvCfg

# Environment for training Reach-avoid network
@configclass
class R1InterventionEnvCfg(R1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 8.0
    sim_dt = 0.005
    decimation = 4

    ## ========== Multi Agent Setting =========== ##
    # possible_agents = ["arm", "leg"]
    # action_space = {"arm": 14, "leg": 12}
    # observation_space = {"arm": 56, "leg": 50}
    # state_space = {"arm": 93, "leg": 93}
    # num_agents = 2
    # action_scale_factor = {"arm": [0.5, ()],
    #                        "leg": [0.5, ()]}

    ## ========== Single Agent Setting ========== ##
    action_space = 26
    observation_space = 92
    num_agents = 1
    action_scale_factor = 0.5

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

    def __post_init__(self):
        super().__post_init__()

# Environment for eveluating Reach-avoid network
@configclass
class R1InterventionPlayEnvCfg(R1InterventionEnvCfg):
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