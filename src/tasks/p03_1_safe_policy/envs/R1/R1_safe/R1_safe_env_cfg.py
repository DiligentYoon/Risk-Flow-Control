
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from lib.utils.plot_utils import PNGSavePlotter

from tasks.p02_safety_value.envs.R1.R1_base_env_cfg import R1BaseEnvCfg
from tasks.p03_1_safe_policy.envs.mdp.randomizer import reset_state_from_dataset

# Environment for training Reach-avoid network
@configclass
class R1SafeEnvCfg(R1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 4.0
    sim_dt = 0.005
    decimation = 4

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 12, "leg": 12}
    observation_space = {"arm": 45, "leg": 45}
    state_space = {"arm": 82, "leg": 82}
    num_agents = 2
    action_scale_factor = {"arm": [0.5, ()],
                           "leg": [0.5, ()]}

    ## ========== Single Agent Setting ========== ##
    # action_space = 24
    # observation_space = 81
    # num_agents = 1
    # action_scale_factor = 0.5

    # ===== Gait guidance ===== #
    time_period = 0.45

    # === safety value config === #
    safety_state_space = 57

    success_value = -0.1
    termination_height  = 0.3
    phi_max = 3.14/4
    
    # ==== Reward ===== #
    r_success: float = 200
    r_delta_safety: float = 5.0
    
    p_pred_error: float = 5.0
    p_lin_vel: float = 0.1
    p_ang_vel_xy: float = 0.2
    
    p_deviation_swing: float = 0.0
    p_deviation_hip: float = 1.0
    p_deviation_arm: float = 0.5
    p_action_rate: float = 5.0e-2
    p_limits: float = 10.0
    
    p_joint_torque: float = 1.0e-6
    p_joint_torque_limit: float = 1.0e-4
    p_joint_vel: float = 1.0e-5
    p_joint_acc: float = 1.0e-7
    p_fail: float = 200


    ## ============== Self collision =============== ##
    allowed_collision_bodies = [
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_state_from_dataset,
            mode="reset",
            params={
                "dataset_path": "",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

# Environment for eveluating Reach-avoid network
@configclass
class R1SafePlayEnvCfg(R1SafeEnvCfg):
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