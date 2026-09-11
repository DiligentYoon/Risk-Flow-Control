# Copyright (c) 2026, AISL.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from lib.env.G1.safe.mdp.randomizer import reset_state_from_dataset
from lib.utils.plot_utils import PNGSavePlotter

from envs.G1.base.G1_base_env_cfg import G1BaseEnvCfg

@configclass
class G1RiskEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 2.0
    decimation = 4

    # CoM height below which a contact counts as a fall. 
    termination_height = 0.2

    ## ========== Agent Setting =========== ##
    action_space = 29                         # joint position offsets, native find_joints(".*") order
    observation_space = 96                    # 3 + 3 + 3 + 29 (q) + 29 (q_dot) + 29 (prev_action)
    num_agents = 1
    action_scale_factor = 5.0

    ## ========= Pre-trained Network Setting ========= ##
    # Input of the frozen safety value function V_N.
    # 3 (lin vel) + 3 (ang vel) + 3 (gravity) + 29 (q - q_default) + 29 (q_dot)
    constraint_joint_pos_start = 9
    constraint_state_space = 67

    ## ============== Collision =============== ##
    allowed_collision_bodies = ["left_ankle_pitch_link",
                                "left_ankle_roll_link",
                                "right_ankle_pitch_link",
                                "right_ankle_roll_link"]

    # Risk-bucket reset
    # Sampling weights over {low, mid, high} buckets produced by collect_init_data.py.
    bucket_weights: dict[str, float] = {"low": 0.0, "mid": 1.0, "high": 0.0}

    def __post_init__(self):
        super().__post_init__()

        # visualization -- training records nothing per step
        self.viz_data = None
        self.plotter = None

        # Intervention episodes start from a sampled risk state;
        self.events.push_robot = None

        self.events.reset_state_from_dataset = EventTerm(
            func=reset_state_from_dataset,
            mode="reset",
            params={
                "dataset_dir": "logs/frozen/collected/2",
                "bucket_weights": self.bucket_weights,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class G1RiskPlayEnvCfg(G1RiskEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        self.scene.num_envs = 1

        # plotter
        self.plotter = PNGSavePlotter

        # The three `risk_*` channels are the algorithm's own quantities, not the simulator's:
        # `play.py` writes them into `viz_data` before appending a frame, the way
        # `main/reach_avoid/play.py` injects `risk_value`. They are declared here so that the
        # plotter allocates a column for them from the first frame.
        self.viz_data = {
            "risk_value": 0.0,          # V_N(s_t)
            "risk_flow": 0.0,           # D_H(s_t, a_t)
            "terminal_risk": 0.0,       # V_N(s_t) + D_H(s_t, a_t) - delta_N
            "action_magnitude": 0.0,
            "max_torque": 0.0,
            "CoM_height": 0.0,
        }
