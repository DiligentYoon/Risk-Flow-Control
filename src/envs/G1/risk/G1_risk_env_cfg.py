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

from envs.G1.base.G1_base_env_cfg import G1BaseEnvCfg


@configclass
class G1RiskEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    # The episode length IS the recovery deadline H: the terminal constraint
    # V_N(s) + D_H <= delta_N is evaluated exactly when the episode ends.
    # step_dt = sim.dt (0.005) * decimation (4) = 0.02 s, so 2.0 s = 100 steps.
    episode_length_s = 2.0
    decimation = 4

    # CoM height below which a contact counts as a fall. Matches the value used
    # when the frozen safety value function was trained.
    termination_height = 0.2

    ## ========== Agent Setting =========== ##
    action_space = 29                         # joint position offsets, native find_joints(".*") order
    observation_space = 96                    # 3 + 3 + 3 + 29 (q) + 29 (q_dot) + 29 (prev_action)
    num_agents = 1
    action_scale_factor = 0.5

    # No separate critic channel: the privileged input of the critic is the constraint state
    # below, so declaring a state space would only carry the same tensor twice.
    state_space = 0

    ## ========= Pre-trained Network Setting ========= ##
    # Input of the frozen safety value function V_N.
    # 3 (lin vel) + 3 (ang vel) + 3 (gravity) + 29 (q - q_default) + 29 (q_dot)
    constraint_state_space = 67

    ## ============== Collision =============== ##
    allowed_collision_bodies = ["left_ankle_pitch_link",
                                "left_ankle_roll_link",
                                "right_ankle_pitch_link",
                                "right_ankle_roll_link"]

    # Risk-bucket reset
    # Sampling weights over {low, mid, high} buckets produced by collect_init_data.py.
    # The safe bucket is excluded outright -- there is nothing to intervene on there -- while
    # the two unsafe buckets are both used, since weighting them differently is meaningless.
    bucket_weights: dict[str, float] = {"low": 0.0, "mid": 1.0, "high": 1.0}

    def __post_init__(self):
        super().__post_init__()

        # Intervention episodes start from a sampled risk state; external pushes would make the
        # initial distribution something other than the dataset.
        self.events.push_robot = None

        self.events.reset_state_from_dataset = EventTerm(
            func=reset_state_from_dataset,
            mode="reset",
            params={
                "dataset_dir": "logs/dataset/risk_buffer",
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
