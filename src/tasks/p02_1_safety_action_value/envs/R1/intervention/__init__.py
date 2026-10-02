import gymnasium as gym

gym.register(
    id="R1-intervention",
    entry_point=f"{__name__}.R1_intervention_env:R1InterventionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.R1_intervention_env_cfg:R1InterventionEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "safety_q_cfg_entry_point": f"{__name__}.cfg:safety_q.yaml",
    }
)

gym.register(
    id="R1-intervention-play",
    entry_point=f"{__name__}.R1_intervention_env:R1InterventionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.R1_intervention_env_cfg:R1InterventionPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "safety_q_cfg_entry_point": f"{__name__}.cfg:safety_q.yaml",
    }
)
