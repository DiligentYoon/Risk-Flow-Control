import gymnasium as gym

gym.register(
    id="R1-intervention", 
    entry_point=f"{__name__}.R1_intervention_env:R1InterventionEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_intervention_env_cfg:R1InterventionEnvCfg",
        "rl_single_cfg_entry_point": f"{__name__}.cfg:single_cfg.yaml",
        "rl_multi_cfg_entry_point": f"{__name__}.cfg:multi_cfg.yaml",
        "predictor_cfg_entry_point": f"{__name__}.cfg:predictor_cfg.yaml",
    }
)

gym.register(
    id="R1-intervention-play", 
    entry_point=f"{__name__}.R1_intervention_env:R1InterventionEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_intervention_env_cfg:R1InterventionPlayEnvCfg",
        "rl_single_cfg_entry_point": f"{__name__}.cfg:single_cfg.yaml",
        "rl_multi_cfg_entry_point": f"{__name__}.cfg:multi_cfg.yaml",
        "predictor_cfg_entry_point": f"{__name__}.cfg:predictor_cfg.yaml",
    }
)