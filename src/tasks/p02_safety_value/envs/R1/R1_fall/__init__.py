import gymnasium as gym

gym.register(
    id="R1-fall", 
    entry_point=f"{__name__}.R1_fall_env:R1FallEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_fall_env_cfg:R1FallEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "predictor_cfg_entry_point": f"{__name__}.cfg:predictor_cfg.yaml",
    }
)

gym.register(
    id="R1-fall-play", 
    entry_point=f"{__name__}.R1_fall_env:R1FallEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_fall_env_cfg:R1FallPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "predictor_cfg_entry_point": f"{__name__}.cfg:predictor_cfg.yaml",
    }
)