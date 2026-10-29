import gymnasium as gym

gym.register(
    id="R1-safe", 
    entry_point=f"{__name__}.R1_safe_env:R1SafeEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_safe_env_cfg:R1SafeEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "predictor_cfg_entry_point": f"{__name__}.cfg:predictor_cfg.yaml",
    }
)

gym.register(
    id="R1-safe-play", 
    entry_point=f"{__name__}.R1_safe_env:R1SafeEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_safe_env_cfg:R1SafePlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "predictor_cfg_entry_point": f"{__name__}.cfg:predictor_cfg.yaml",
    }
)