import gymnasium as gym

gym.register(
    id="R1-loco", 
    entry_point=f"{__name__}.R1_loco_env:R1LocoEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_loco_env_cfg:R1LocoEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)

gym.register(
    id="R1-loco-play", 
    entry_point=f"{__name__}.R1_loco_env:R1LocoEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.R1_loco_env_cfg:R1LocoPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)