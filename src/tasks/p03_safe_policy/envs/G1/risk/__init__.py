import gymnasium as gym

gym.register(
    id="G1-risk",
    entry_point=f"{__name__}.G1_risk_env:G1RiskEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_risk_env_cfg:G1RiskEnvCfg",
        "rl_risk_flow_cfg_entry_point": f"{__name__}.config:risk_flow_cfg.yaml",
    }
)

gym.register(
    id="G1-risk-play",
    entry_point=f"{__name__}.G1_risk_env:G1RiskEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_risk_env_cfg:G1RiskPlayEnvCfg",
        "rl_risk_flow_cfg_entry_point": f"{__name__}.config:risk_flow_cfg.yaml",
    }
)
