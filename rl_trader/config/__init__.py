"""
RL Trader — Enterprise Configuration Loader.
Loads YAML settings, with environment variable overrides for production.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = CONFIG_DIR / "settings.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Load YAML config with env var overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Environment variable overrides (RLTRADER__SECTION__KEY=value)
    for env_key, env_val in os.environ.items():
        if env_key.startswith("RLTRADER__"):
            parts = env_key.lower().strip("rltrader_").split("__")
            node = config
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            # Try to parse as int/float/bool
            try:
                node[parts[-1]] = yaml.safe_load(env_val)
            except yaml.YAMLError:
                node[parts[-1]] = env_val

    return config


def setup_logging(config: dict) -> logging.Logger:
    """Configure structured logging."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    log_file = log_config.get("file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format=fmt,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=level, format=fmt, handlers=[logging.StreamHandler()])

    return logging.getLogger("rl_trader")


# ─── Config Dataclasses (type-safe access) ───────────────────────────────────
@dataclass
class TradingConfig:
    symbol: str = "AAPL"
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    initial_balance: float = 10000
    window_size: int = 10
    fee: float = 0.001
    max_position: float = 1.0


@dataclass
class AgentConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 200000
    learning_rate: float = 0.0003
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 2048
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ensemble_seeds: list = field(default_factory=lambda: [42, 123, 777])


@dataclass
class RewardConfig:
    type: str = "risk_adjusted"
    drawdown_penalty: float = 0.5
    sharpe_window: int = 20
    volatility_penalty: float = 0.1


@dataclass
class AppConfig:
    trading: TradingConfig
    agent: AgentConfig
    reward: RewardConfig
    raw: dict

    @classmethod
    def from_dict(cls, config: dict):
        return cls(
            trading=TradingConfig(**config.get("trading", {})),
            agent=AgentConfig(
                **{k: v for k, v in config.get("agent", {}).items() if k != "network_arch"}
            ),
            reward=RewardConfig(**config.get("reward", {})),
            raw=config,
        )

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG):
        return cls.from_dict(load_config(config_path))
