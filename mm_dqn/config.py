from dataclasses import dataclass


@dataclass
class EnvConfig:
    n_levels: int = 10
    window_size: int = 50
    episode_events: int = 800
    min_trade_unit: int = 100
    max_inventory_units: int = 10
    max_bias: float = 0.05
    max_spread: float = 0.10
    eta: float = 0.5
    zeta: float = 0.01


@dataclass
class ModelConfig:
    dynamic_dim: int = 24
    agent_dim: int = 2
    hidden_dim: int = 64
    n_heads: int = 4
    action_bins_a1: int = 11
    action_bins_a2: int = 11
    use_dueling: bool = True  # NEW: Enable Dueling DQN
    use_residual: bool = True  # NEW: Enable residual connections
    use_layer_norm: bool = True  # NEW: Enable LayerNorm

    @property
    def action_dim(self) -> int:
        return self.action_bins_a1 * self.action_bins_a2


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    buffer_size: int = 50_000
    warmup_steps: int = 200
    target_update_steps: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    # NEW: Double DQN and other improvements
    use_double_dqn: bool = True
    use_prioritized_replay: bool = False  # Optional: can be enabled later
