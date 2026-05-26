"""All hyperparameters, paths, and model dimensions in one place."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Neural network architecture dimensions."""
    input_channels: int = 119
    board_size: int = 8
    filters: int = 384
    num_blocks: int = 10
    policy_channels: int = 32
    policy_output_size: int = 4096
    value_hidden: int = 256
    dropout: float = 0.1
    activation: str = "swiglu"

    @property
    def total_planes(self) -> Tuple[int, int, int]:
        return (self.input_channels, self.board_size, self.board_size)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-6
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    batch_size: int = 64
    num_workers: int = 2
    max_position_per_epoch: int = 10_000_000
    policy_weight: float = 0.5
    value_weight: float = 0.3
    top10_reg_weight: float = 0.2
    gradient_clip_norm: float = 1.0
    checkpoint_every_n_steps: int = 10_000
    mixed_precision: str = "fp16"
    cosine_decay_steps: int = 100_000
    trap_loss_weight: float = 2.0


@dataclass
class TrapConfig:
    """Trap specialization hyperparameters."""
    trap_data_ratio: float = 0.5
    trap_oversample: int = 3
    trap_loss_weight: float = 2.0
    trap_positional_threshold: float = 0.1
    trap_boost_factor: float = 1.2
    trap_guard_threshold: float = -0.3


@dataclass
class MCTSConfig:
    """MCTS search parameters."""
    num_simulations: int = 400
    c_puct: float = 1.4
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature_self_play: float = 1.0
    temperature_eval: float = 0.15
    temperature_analysis: float = 0.0
    top_k_moves: int = 20


@dataclass
class SelfPlayConfig:
    """Self-play continuous learning."""
    games_per_session: int = 500
    improvement_test_games: int = 200
    improvement_threshold: float = 0.55
    replay_buffer_size: int = 200_000
    replay_fresh_ratio: float = 0.2
    train_hours_per_session: float = 2.0


@dataclass
class PathConfig:
    """All file paths. Override these for Kaggle environment."""
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./data"
    trap_data_path: str = "./data/trap_positions.h5"
    supervised_data_path: str = "./data/supervised_positions.h5"
    replay_buffer_path: str = "./data/replay_buffer.h5"
    log_dir: str = "./logs"


@dataclass
class ChessConfig:
    """Aggregate config for easy passing."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trap: TrapConfig = field(default_factory=TrapConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    paths: PathConfig = field(default_factory=PathConfig)
