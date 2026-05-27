"""Phase 3: Continuous self-play learning.

Each session:
  1. Play 500 games vs the latest checkpoint (using MCTS + temperature 1.0)
  2. Extract positions from wins and close losses
  3. Mix 80% replay buffer + 20% fresh self-play data
  4. Train for ~2 hours
  5. Gate: new checkpoint must beat old checkpoint >55% in 200 game match
"""
import os
import time
import logging
import torch
import torch.optim as optim
import numpy as np
import chess
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.config import ChessConfig
from src.model.chess_net import ChessNet
from src.model.feature_encoder import encode_board, encode_move
from src.model.losses import combined_loss
from src.search.mcts import MCTS
from src.inference.move_selector import select_move
from src.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

logger = logging.getLogger(__name__)


class ReplayBuffer(Dataset):
    """Rolling replay buffer for self-play positions."""

    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.positions = []
        self.size = 0

    def add(self, positions: list):
        self.positions.extend(positions)
        if len(self.positions) > self.max_size:
            self.positions = self.positions[-self.max_size:]
        self.size = len(self.positions)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pos = self.positions[idx]
        return {
            "X": torch.from_numpy(pos["X"]),
            "y_policy": torch.tensor(pos["y_policy"], dtype=torch.long),
            "y_value": torch.tensor(pos["y_value"], dtype=torch.float32),
            "legal_mask": torch.ones(4096, dtype=torch.float32),
        }


def play_self_play_game(model: ChessNet, config: ChessConfig, device: torch.device) -> list:
    """Play a single self-play game. Returns list of (X, y_policy, y_value)."""
    board = chess.Board()
    positions = []
    mcts = MCTS(model, config.mcts, device)
    temp = config.mcts.temperature_self_play

    moves_played = 0
    while not board.is_game_over() and moves_played < 200:
        board_planes = encode_board(board)
        move, _, _ = select_move(board, model, config.mcts, temperature=temp, trap_db=None)
        positions.append({
            "X": board_planes,
            "y_policy": encode_move(move, board),
            "y_value": 0.0,
        })
        board.push(move)
        moves_played += 1
        if moves_played > 30:
            temp = 0.5

    if board.is_checkmate():
        outcome = 1.0 if (moves_played % 2 == 1) else -1.0
    elif board.is_game_over():
        outcome = 0.0
    else:
        outcome = 0.0

    for pos in positions:
        pos["y_value"] = outcome

    return positions


def run_self_play_session(config: ChessConfig):
    """Run one self-play session: play games -> train -> gate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessNet(config.model).to(device)
    latest_path = find_latest_checkpoint(config.paths.checkpoint_dir)
    if latest_path:
        load_checkpoint(latest_path, model, device=device)
        logger.info(f"Loaded model from {latest_path}")
    else:
        raise FileNotFoundError("No checkpoint found -- train Phase 1 first!")

    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate * 0.05)
    scaler = GradScaler(enabled=(config.training.mixed_precision == "fp16"))

    replay_buffer = ReplayBuffer(config.self_play.replay_buffer_size)

    logger.info(f"Playing {config.self_play.games_per_session} self-play games...")
    for game_idx in range(config.self_play.games_per_session):
        positions = play_self_play_game(model, config, device)
        replay_buffer.add(positions)
        if (game_idx + 1) % 50 == 0:
            logger.info(f"  Played {game_idx + 1}/{config.self_play.games_per_session} games ({replay_buffer.size} positions)")

    logger.info(f"Replay buffer: {replay_buffer.size} positions. Training...")
    model.train()

    dataloader = DataLoader(
        replay_buffer, batch_size=config.training.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    for batch in dataloader:
        X = batch["X"].to(device, non_blocking=True)
        y_policy = batch["y_policy"].to(device, non_blocking=True)
        y_value = batch["y_value"].to(device, non_blocking=True)
        legal_masks = batch["legal_mask"].to(device, non_blocking=True)

        with autocast(enabled=(config.training.mixed_precision == "fp16")):
            policy_logits, value_pred = model(X)
            loss = combined_loss(policy_logits, y_policy, value_pred, y_value, legal_masks, config.training)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    save_checkpoint(model, optimizer, step=0, epoch=0, loss=0.0, tag=f"self_play_{int(time.time())}")
    logger.info("Self-play session complete. Checkpoint saved.")
