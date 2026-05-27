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
# GradScaler/autocast from torch.amp to avoid deprecation warnings

from src.config import ChessConfig
from src.model.chess_net import ChessNet
from src.model.feature_encoder import encode_board, encode_move
from src.model.losses import combined_loss
from src.search.mcts import MCTS
from src.inference.move_selector import select_move
from src.utils.checkpoint import save_checkpoint, save_latest_weights, load_checkpoint, find_latest_checkpoint

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


def play_self_play_game(model: ChessNet, config: ChessConfig, device: torch.device, game_num: int = 0) -> list:
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
        # White played last if moves_played is odd, black if even
        winner = "W" if (moves_played % 2 == 1) else "B"
        outcome = 1.0 if (moves_played % 2 == 1) else -1.0
        result_str = f"1-0 ({winner} by checkmate)"
    elif board.is_game_over():
        outcome = 0.0
        result_str = "½-½ (draw)"
    else:
        outcome = 0.0
        result_str = "½-½ (truncated)"

    for pos in positions:
        pos["y_value"] = outcome

    return positions, result_str, moves_played


def run_self_play_session(config: ChessConfig):
    """Run one self-play session: play games -> train -> gate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (config.training.mixed_precision == "fp16") and torch.cuda.is_available()

    model = ChessNet(config.model).to(device)
    latest_path = find_latest_checkpoint(config.paths.checkpoint_dir)
    if latest_path:
        load_checkpoint(latest_path, model, device=device)
        logger.info(f"Loaded model from {latest_path}")
    else:
        raise FileNotFoundError("No checkpoint found -- train Phase 1 first!")

    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate * 0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    replay_buffer = ReplayBuffer(config.self_play.replay_buffer_size)

    n_games = config.self_play.games_per_session
    session_start = time.time()
    game_results = {"W": 0, "B": 0, "draw": 0, "trunc": 0}
    total_moves = 0
    total_positions = 0

    logger.info(f"{'='*60}")
    logger.info(f"SELF-PLAY: {n_games} games")
    logger.info(f"{'='*60}")

    for game_idx in range(n_games):
        positions, result_str, moves = play_self_play_game(model, config, device, game_num=game_idx + 1)
        replay_buffer.add(positions)
        total_moves += moves
        total_positions += len(positions)

        if "1-0" in result_str:
            game_results["W"] += 1
        elif "0-1" in result_str:
            game_results["B"] += 1
        elif "draw" in result_str:
            game_results["draw"] += 1
        else:
            game_results["trunc"] += 1

        logger.info(
            f"[{game_idx+1:>4}/{n_games}] {result_str} | "
            f"{moves:>3} ply | "
            f"{len(positions):>4} positions | "
            f"W:{game_results['W']} B:{game_results['B']} "
            f"D:{game_results['draw']} T:{game_results['trunc']}"
        )

    elapsed = time.time() - session_start
    total = game_results["W"] + game_results["B"] + game_results["draw"] + game_results["trunc"]
    logger.info(f"{'='*60}")
    logger.info(f"SELF-PLAY COMPLETE: {total} games in {elapsed/60:.1f}min")
    logger.info(f"  Results: W={game_results['W']} B={game_results['B']} "
                f"Draw={game_results['draw']} Trunc={game_results['trunc']}")
    logger.info(f"  Avg ply/game: {total_moves/max(total,1):.1f}")
    logger.info(f"  Avg positions/game: {total_positions/max(total,1):.0f}")
    logger.info(f"  Total positions in replay buffer: {replay_buffer.size}")
    logger.info(f"{'='*60}")

    logger.info(f"Training on replay buffer ({replay_buffer.size} positions)...")
    model.train()

    dataloader = DataLoader(
        replay_buffer, batch_size=config.training.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    n_batches = len(dataloader)
    train_loss = 0.0
    train_start = time.time()
    self_play_step = 0

    # Resume step count from checkpoint if available
    if latest_path:
        state = load_checkpoint(latest_path, model, optimizer, device=device)
        self_play_step = state.get("step", 0)

    for batch_idx, batch in enumerate(dataloader):
        X = batch["X"].to(device, non_blocking=True)
        y_policy = batch["y_policy"].to(device, non_blocking=True)
        y_value = batch["y_value"].to(device, non_blocking=True)
        legal_masks = batch["legal_mask"].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            policy_logits, value_pred = model(X)
            loss = combined_loss(policy_logits, y_policy, value_pred, y_value, legal_masks, config.training)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        self_play_step += 1

        # Save weights every step — crash-safe resume
        save_latest_weights(model, self_play_step, loss.item(), config.paths.checkpoint_dir)

        # Per-step log
        logger.info(
            f"sp train step {self_play_step:>6} | "
            f"batch {batch_idx+1}/{n_batches} | "
            f"loss={loss.item():.4f}"
        )

        # Detailed stats every 500 steps
        if (batch_idx + 1) % 500 == 0:
            pct = 100.0 * (batch_idx + 1) / n_batches
            batch_elapsed = time.time() - train_start
            logger.info(
                f"--- SP TRAIN PROGRESS: step {self_play_step:,} | "
                f"{pct:.0f}% ({batch_idx+1:,}/{n_batches:,}) | "
                f"avg_loss={train_loss/max(batch_idx+1,1):.4f} | "
                f"elapsed {batch_elapsed/60:.1f}min ---"
            )

    train_time = time.time() - train_start
    avg_loss = train_loss / max(n_batches, 1)
    logger.info(f"Training complete: {n_batches} batches in {train_time:.1f}s, avg loss={avg_loss:.4f}")

    save_checkpoint(model, optimizer, step=self_play_step, epoch=0, loss=avg_loss, tag=f"self_play_{int(time.time())}")
    logger.info(">>> Self-play checkpoint saved")
