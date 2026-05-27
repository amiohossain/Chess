"""Trap dataset: tactical positions with theme labels and positional improvement scores.

Each trap position is labeled with:
  - Theme: fork, pin, skewer, discovered_attack, sacrifice, double_check, mate_threat, other
  - Positional improvement: how much the trap move improves the eval (0.0 to 1.0)
  - Priority: computed from improvement score * theme_weight
"""
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

THEME_WEIGHTS = {
    "mate_threat": 1.0,
    "sacrifice": 0.9,
    "double_check": 0.85,
    "fork": 0.7,
    "pin": 0.6,
    "skewer": 0.6,
    "discovered_attack": 0.5,
    "other": 0.3,
}

THEMES = list(THEME_WEIGHTS.keys())


def compute_trap_priority(improvement_score: float, theme: str) -> float:
    return improvement_score * THEME_WEIGHTS.get(theme, 0.3)


def process_lichess_puzzles(
    puzzle_json_path: str,
    output_h5_path: str,
    max_puzzles: int = 500_000,
) -> int:
    """Convert Lichess puzzle JSONL to HDF5 trap dataset."""
    import chess
    from src.model.feature_encoder import encode_board, encode_move

    X = np.zeros((max_puzzles, 119, 8, 8), dtype=np.float32)
    y_policy = np.zeros(max_puzzles, dtype=np.int32)
    y_value = np.ones(max_puzzles, dtype=np.float32)
    themes = np.zeros(max_puzzles, dtype=np.int32)
    improvements = np.zeros(max_puzzles, dtype=np.float32)
    priorities = np.zeros(max_puzzles, dtype=np.float32)

    count = 0
    with open(puzzle_json_path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_puzzles:
                break
            try:
                puzzle = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            rating = puzzle.get("Rating", 0)
            if rating < 1800:
                continue

            fen = puzzle.get("FEN")
            moves_str = puzzle.get("Moves", "")
            puzzle_themes = puzzle.get("Themes", "")

            if not fen or not moves_str:
                continue

            board = chess.Board(fen)
            first_move_uci = moves_str.split()[0]
            try:
                move = chess.Move.from_uci(first_move_uci)
            except ValueError:
                continue

            if move not in board.legal_moves:
                continue

            X[count] = encode_board(board)
            y_policy[count] = encode_move(move, board)

            theme_label = _classify_themes(puzzle_themes)
            themes[count] = THEMES.index(theme_label)
            improvements[count] = min(rating / 3000.0, 1.0)
            priorities[count] = compute_trap_priority(improvements[count], theme_label)

            count += 1

    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("X", data=X[:count], compression="lzf", chunks=True)
        f.create_dataset("y_policy", data=y_policy[:count], compression="lzf", chunks=True)
        f.create_dataset("y_value", data=y_value[:count], compression="lzf", chunks=True)
        f.create_dataset("theme", data=themes[:count], compression="lzf")
        f.create_dataset("improvement", data=improvements[:count], compression="lzf")
        f.create_dataset("priority", data=priorities[:count], compression="lzf")
        f.attrs["num_positions"] = count

    return count


def _classify_themes(themes_str: str) -> str:
    themes_lower = themes_str.lower()
    theme_map = [
        ("mate", "mate_threat"),
        ("sacrifice", "sacrifice"),
        ("doubleCheck", "double_check"),
        ("fork", "fork"),
        ("pin", "pin"),
        ("skewer", "skewer"),
        ("discoveredAttack", "discovered_attack"),
    ]
    for keyword, theme in theme_map:
        if keyword in themes_lower:
            return theme
    return "other"


class TrapDataset(Dataset):
    """PyTorch Dataset for trap positions with priority-weighted sampling."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        logger.info(f"Opening trap dataset: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            self.num_positions = f.attrs["num_positions"]
            self.priorities = f["priority"][:]
        self.sampling_weights = self.priorities / self.priorities.sum() if self.priorities.sum() > 0 else None

    def __len__(self):
        return self.num_positions

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            X = torch.from_numpy(f["X"][idx].astype(np.float32))
            y_policy = torch.tensor(f["y_policy"][idx], dtype=torch.long)
            y_value = torch.tensor(f["y_value"][idx], dtype=torch.float32)
            theme = torch.tensor(f["theme"][idx], dtype=torch.long)
            improvement = torch.tensor(f["improvement"][idx], dtype=torch.float32)
            priority = torch.tensor(f["priority"][idx], dtype=torch.float32)

        legal_mask = torch.ones(4096, dtype=torch.float32)
        return {
            "X": X, "y_policy": y_policy, "y_value": y_value,
            "legal_mask": legal_mask, "theme": theme,
            "improvement": improvement, "priority": priority,
        }
