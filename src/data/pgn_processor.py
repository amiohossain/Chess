"""Convert PGN files to an HDF5 dataset of encoded positions.

Reads a PGN file, plays through each game, and for each position:
  - Encodes the board as 8x8x119 float32 planes
  - Records the move played as a 4096-class label
  - Records the game outcome as +1 (White wins), -1 (Black wins), 0 (draw)

Output: HDF5 file with datasets:
  - X: (N, 119, 8, 8) float32 — board planes
  - y_policy: (N,) int32 — encoded move index
  - y_value: (N,) float32 — game outcome
"""
import os
import numpy as np
import h5py
import chess
import chess.pgn
from tqdm import tqdm
from src.model.feature_encoder import encode_board, encode_move


def process_pgn(
    pgn_path: str,
    output_path: str,
    max_positions: int = 10_000_000,
    min_elo: int = 0,
    max_games: int = None,
) -> int:
    """Convert a PGN file to an HDF5 dataset.

    Args:
        pgn_path: Path to the PGN file.
        output_path: Path for the output .h5 file.
        max_positions: Maximum number of positions to store.
        min_elo: Minimum average Elo to include a game.
        max_games: Maximum number of games to process.

    Returns:
        Number of positions written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    X = np.zeros((max_positions, 119, 8, 8), dtype=np.float32)
    y_policy = np.zeros(max_positions, dtype=np.int32)
    y_value = np.zeros(max_positions, dtype=np.float32)

    count = 0
    game_count = 0

    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if max_games and game_count >= max_games:
                break

            # Elo filter
            if min_elo > 0:
                white_elo = game.headers.get("WhiteElo", "0")
                black_elo = game.headers.get("BlackElo", "0")
                try:
                    avg_elo = (int(white_elo) + int(black_elo)) / 2
                except (ValueError, TypeError):
                    avg_elo = 0
                if avg_elo < min_elo:
                    game_count += 1
                    continue

            # Determine game outcome
            result = game.headers.get("Result", "*")
            outcome_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
            outcome = outcome_map.get(result, 0.0)

            # Play through the game
            board = game.board()
            for move in game.mainline_moves():
                if count >= max_positions:
                    break

                X[count] = encode_board(board)
                y_policy[count] = encode_move(move, board)
                y_value[count] = outcome

                board.push(move)
                count += 1

            game_count += 1
            if count >= max_positions:
                break

    # Trim and write HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("X", data=X[:count], compression="lzf", chunks=True)
        f.create_dataset("y_policy", data=y_policy[:count], compression="lzf", chunks=True)
        f.create_dataset("y_value", data=y_value[:count], compression="lzf", chunks=True)
        f.attrs["num_positions"] = count

    return count


def process_multiple_pgns(
    pgn_paths: list,
    output_path: str,
    max_positions: int = 10_000_000,
    min_elo: int = 0,
) -> int:
    """Process multiple PGN files into a single HDF5 dataset."""
    total = 0
    all_X = []
    all_policy = []
    all_value = []

    for pgn_path in pgn_paths:
        temp_path = output_path + f".temp_{len(all_X)}"
        count = process_pgn(pgn_path, temp_path, max_positions // len(pgn_paths), min_elo)
        if count > 0:
            with h5py.File(temp_path, "r") as f:
                all_X.append(f["X"][:])
                all_policy.append(f["y_policy"][:])
                all_value.append(f["y_value"][:])
            os.remove(temp_path)
        total += count

    with h5py.File(output_path, "w") as f:
        f.create_dataset("X", data=np.concatenate(all_X, axis=0), compression="lzf", chunks=True)
        f.create_dataset("y_policy", data=np.concatenate(all_policy, axis=0), compression="lzf", chunks=True)
        f.create_dataset("y_value", data=np.concatenate(all_value, axis=0), compression="lzf", chunks=True)
        f.attrs["num_positions"] = total

    return total
