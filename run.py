"""Local development entry point.

Usage:
  python run.py --test-model    # Quick model smoke test
  python run.py --interactive   # Play against the model
  python run.py --run-tests     # Run all tests
"""
import argparse
import chess
import torch
import numpy as np
import subprocess
import sys

from src.config import ChessConfig
from src.model.chess_net import ChessNet
from src.model.feature_encoder import encode_board, legal_move_mask, decode_move
from src.inference.move_selector import select_move
from src.utils.checkpoint import load_checkpoint, find_latest_checkpoint


def test_model():
    """Quick smoke test: load model and check forward pass."""
    config = ChessConfig()
    model = ChessNet(config.model)
    model.eval()

    x = torch.randn(1, 119, 8, 8)
    with torch.no_grad():
        policy, value = model(x)

    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    board = chess.Board()
    planes = encode_board(board)
    x2 = torch.from_numpy(planes).unsqueeze(0)
    mask = torch.from_numpy(legal_move_mask(board)).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value = model(x2)
        masked = policy_logits + (1.0 - mask) * -1e9
        probs = torch.softmax(masked, dim=-1)
        best_idx = probs.argmax().item()

    best_move = decode_move(best_idx, board)
    legal_moves = list(board.legal_moves)
    print(f"\nStarting position evaluation: {value.item():.3f}")
    print(f"Best move (model): {best_move}")
    print(f"Is legal: {best_move in legal_moves}")
    print(f"Legal moves count: {len(legal_moves)}")


def play_interactive(config: ChessConfig):
    """Play a game against the model in the terminal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(config.model).to(device)

    latest_path = find_latest_checkpoint(config.paths.checkpoint_dir)
    if latest_path:
        load_checkpoint(latest_path, model, device=device)
        print(f"Loaded model from {latest_path}")
    else:
        print("No checkpoint found -- using untrained model")

    model.eval()
    board = chess.Board()

    print("\n=== Chess Model Interactive ===")
    print("Enter moves in UCI format (e.g., e2e4) or 'quit'")

    while not board.is_game_over():
        print(f"\n{board}")
        print(f"FEN: {board.fen()}")

        if board.turn == chess.WHITE:
            move, pv, meta = select_move(board, model, config.mcts, config.trap)
            print(f"Model plays: {move} (value={meta['root_value']:.3f})")
            board.push(move)
        else:
            uci = input("Your move: ").strip()
            if uci.lower() == "quit":
                break
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move!")
            except ValueError:
                print("Invalid format! Use UCI (e.g., e2e4)")

    print(f"\nFinal: {board.result()}")
    print(board)


def run_tests():
    """Run the full test suite."""
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Chess Model Training & Inference")
    parser.add_argument("--test-model", action="store_true", help="Run model smoke test")
    parser.add_argument("--interactive", action="store_true", help="Play against model")
    parser.add_argument("--run-tests", action="store_true", help="Run all tests")
    args = parser.parse_args()

    config = ChessConfig()

    if args.test_model:
        test_model()
    elif args.interactive:
        play_interactive(config)
    elif args.run_tests:
        sys.exit(run_tests())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
