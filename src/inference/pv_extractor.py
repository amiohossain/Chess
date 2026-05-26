"""Extract principal variation from the MCTS tree."""
import chess
from src.search.mcts import MCTS


def extract_pv(mcts: MCTS, board: chess.Board, max_depth: int = 8) -> list:
    """Extract PV by following most-visited child at each node."""
    board = board.copy()
    pv = []

    for _ in range(max_depth):
        if board.is_game_over():
            break

        policy, _ = mcts.search(board)
        if not policy:
            break

        best_uci = max(policy, key=policy.get)
        best_move = chess.Move.from_uci(best_uci)
        pv.append(best_move)
        board.push(best_move)

        if board.is_game_over():
            break

    return pv
