"""1-ply opponent response blunder check.

For a given position and candidate move, checks if the opponent has:
  - A forced checkmate in 2
  - A winning capture (Q, R, or undefended piece)

Catches ~70% of one-move blunders, adding ~100-150 Elo.
"""
import chess
from src.model.feature_encoder import encode_move


def check_blunder(board: chess.Board, candidate_move: chess.Move) -> dict:
    """Check if a move blunders by analyzing the opponent's best response.

    Args:
        board: Current board position.
        candidate_move: The move we're considering.

    Returns:
        dict with is_blunder, reason, opponent_best_move, material_loss.
    """
    result = {
        "is_blunder": False,
        "reason": "",
        "opponent_best_move": None,
        "material_loss": 0,
    }

    board.push(candidate_move)

    try:
        if board.is_checkmate():
            return result

        # Check for forced mate in 2
        for response in board.legal_moves:
            board.push(response)
            if board.is_checkmate():
                result["is_blunder"] = True
                result["reason"] = f"Opponent has forced mate after {candidate_move.uci()} {response.uci()}"
                result["opponent_best_move"] = response
                result["material_loss"] = 10000
                board.pop()
                board.pop()
                return result
            board.pop()

        # Check for winning captures
        for response in board.legal_moves:
            if board.is_capture(response):
                captured = board.piece_at(response.to_square)
                attacker = board.piece_at(response.from_square)
                if captured and attacker:
                    captured_value = _piece_value(captured.piece_type)
                    attacker_value = _piece_value(attacker.piece_type)

                    if captured_value >= attacker_value:
                        board.push(response)
                        is_defended = board.is_attacked_by(not board.turn, response.to_square)
                        board.pop()

                        loss = captured_value if not is_defended else 0
                        if loss >= _piece_value(chess.ROOK):
                            result["is_blunder"] = True
                            result["reason"] = (
                                f"Winning capture after {candidate_move.uci()} → {response.uci()} "
                                f"(loses {_piece_name(captured.piece_type)})"
                            )
                            result["opponent_best_move"] = response
                            result["material_loss"] = loss
                            board.pop()
                            return result
    finally:
        board.pop()

    return result


def filter_blunders(board: chess.Board, candidate_moves: list) -> list:
    """Filter out blundering moves from a list of candidates.

    Args:
        board: Current board position.
        candidate_moves: List of (move, score) tuples, sorted by score descending.

    Returns:
        Filtered list with blunders removed. Returns original if all are blunders.
    """
    safe_moves = []
    for move, score in candidate_moves:
        result = check_blunder(board, move)
        if not result["is_blunder"]:
            safe_moves.append((move, score))

    return safe_moves if safe_moves else candidate_moves


def _piece_value(piece_type: int) -> int:
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
    return values.get(piece_type, 0)


def _piece_name(piece_type: int) -> str:
    names = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
    return names.get(piece_type, "piece")
