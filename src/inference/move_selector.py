"""Full inference pipeline: integrates MCTS, trap bias, and blunder check."""
import chess
import numpy as np
from src.config import MCTSConfig
from src.model.chess_net import ChessNet
from src.search.mcts import MCTS
from src.search.blunder_check import filter_blunders


def select_move(
    board: chess.Board,
    model: ChessNet,
    mcts_config: MCTSConfig,
    trap_config=None,
    temperature: float = 0.15,
    trap_db: dict = None,
):
    """Full inference pipeline to select the best move.

    Args:
        board: Current position.
        model: Trained ChessNet model.
        mcts_config: MCTS configuration.
        trap_config: Trap configuration (optional).
        temperature: Selection temperature.
        trap_db: Dict mapping FEN piece positions to trap move UCI strings.

    Returns:
        (selected_move, principal_variation, metadata)
    """
    mcts = MCTS(model, mcts_config)
    policy, root_value = mcts.search(board)

    metadata = {"root_value": root_value, "moves_considered": len(policy)}

    # Apply trap bias
    if trap_config and trap_db and hasattr(trap_config, 'trap_boost_factor') and trap_config.trap_boost_factor > 1.0:
        if root_value >= getattr(trap_config, 'trap_guard_threshold', -0.3):
            fen_key = board.fen().split(" ")[0]
            trap_moves = trap_db.get(fen_key, [])
            if trap_moves:
                policy = dict(policy)
                for trap_uci in trap_moves:
                    if trap_uci in policy:
                        policy[trap_uci] *= trap_config.trap_boost_factor
                total = sum(policy.values())
                if total > 0:
                    for k in policy:
                        policy[k] /= total

    # Top-20 candidates for blunder check
    sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [(chess.Move.from_uci(uci), prob) for uci, prob in sorted_moves[:20]]

    # Blunder check on top 3
    safe_moves = filter_blunders(board, top_candidates[:3])

    if len(safe_moves) < len(top_candidates[:3]) and len(safe_moves) > 0:
        metadata["blunders_filtered"] = len(top_candidates[:3]) - len(safe_moves)
    elif len(safe_moves) == 0:
        safe_moves = top_candidates[:5]
        metadata["blunders_filtered"] = "expanded_search"

    # Tempered selection
    if temperature == 0.0:
        move = safe_moves[0][0]
    else:
        probs = np.array([p for _, p in safe_moves], dtype=np.float64)
        probs = np.power(np.maximum(probs, 1e-8), 1.0 / temperature)
        probs /= probs.sum()
        idx = np.random.choice(len(safe_moves), p=probs)
        move = safe_moves[idx][0]

    metadata["method"] = "argmax" if temperature == 0.0 else "tempered"
    metadata["selected_from"] = len(safe_moves)

    pv = [move]
    return move, pv, metadata
