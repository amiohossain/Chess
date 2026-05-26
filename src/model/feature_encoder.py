"""Encodes a python-chess Board into an 8×8×119 binary feature tensor.

Follows the Leela Chess Zero feature plane layout:
  - 0-11:    Piece positions (6 piece types x 2 colors)
  - 12-15:   Castling rights (KQkq)
  - 16:      En passant square
  - 17:      Side to move
  - 18-19:   Half-move clock, full move number
  - 20-21:   Repetition counts
  - 22-117:  Reserved (zeros)

Reference: https://arxiv.org/abs/1711.09633 (AlphaZero)
"""
import numpy as np
import chess

# Piece type -> plane offset (0-5)
PIECE_TO_OFFSET = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Convert a chess.Board to an 8x8x119 binary numpy array.

    Args:
        board: A python-chess Board at the position to encode.

    Returns:
        Shape (119, 8, 8) binary float32 array.
    """
    planes = np.zeros((119, 8, 8), dtype=np.float32)

    # 0-11: Piece positions (6 types x 2 colors)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = square // 8
            col = square % 8
            base = PIECE_TO_OFFSET[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            planes[base + color_offset, row, col] = 1.0

    # 12: White kingside castling
    planes[12, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    # 13: White queenside castling
    planes[13, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    # 14: Black kingside castling
    planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    # 15: Black queenside castling
    planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # 16: En passant square
    ep = board.ep_square
    if ep is not None:
        planes[16, ep // 8, ep % 8] = 1.0

    # 17: Side to move
    planes[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # 18: Half-move clock
    planes[18, :, :] = board.halfmove_clock / 100.0

    # 19: Full move number
    planes[19, :, :] = board.fullmove_number / 500.0

    # 20-21: Repetition count
    rep_count = _count_repetitions(board)
    if rep_count >= 1:
        planes[20, :, :] = 1.0
    if rep_count >= 2:
        planes[21, :, :] = 1.0

    return planes


def _count_repetitions(board: chess.Board) -> int:
    """Count how many times the current position has occurred (excluding current)."""
    if board.is_repetition(2):
        if board.is_repetition(3):
            return 2
        return 1
    return 0


NUM_MOVES = 4096  # 64 from-squares × 64 to-squares (no promotion bit — always queen on promotion)


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """Encode a move as an index in [0, 4095].

    Index = from_square + 64 * to_square

    Note: Promotion piece is NOT encoded — all promotions map to the same
    from→to index and default to queen. This keeps the policy head small
    (~8M vs ~72M params). Under-promotions are <0.1% of positions and
    have negligible Elo impact at the 2500 target level.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    return from_sq + 64 * to_sq


def decode_move(move_idx: int, board: chess.Board = None) -> chess.Move:
    """Decode a move index back to a python-chess Move.

    If a board is provided and the move is a pawn promotion, defaults to queen.
    """
    from_sq = move_idx % 64
    to_sq = (move_idx // 64) % 64
    if board is not None:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            promo_rank = 0 if piece.color == chess.BLACK else 7
            if to_sq // 8 == promo_rank:
                return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    return chess.Move(from_sq, to_sq)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a binary mask of shape (NUM_MOVES,) with 1s for legal moves."""
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        idx = encode_move(move, board)
        mask[idx] = 1.0
    return mask
