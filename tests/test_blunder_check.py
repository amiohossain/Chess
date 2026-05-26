import chess
import pytest
from src.search.blunder_check import check_blunder, filter_blunders


class TestBlunderCheck:
    def test_non_blunder_move(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = check_blunder(board, move)
        assert not result["is_blunder"]

    def test_checkmate_override(self):
        """Delivering checkmate should never be flagged as a blunder."""
        board = chess.Board("k7/8/8/8/8/8/8/R3K3 w - - 0 1")
        move = chess.Move.from_uci("a1a8")
        result = check_blunder(board, move)
        assert not result["is_blunder"]

    def test_filter_blunders_empty(self):
        board = chess.Board()
        assert filter_blunders(board, []) == []

    def test_filter_blunders_preserves_order(self):
        board = chess.Board()
        moves = [
            (chess.Move.from_uci("e2e4"), 0.9),
            (chess.Move.from_uci("d2d4"), 0.8),
            (chess.Move.from_uci("g1f3"), 0.7),
        ]
        result = filter_blunders(board, moves)
        assert len(result) == 3
        assert result[0][0].uci() == "e2e4"
