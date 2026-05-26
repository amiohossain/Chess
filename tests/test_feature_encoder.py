"""tests/test_feature_encoder.py"""
import numpy as np
import chess
import pytest
from src.model.feature_encoder import (
    encode_board,
    encode_move,
    decode_move,
    legal_move_mask,
)


class TestEncodeBoard:
    def test_starting_position_shape(self):
        board = chess.Board()
        planes = encode_board(board)
        assert planes.shape == (119, 8, 8)
        assert planes.dtype == np.float32

    def test_starting_position_pieces(self):
        board = chess.Board()
        planes = encode_board(board)
        assert planes[0, 1, 0] == 1.0  # White pawn on a2
        assert planes[0, 1, 7] == 1.0  # White pawn on h2
        assert planes[6, 6, 0] == 1.0  # Black pawn on a7
        assert planes[5, 0, 4] == 1.0  # White king on e1
        assert planes[11, 7, 4] == 1.0  # Black king on e8

    def test_castling_rights(self):
        board = chess.Board()
        planes = encode_board(board)
        assert planes[12, 0, 0] == 1.0
        assert planes[13, 0, 0] == 1.0
        assert planes[14, 0, 0] == 1.0
        assert planes[15, 0, 0] == 1.0

    def test_side_to_move(self):
        board = chess.Board()
        planes = encode_board(board)
        assert planes[17, 0, 0] == 1.0
        board.push_san("e4")
        planes = encode_board(board)
        assert planes[17, 0, 0] == 0.0

    def test_en_passant(self):
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        planes = encode_board(board)
        assert planes[16, 2, 4] == 1.0


class TestMoveEncoding:
    def test_encode_decode_roundtrip(self):
        board = chess.Board()
        for move_san in ["e4", "d4", "Nf3", "Nc3", "g3"]:
            move = chess.Move.from_uci(board.parse_san(move_san).uci())
            idx = encode_move(move, board)
            decoded = decode_move(idx, board)
            assert decoded == move

    def test_legal_move_mask_starting_position(self):
        board = chess.Board()
        mask = legal_move_mask(board)
        assert mask.shape == (4096,)
        assert mask.sum() == 20
        assert all(m == 1.0 or m == 0.0 for m in mask)

    def test_legal_move_mask_midgame(self):
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        mask = legal_move_mask(board)
        assert mask.sum() > 20
