import chess
import torch
import pytest
from src.search.mcts import MCTS, Node
from src.config import MCTSConfig
from src.model.chess_net import ChessNet
from src.config import ModelConfig


@pytest.fixture
def model():
    return ChessNet(ModelConfig())


@pytest.fixture
def mcts(model):
    return MCTS(model, MCTSConfig(num_simulations=50), device=torch.device("cpu"))


class TestNode:
    def test_new_node(self):
        node = Node(prior=0.5)
        assert node.visit_count == 0
        assert node.value == 0.0
        assert not node.is_expanded()

    def test_update(self):
        node = Node()
        node.update(0.5)
        assert node.visit_count == 1
        assert node.value == 0.5

    def test_expand(self):
        node = Node()
        board = chess.Board()
        moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        node.expand({"e2e4": 0.6, "d2d4": 0.4}, moves)
        assert node.is_expanded()
        assert len(node.children) == 2


class TestMCTS:
    def test_search_returns_policy(self, mcts):
        board = chess.Board()
        policy, value = mcts.search(board)
        assert isinstance(policy, dict)
        assert len(policy) > 0
        total = sum(policy.values())
        assert abs(total - 1.0) < 0.01

    def test_search_value_is_scalar(self, mcts):
        board = chess.Board()
        _, value = mcts.search(board)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_select_move_deterministic(self, mcts):
        board = chess.Board()
        policy, _ = mcts.search(board)
        move = mcts.select_move(policy, temperature=0.0)
        assert isinstance(move, chess.Move)

    def test_select_move_stochastic(self, mcts):
        board = chess.Board()
        policy, _ = mcts.search(board)
        move = mcts.select_move(policy, temperature=1.0)
        assert isinstance(move, chess.Move)
