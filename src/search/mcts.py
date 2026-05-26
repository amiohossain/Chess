"""Monte Carlo Tree Search engine for chess.

AlphaZero-style MCTS with:
  - UCB exploration (c_puct = 1.4)
  - Model policy as prior P(s, a)
  - Model value as leaf evaluation V(s)
  - Dirichlet noise at root for exploration
  - Visit-count-weighted move selection
"""
import math
import chess
import torch
import numpy as np
from src.config import MCTSConfig
from src.model.chess_net import ChessNet
from src.model.feature_encoder import encode_board, legal_move_mask, decode_move


class Node:
    """A node in the MCTS tree."""

    __slots__ = ("visit_count", "total_value", "prior", "children", "parent", "move")

    def __init__(self, prior: float = 0.0, move=None, parent=None):
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.children = {}
        self.parent = parent
        self.move = move

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, policy_probs: dict, legal_moves: list):
        for move in legal_moves:
            move_uci = move.uci()
            prob = policy_probs.get(move_uci, 0.0)
            if prob > 0:
                self.children[move_uci] = Node(prior=prob, move=move, parent=self)

    def best_child(self, c_puct: float):
        best_score = -float("inf")
        best_child = None
        best_move = None
        for move_uci, child in self.children.items():
            ucb = child.value + c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_child = child
                best_move = move_uci
        return best_move, best_child

    def update(self, value: float):
        self.visit_count += 1
        self.total_value += value


class MCTS:
    """Monte Carlo Tree Search using a neural network policy-value oracle."""

    def __init__(self, model: ChessNet, config: MCTSConfig, device: torch.device = None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    @torch.no_grad()
    def search(self, board: chess.Board):
        """Run MCTS from the given position.

        Returns:
            policy: dict {move_uci: visit_count / total_visits}
            value: float (root position evaluation)
        """
        root = Node()

        policy_logits, root_value = self._evaluate(board)
        legal_moves = list(board.legal_moves)
        policy_probs = self._policy_to_dict(policy_logits, legal_moves, board)

        # Add Dirichlet noise at root
        if self.config.dirichlet_weight > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                policy_probs[move.uci()] = (
                    (1 - self.config.dirichlet_weight) * policy_probs.get(move.uci(), 0.0)
                    + self.config.dirichlet_weight * noise[i]
                )

        root.expand(policy_probs, legal_moves)

        for _ in range(self.config.num_simulations):
            node = root
            path = [node]
            board_copy = board.copy()

            # Select
            while node.is_expanded() and not board_copy.is_game_over():
                _, node = node.best_child(self.config.c_puct)
                path.append(node)
                board_copy.push(node.move)
                if board_copy.is_game_over():
                    break

            # Evaluate leaf
            if board_copy.is_game_over():
                leaf_value = self._game_outcome(board_copy, original_turn=board.turn)
            else:
                _, leaf_value = self._evaluate(board_copy)

            # Expand leaf
            if not board_copy.is_game_over() and not node.is_expanded():
                leaf_policy, _ = self._evaluate(board_copy)
                leaf_legal = list(board_copy.legal_moves)
                leaf_probs = self._policy_to_dict(leaf_policy, leaf_legal, board_copy)
                node.expand(leaf_probs, leaf_legal)

            # Backpropagate
            for n in reversed(path):
                n.update(leaf_value)
                leaf_value = -leaf_value

        total_visits = sum(child.visit_count for child in root.children.values())
        policy = {}
        for move_uci, child in root.children.items():
            policy[move_uci] = child.visit_count / max(total_visits, 1)

        return policy, root_value

    def select_move(self, policy: dict, temperature: float = 0.15) -> chess.Move:
        """Select a move from the MCTS policy distribution."""
        if temperature == 0.0:
            best_uci = max(policy, key=policy.get)
            return chess.Move.from_uci(best_uci)

        moves = list(policy.keys())
        probs = np.array([policy[m] for m in moves])

        if temperature != 1.0:
            probs = np.power(np.maximum(probs, 1e-8), 1.0 / temperature)

        probs /= probs.sum()
        selected = np.random.choice(moves, p=probs)
        return chess.Move.from_uci(selected)

    @torch.no_grad()
    def _evaluate(self, board):
        planes = encode_board(board)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(tensor)

        mask = torch.from_numpy(legal_move_mask(board)).unsqueeze(0).to(self.device)
        policy_logits = policy_logits + (1.0 - mask) * -1e9

        return policy_logits.squeeze(0).cpu().numpy(), value.item()

    def _policy_to_dict(self, policy_logits, legal_moves, board):
        import scipy.special
        probs = scipy.special.softmax(policy_logits)
        result = {}
        for move in legal_moves:
            idx = (move.from_square + 64 * move.to_square)
            result[move.uci()] = float(probs[idx])
        return result

    @staticmethod
    def _game_outcome(board, original_turn):
        if board.is_checkmate():
            # Current player won
            return 1.0 if board.turn != original_turn else -1.0
        return 0.0
