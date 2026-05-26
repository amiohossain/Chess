import torch
import pytest
from src.model.chess_net import ChessNet
from src.config import ModelConfig


class TestChessNet:
    @pytest.fixture
    def model(self):
        config = ModelConfig()
        return ChessNet(config)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 119, 8, 8)

    def test_forward_output_shapes(self, model, sample_input):
        policy, value = model(sample_input)
        assert policy.shape == (4, 4096), f"Expected (4, 4096), got {policy.shape}"
        assert value.shape == (4, 1), f"Expected (4, 1), got {value.shape}"

    def test_value_in_range(self, model, sample_input):
        _, value = model(sample_input)
        assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

    def test_parameter_count(self):
        config = ModelConfig(filters=384, num_blocks=10)
        model = ChessNet(config)
        total_params = sum(p.numel() for p in model.parameters())
        assert 30_000_000 < total_params < 45_000_000

    def test_policy_masking(self, model):
        logits = torch.randn(2, 4096)
        mask = torch.zeros(2, 4096)
        mask[:, :10] = 1.0
        probs = model.get_policy(logits, mask)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2))
        assert torch.all(probs[:, 10:] == 0.0)

    def test_different_batch_sizes(self, model):
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 119, 8, 8)
            policy, value = model(x)
            assert policy.shape[0] == batch_size
            assert value.shape[0] == batch_size

    def test_model_save_load(self, model, tmp_path):
        x = torch.randn(2, 119, 8, 8)
        model.eval()
        with torch.no_grad():
            policy_before, value_before = model(x)
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        model_loaded = ChessNet(ModelConfig())
        model_loaded.load_state_dict(torch.load(save_path, weights_only=True))
        model_loaded.eval()
        with torch.no_grad():
            policy_after, value_after = model_loaded(x)
        assert torch.allclose(policy_before, policy_after)
        assert torch.allclose(value_before, value_after)
