"""ChessNet: A 10-block residual CNN with dual policy+value heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU activation: swish(x) * gate(x) with learnable gating."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.silu(x) * torch.sigmoid(self.beta * self.gate(x))


class ResidualBlock(nn.Module):
    """Pre-activation residual block with SwiGLU activation."""

    def __init__(self, channels: int, activation: str = "swiglu"):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = SwiGLU(channels) if activation == "swiglu" else nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = SwiGLU(channels) if activation == "swiglu" else nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class ChessNet(nn.Module):
    """Full chess neural network with policy and value heads."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.input_conv = nn.Conv2d(
            config.input_channels, config.filters,
            kernel_size=3, padding=1, bias=False,
        )
        self.input_bn = nn.BatchNorm2d(config.filters)

        self.blocks = nn.ModuleList([
            ResidualBlock(config.filters, config.activation)
            for _ in range(config.num_blocks)
        ])
        self.dropout = nn.Dropout2d(config.dropout)

        # Policy head
        self.policy_conv = nn.Conv2d(
            config.filters, config.policy_channels,
            kernel_size=1, bias=False,
        )
        self.policy_bn = nn.BatchNorm2d(config.policy_channels)
        self.policy_fc = nn.Linear(
            config.policy_channels * config.board_size * config.board_size,
            config.policy_output_size,
        )

        # Value head
        self.value_conv = nn.Conv2d(
            config.filters, 1,
            kernel_size=1, bias=False,
        )
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.board_size * config.board_size, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.silu(x)

        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def get_policy(self, policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Apply legal move mask and return softmax probabilities."""
        masked = policy_logits + (1.0 - legal_mask) * -1e9
        return F.softmax(masked, dim=-1)
