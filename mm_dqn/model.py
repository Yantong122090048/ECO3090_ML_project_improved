from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLOBEncoder(nn.Module):
    """
    Original AttnLOBEncoder with minor improvements:
    - Uses the same architecture (Conv1d + Inception + Self-Attention)
    """
    def __init__(self, n_levels: int, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        in_dim = 4 * n_levels

        self.spatial = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )

        # Inception module
        self.inception_1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.inception_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.inception_3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )
        self.proj = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        # lob_window: [b, t, 4*n]
        x = lob_window.transpose(1, 2)  # [b, 4*n, t]
        x = self.spatial(x)

        # Inception
        x_cat = torch.cat([
            self.inception_1(x),
            self.inception_2(x),
            self.inception_3(x)
        ], dim=1)
        x = self.proj(x_cat).transpose(1, 2)  # [b, t, h]

        # Self-attention
        x, _ = self.attn(x, x, x, need_weights=False)
        return x[:, -1, :]  # Take last timestep


class ResidualLinearBlock(nn.Module):
    """
    NEW: Residual block with LayerNorm for MLP
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.norm(x + residual)


class DuelingDQNNet(nn.Module):
    """
    NEW: Dueling DQN Architecture
    Separates state value V(s) and advantage A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """
    def __init__(
        self,
        n_levels: int,
        dynamic_dim: int,
        agent_dim: int,
        hidden_dim: int,
        action_dim: int,
        n_heads: int = 4,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.encoder = AttnLOBEncoder(n_levels=n_levels, hidden_dim=hidden_dim, n_heads=n_heads)

        # Input projection
        input_dim = hidden_dim + dynamic_dim + agent_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim * 2)
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.input_norm = nn.Identity()

        # Value stream: V(s) - how good is this state?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single scalar value
        )

        # Advantage stream: A(s,a) - how good is each action in this state?
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Value for each action
        )

        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

    def forward(self, lob_window: torch.Tensor, dynamic_state: torch.Tensor, agent_state: torch.Tensor) -> torch.Tensor:
        # Encode LOB
        z = self.encoder(lob_window)

        # Concatenate all inputs
        x = torch.cat([z, dynamic_state, agent_state], dim=-1)

        # Input projection
        x = self.input_proj(x)
        if self.use_layer_norm:
            x = self.input_norm(x)

        # Compute value and advantage
        value = self.value_stream(x)  # [batch, 1]
        advantage = self.advantage_stream(x)  # [batch, action_dim]

        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        # This ensures the advantages have zero mean
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class OriginalDQNNet(nn.Module):
    """
    Original DQNNet for comparison
    """
    def __init__(self, n_levels: int, dynamic_dim: int, agent_dim: int, hidden_dim: int, action_dim: int, n_heads: int = 4):
        super().__init__()
        self.encoder = AttnLOBEncoder(n_levels=n_levels, hidden_dim=hidden_dim, n_heads=n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + dynamic_dim + agent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, lob_window: torch.Tensor, dynamic_state: torch.Tensor, agent_state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(lob_window)
        x = torch.cat([z, dynamic_state, agent_state], dim=-1)
        return self.mlp(x)


def get_model(config):
    """Factory function to get model based on config"""
    if config.use_dueling:
        return DuelingDQNNet(
            n_levels=10,  # From EnvConfig
            dynamic_dim=config.dynamic_dim,
            agent_dim=config.agent_dim,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            n_heads=config.n_heads,
            use_residual=config.use_residual,
            use_layer_norm=config.use_layer_norm
        )
    else:
        return OriginalDQNNet(
            n_levels=10,
            dynamic_dim=config.dynamic_dim,
            agent_dim=config.agent_dim,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            n_heads=config.n_heads
        )
