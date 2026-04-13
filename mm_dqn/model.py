from __future__ import annotations

import torch
import torch.nn as nn


class AttnLOBEncoder(nn.Module):
    def __init__(self, n_levels: int, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        in_dim = 4 * n_levels
        self.spatial = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.inception_1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.inception_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.inception_3 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1), nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
        self.proj = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        # lob_window: [b, t, 4*n]
        x = lob_window.transpose(1, 2)  # [b, 4*n, t]
        x = self.spatial(x)
        x_cat = torch.cat([self.inception_1(x), self.inception_2(x), self.inception_3(x)], dim=1)
        x = self.proj(x_cat).transpose(1, 2)  # [b, t, h]
        x, _ = self.attn(x, x, x, need_weights=False)
        return x[:, -1, :]


class DQNNet(nn.Module):
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
