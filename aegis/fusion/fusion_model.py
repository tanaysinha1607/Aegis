"""
Fuses temporal (Transformer) and relational (GNN) signals into a calibrated risk score.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionRiskModel(nn.Module):
    def __init__(self, gnn_dim: int, hidden_dim: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        # transformer scalar + gnn user embedding
        in_dim = 1 + gnn_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, transformer_signal: torch.Tensor, gnn_user_emb: torch.Tensor) -> torch.Tensor:
        """
        transformer_signal: [B] (typically transformer logit / pre-sigmoid)
        gnn_user_emb: [B, gnn_dim]
        """
        z = torch.cat([transformer_signal.unsqueeze(-1), gnn_user_emb], dim=-1)
        logit = self.net(z).squeeze(-1)
        return torch.sigmoid(logit)
