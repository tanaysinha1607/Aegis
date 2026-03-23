"""
Relational behavior: GraphSAGE or GAT on user / device / merchant graph.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class RelationalRiskGNN(nn.Module):
    """Node-level embeddings for users, devices, and merchants."""

    def __init__(
        self,
        num_nodes: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_type: Literal["graphsage", "gat"] = "graphsage",
        in_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.model_type = model_type
        self.node_emb = nn.Embedding(num_nodes, in_emb_dim)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = in_emb_dim
        for i in range(num_layers):
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            if model_type == "graphsage":
                self.convs.append(SAGEConv(in_dim, out_dim))
            else:
                self.convs.append(GATConv(in_dim, out_dim, heads=2, concat=False))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.classifier = nn.Linear(out_channels, 2)

    def forward(self, edge_index: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = self.node_emb.weight
        for conv, ln in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = ln(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_logits:
            return self.classifier(x)
        return x

    def user_logits(self, edge_index: torch.Tensor, user_indices: torch.Tensor) -> torch.Tensor:
        """Binary logits for a subset of user nodes."""
        logits = self.forward(edge_index, return_logits=True)
        return logits[user_indices]
