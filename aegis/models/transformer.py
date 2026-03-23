"""
Temporal behavior model: Transformer over transaction sequences with positional encoding.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalRiskTransformer(nn.Module):
    """
    Input channels: [amount_norm, hour_norm, dow_sin, dow_cos, location_index].
    Outputs a scalar risk probability in (0,1) using the last valid timestep representation.
    """

    def __init__(
        self,
        num_locations: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        loc_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_locations = num_locations
        self.loc_emb = nn.Embedding(num_locations, loc_emb_dim)
        in_dim = 4 + loc_emb_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=256, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, 5] — last dim is location index stored as float.
        mask: [B, L] bool, True = valid token (not padding).
        """
        logit = self.forward_logit(x, mask)
        return torch.sigmoid(logit)

    def forward_logit(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Same as `forward`, but returns logits before sigmoid.
        Useful for resilience/stability where probabilities may saturate.
        """
        loc = x[:, :, 4].long().clamp(0, self.num_locations - 1)
        le = self.loc_emb(loc)
        z = torch.cat([x[:, :, :4], le], dim=-1)
        h = self.input_proj(z)
        h = self.pos_encoder(h)
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        pad = ~mask
        out = self.encoder(h, src_key_padding_mask=pad)
        out = self.norm(out)
        idx = mask.long().sum(dim=1) - 1
        idx = idx.clamp(min=0)
        batch = torch.arange(out.size(0), device=out.device)
        pooled = out[batch, idx]
        logit = self.head(pooled).squeeze(-1)
        return logit
