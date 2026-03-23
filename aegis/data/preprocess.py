"""
Build transformer sequences and a PyTorch Geometric graph from tabular transactions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _encode_locations(df: pd.DataFrame, location_vocab: Optional[List[str]] = None) -> Tuple[pd.Series, List[str]]:
    if location_vocab is None:
        location_vocab = sorted(df["location"].astype(str).unique().tolist())
    loc_to_idx = {loc: i for i, loc in enumerate(location_vocab)}
    return df["location"].map(lambda x: loc_to_idx[str(x)]), location_vocab


def build_node_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int]:
    """Map string IDs to contiguous indices; returns (user_map, device_map, merchant_map, num_nodes)."""
    users = sorted(df["user_id"].astype(str).unique())
    devices = sorted(df["device_id"].astype(str).unique())
    merchants = sorted(df["merchant_id"].astype(str).unique())
    user_map = {u: i for i, u in enumerate(users)}
    d0 = len(users)
    device_map = {d: d0 + i for i, d in enumerate(devices)}
    m0 = d0 + len(devices)
    merchant_map = {m: m0 + i for i, m in enumerate(merchants)}
    num_nodes = m0 + len(merchants)
    return user_map, device_map, merchant_map, num_nodes


def build_edge_index(
    df: pd.DataFrame,
    user_map: Dict[str, int],
    device_map: Dict[str, int],
    merchant_map: Dict[str, int],
) -> torch.Tensor:
    """Undirected edges: user–device and user–merchant for each transaction."""
    src: List[int] = []
    dst: List[int] = []
    for _, row in df.iterrows():
        u = user_map[str(row["user_id"])]
        d = device_map[str(row["device_id"])]
        m = merchant_map[str(row["merchant_id"])]
        src.extend([u, d, u, m])
        dst.extend([d, u, m, u])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def _row_features(g: pd.DataFrame, loc_to_idx: Dict[str, int], am_mean: float, am_std: float) -> np.ndarray:
    """Feature matrix for ordered group `g` (same order as sorted timestamps)."""
    amounts = g["amount"].astype(float).values
    am_log = np.log1p(amounts)
    am_norm = (am_log - am_mean) / am_std
    ts = pd.to_datetime(g["timestamp"]).dt
    hour = ts.hour.values + ts.minute.values / 60.0
    hour_norm = hour / 24.0
    dow = ts.dayofweek.values
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    loc_ids = g["location"].astype(str).map(lambda x: float(loc_to_idx[x])).values.astype(np.float32)
    return np.stack([am_norm, hour_norm, dow_sin, dow_cos, loc_ids], axis=1).astype(np.float32)


def build_sequence_tensors(
    df: pd.DataFrame,
    seq_len: int,
    location_vocab: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    For each user, sliding windows over ordered transactions.
    Features per step: [amount_norm, hour_norm, dow_sin, dow_cos, loc_id].
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    loc_ids, vocab = _encode_locations(df, location_vocab)
    df["_loc_id"] = loc_ids
    loc_to_idx = {loc: i for i, loc in enumerate(vocab)}

    am_log_all = np.log1p(df["amount"].astype(float).values)
    am_mean = float(am_log_all.mean())
    am_std = float(am_log_all.std() + 1e-6)

    all_x: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    all_y: List[float] = []
    all_users: List[str] = []

    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp")
        f = _row_features(g, loc_to_idx, am_mean, am_std)
        labels = g["risk_label"].values.astype(np.float32)
        if len(g) < 2:
            continue
        for end in range(1, len(g)):
            start = max(0, end - seq_len + 1)
            chunk = f[start : end + 1]
            L = chunk.shape[0]
            pad = seq_len - L
            if pad > 0:
                pad_arr = np.zeros((pad, chunk.shape[1]), dtype=np.float32)
                chunk_p = np.vstack([pad_arr, chunk])
                mask_row = np.array([False] * pad + [True] * L, dtype=bool)
            else:
                chunk_p = chunk[-seq_len:]
                mask_row = np.ones((seq_len,), dtype=bool)
            all_x.append(torch.from_numpy(chunk_p.astype(np.float32)))
            all_masks.append(torch.from_numpy(mask_row))
            all_y.append(float(labels[end]))
            all_users.append(str(uid))

    if not all_x:
        raise ValueError("No sequences produced — increase data size.")

    X = torch.stack(all_x, dim=0)
    mask_t = torch.stack(all_masks, dim=0)
    y = torch.tensor(all_y, dtype=torch.float32)
    return X, mask_t, y, all_users, vocab


def build_pyg_data(
    df: pd.DataFrame,
    num_nodes: int,
    user_map: Dict[str, int],
    device_map: Dict[str, int],
    merchant_map: Dict[str, int],
    node_labels: Optional[torch.Tensor] = None,
) -> Data:
    """Construct a single `Data` object for transductive node classification on users."""
    edge_index = build_edge_index(df, user_map, device_map, merchant_map)
    if node_labels is None:
        # default: user fraud label = max risk over their transactions
        ulab = df.groupby("user_id")["risk_label"].max()
        y = torch.zeros(num_nodes, dtype=torch.long)
        for u, v in ulab.items():
            y[user_map[str(u)]] = int(v)
    else:
        y = node_labels
    x = torch.randn(num_nodes, 16)  # placeholder features — model uses own embedding
    data = Data(x=x, edge_index=edge_index, y=y)
    data.user_map = user_map
    data.device_map = device_map
    data.merchant_map = merchant_map
    data.num_users = len(user_map)
    return data


def train_val_test_split_users(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = df["user_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = len(users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_u = set(users[:n_train])
    val_u = set(users[n_train : n_train + n_val])
    test_u = set(users[n_train + n_val :])
    tr = df[df["user_id"].isin(train_u)].copy()
    va = df[df["user_id"].isin(val_u)].copy()
    te = df[df["user_id"].isin(test_u)].copy()
    return tr, va, te


def filter_sequences_by_users(
    df: pd.DataFrame,
    seq_len: int,
    allowed_users: Set[str],
    location_vocab: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Build sequences only for users in allowed_users."""
    df = df[df["user_id"].isin(allowed_users)].copy()
    X, mask, y, users, vocab = build_sequence_tensors(df, seq_len, location_vocab)
    return X, mask, y, users, vocab
