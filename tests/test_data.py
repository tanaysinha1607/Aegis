import numpy as np
import pandas as pd

from data.preprocess import build_node_mappings, build_sequence_tensors
from data.synthetic_generator import SyntheticDataGenerator


def test_synthetic_generator_schema():
    gen = SyntheticDataGenerator(num_users=20, num_devices=15, num_merchants=12, seed=0)
    df = gen.generate_transactions()
    assert set(df.columns) >= {
        "transaction_id",
        "user_id",
        "merchant_id",
        "device_id",
        "amount",
        "timestamp",
        "location",
        "risk_label",
    }
    assert df["risk_label"].isin([0, 1]).all()
    assert len(df) > 0


def test_graph_and_sequences():
    gen = SyntheticDataGenerator(num_users=15, seed=1)
    df = gen.generate_transactions()
    user_map, device_map, merchant_map, num_nodes = build_node_mappings(df)
    assert num_nodes == len(user_map) + len(device_map) + len(merchant_map)
    X, mask, y, users, vocab = build_sequence_tensors(df, seq_len=8)
    assert X.ndim == 3 and mask.shape == X.shape[:2] and len(y) == len(users)
