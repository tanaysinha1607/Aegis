"""
Aegis end-to-end pipeline: synthetic data → train dual engines → fusion → RAG → resilience.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.preprocess import (
    build_node_mappings,
    build_pyg_data,
    filter_sequences_by_users,
    train_val_test_split_users,
)
from data.synthetic_generator import generate_sample_dataset
from evaluation.metrics import auc_binary
from fusion.fusion_model import FusionRiskModel
from models.gnn import RelationalRiskGNN
from models.transformer import TemporalRiskTransformer
from rag.rag_pipeline import RAGPipeline
from resilience.resilience_test import ResilienceAnalyzer
from utils.config_loader import load_config
from utils.logger import setup_logging


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_transformer(
    model: TemporalRiskTransformer,
    X: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> None:
    model.to(device)
    X = X.to(device)
    mask = mask.to(device)
    y = y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = X.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(X[idx], mask[idx])
            loss = F.binary_cross_entropy(pred, y[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * idx.numel()
        avg = total_loss / n
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch == 0:
            print(f"  [Transformer] epoch {epoch+1}/{epochs} loss={avg:.4f}")


@torch.no_grad()
def predict_transformer(
    model: TemporalRiskTransformer,
    X: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    X = X.to(device)
    mask = mask.to(device)
    outs: list[float] = []
    n = X.size(0)
    for start in range(0, n, batch_size):
        pred = model(X[start : start + batch_size], mask[start : start + batch_size])
        outs.extend(pred.cpu().numpy().tolist())
    return np.array(outs, dtype=np.float32)


@torch.no_grad()
def predict_transformer_logits(
    model: TemporalRiskTransformer,
    X: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Pre-sigmoid transformer logits for better dynamic range in fusion."""
    model.eval()
    X = X.to(device)
    mask = mask.to(device)
    outs: list[float] = []
    n = X.size(0)
    for start in range(0, n, batch_size):
        logits = model.forward_logit(X[start : start + batch_size], mask[start : start + batch_size])
        outs.extend(logits.cpu().numpy().tolist())
    return np.array(outs, dtype=np.float32)


def train_gnn(
    model: RelationalRiskGNN,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    model.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(edge_index, return_logits=True)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch == 0:
            print(f"  [GNN] epoch {epoch+1}/{epochs} loss={float(loss.item()):.4f}")


@torch.no_grad()
def gnn_user_embeddings(model: RelationalRiskGNN, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    edge_index = edge_index.to(device)
    return model(edge_index, return_logits=False)


def train_fusion(
    fusion: FusionRiskModel,
    tf_pred: torch.Tensor,
    gnn_emb: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 0.01,
) -> None:
    fusion.to(device)
    tf_pred = tf_pred.to(device)
    gnn_emb = gnn_emb.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(fusion.parameters(), lr=lr)
    for epoch in range(epochs):
        fusion.train()
        opt.zero_grad()
        out = fusion(tf_pred, gnn_emb)
        loss = F.binary_cross_entropy(out, y)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Fusion] epoch {epoch+1}/{epochs} loss={float(loss.item()):.4f}")


def main() -> None:
    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file"),
    )
    set_seed(int(cfg.get("seed", 42)))
    device_str = cfg.get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    print("=" * 60)
    print("Aegis: Dual-Engine AI for Resilient Risk Detection")
    print("=" * 60)

    # 1) Synthetic data
    print("\n[1] Generating synthetic transactions + risk narratives...")
    df, narratives, csv_path, json_path = generate_sample_dataset(cfg)
    print(f"    Saved: {csv_path} ({len(df)} rows)")
    print(f"    Saved: {json_path} ({len(narratives)} narratives)")

    tr_df, va_df, te_df = train_val_test_split_users(
        df,
        float(cfg["training"]["train_ratio"]),
        float(cfg["training"]["val_ratio"]),
        int(cfg["seed"]),
    )
    train_users = set(tr_df["user_id"].astype(str).unique())
    val_users = set(va_df["user_id"].astype(str).unique())
    test_users = set(te_df["user_id"].astype(str).unique())

    seq_len = int(cfg["transformer"]["seq_len"])
    # Shared location vocab from training split
    X_tr, m_tr, y_tr, users_tr, vocab = filter_sequences_by_users(tr_df, seq_len, train_users, None)
    X_va, m_va, y_va, _, _ = filter_sequences_by_users(va_df, seq_len, val_users, vocab)
    X_te, m_te, y_te, users_te, _ = filter_sequences_by_users(te_df, seq_len, test_users, vocab)
    if X_te.size(0) == 0:
        raise RuntimeError(
            "No test sequences produced — increase num_users or lower seq_len in config."
        )

    num_locations = max(len(vocab), int(cfg["transformer"].get("num_locations", 32)))
    tcfg = cfg["transformer"]

    tf_model = TemporalRiskTransformer(
        num_locations=num_locations,
        d_model=int(tcfg["d_model"]),
        nhead=int(tcfg["nhead"]),
        num_layers=int(tcfg["num_layers"]),
        dim_feedforward=int(tcfg["dim_feedforward"]),
        dropout=float(tcfg["dropout"]),
    )

    print("\n[2] Training temporal Transformer...")
    train_transformer(
        tf_model,
        X_tr,
        m_tr,
        y_tr,
        device,
        epochs=int(tcfg["epochs"]),
        batch_size=int(tcfg["batch_size"]),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    # 3) Graph + GNN
    print("\n[3] Building graph and training GNN...")
    user_map, device_map, merchant_map, num_nodes = build_node_mappings(df)
    data = build_pyg_data(df, num_nodes, user_map, device_map, merchant_map)
    edge_index = data.edge_index

    train_idx = torch.tensor([user_map[u] for u in train_users], dtype=torch.long)
    gcfg = cfg["gnn"]
    gnn_model = RelationalRiskGNN(
        num_nodes=num_nodes,
        hidden_channels=int(gcfg["hidden_channels"]),
        out_channels=int(gcfg["out_channels"]),
        num_layers=int(gcfg["num_layers"]),
        dropout=float(gcfg["dropout"]),
        model_type=str(gcfg.get("model_type", "graphsage")),
    )
    train_gnn(
        gnn_model,
        edge_index,
        data.y,
        train_idx,
        device,
        epochs=int(gcfg["epochs"]),
        lr=float(gcfg["lr"]),
        weight_decay=float(gcfg["weight_decay"]),
    )

    # 4) Fusion (train on train split)
    print("\n[4] Training fusion head...")
    with torch.no_grad():
        tf_train_logits = torch.tensor(
            predict_transformer_logits(tf_model, X_tr, m_tr, device),
            dtype=torch.float32,
            device=device,
        )
        emb_all = gnn_user_embeddings(gnn_model, edge_index, device)
        gnn_tr = emb_all[[user_map[u] for u in users_tr]].clone().float()

    fusion = FusionRiskModel(gnn_dim=int(gcfg["out_channels"]), hidden_dim=int(cfg["fusion"]["hidden_dim"]))
    train_fusion(
        fusion,
        tf_train_logits,
        gnn_tr,
        y_tr.to(device),
        device,
        epochs=25,
        lr=0.02,
    )

    # 5) Evaluation (test split)
    print("\n[5] Evaluation (AUC-ROC on held-out users)...")
    y_true = y_te.numpy()

    tf_test = predict_transformer(tf_model, X_te, m_te, device)
    auc_tf = auc_binary(y_true, tf_test)

    with torch.no_grad():
        logits = gnn_model(edge_index.to(device), return_logits=True)
        prob_gnn = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    gnn_user_idx = np.array([user_map[u] for u in users_te], dtype=np.int64)
    auc_gnn = auc_binary(y_true, prob_gnn[gnn_user_idx])

    with torch.no_grad():
        emb_all = gnn_user_embeddings(gnn_model, edge_index, device)
        tf_te_logits = torch.tensor(
            predict_transformer_logits(tf_model, X_te, m_te, device),
            dtype=torch.float32,
            device=device,
        )
        gnn_te = emb_all[[user_map[u] for u in users_te]]
        fused = fusion(tf_te_logits, gnn_te).cpu().numpy()
    auc_fused = auc_binary(y_true, fused)

    print(f"    Transformer AUC: {auc_tf:.4f}")
    print(f"    GNN (user-level) AUC: {auc_gnn:.4f}")
    print(f"    Fused AUC: {auc_fused:.4f}")

    # 6) RAG sample
    print("\n[6] RAG explanation (sample transaction)...")
    r_cfg = cfg["rag"]
    rag = RAGPipeline(
        embedding_model=r_cfg["embedding_model"],
        generator_model=r_cfg["generator_model"],
        top_k=int(r_cfg["top_k"]),
        max_context_chars=int(r_cfg["max_context_chars"]),
        max_new_tokens=int(r_cfg["max_new_tokens"]),
        latency_budget_seconds=float(r_cfg["latency_budget_seconds"]),
        openai_model=str(r_cfg.get("openai_model", "gpt-4o-mini")),
    )
    rag.build_corpus(narratives)
    sample_row = te_df.iloc[0]
    summary = (
        f"User {sample_row['user_id']} transacted ${sample_row['amount']:.2f} at merchant "
        f"{sample_row['merchant_id']} from device {sample_row['device_id']} in {sample_row['location']}."
    )
    explanation, latency, meta = rag.explain(summary)
    print(f"    Latency: {latency:.2f}s")
    print(f"    Explanation: {explanation[:500]}{'...' if len(explanation) > 500 else ''}")

    # 7) Resilience
    print("\n[7] Resilience stress testing (subset of test batch)...")
    rcfg = cfg["resilience"]
    analyzer = ResilienceAnalyzer(
        num_scenarios=int(rcfg["num_scenarios"]),
        noise_std_amount=float(rcfg["noise_std_amount"]),
        spike_multiplier=float(rcfg["spike_multiplier"]),
    )
    n_sub = min(16, X_te.size(0))
    X_sub = X_te[:n_sub].clone()
    m_sub = m_te[:n_sub].clone()

    def predict_fn(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        # Use logits (pre-sigmoid) to keep resilience sensitive under saturation.
        return predict_transformer_logits(tf_model, x, mask, device)

    res_stats = analyzer.run(predict_fn, X_sub, m_sub, plot_path=rcfg.get("plot_path"))
    print(f"    Mean resilience factor: {res_stats['mean_resilience']:.4f}")
    print(f"    Plot: {rcfg.get('plot_path')}")

    out_dir = Path(cfg["data"].get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "tf": tf_model.state_dict(),
            "gnn": gnn_model.state_dict(),
            "fusion": fusion.state_dict(),
            "num_nodes": num_nodes,
            "num_locations": num_locations,
            "user_map": user_map,
            "vocab": vocab,
            "seq_len": seq_len,
        },
        ckpt_path,
    )
    print(f"\n    Saved checkpoint for dashboard / reuse: {ckpt_path}")

    print("\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    main()
