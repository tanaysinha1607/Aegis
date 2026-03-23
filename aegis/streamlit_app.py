"""
Streamlit dashboard: transaction summary → risk score, RAG explanation, resilience (when checkpoint exists).

Run from `aegis/`:  streamlit run streamlit_app.py
Train first: `python main.py` (creates `outputs/checkpoint.pt`, CSV, narratives).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from datetime import datetime, timezone

from data.preprocess import build_node_mappings, build_pyg_data, filter_sequences_by_users
from data.synthetic_generator import generate_sample_dataset
from fusion.fusion_model import FusionRiskModel
from models.gnn import RelationalRiskGNN
from models.transformer import TemporalRiskTransformer
from rag.rag_pipeline import RAGPipeline
from resilience.resilience_test import ResilienceAnalyzer
from utils.config_loader import load_config


@st.cache_resource
def load_rag_with_cfg():
    cfg = load_config()
    r = cfg["rag"]
    narratives_path = Path(cfg["data"].get("narratives_json", "outputs/risk_narratives.json"))
    if not narratives_path.is_file():
        _, n, _, _ = generate_sample_dataset(cfg)
        narratives = n
    else:
        with open(narratives_path, encoding="utf-8") as f:
            narratives = json.load(f)
    pipe = RAGPipeline(
        embedding_model=r["embedding_model"],
        generator_model=r["generator_model"],
        top_k=int(r["top_k"]),
        max_context_chars=int(r["max_context_chars"]),
        max_new_tokens=int(r["max_new_tokens"]),
        latency_budget_seconds=float(r["latency_budget_seconds"]),
        openai_model=str(r.get("openai_model", "gpt-4o-mini")),
    )
    pipe.build_corpus(narratives)
    return pipe, cfg


@st.cache_resource
def load_trained_stack():
    cfg = load_config()
    ckpt_path = Path(cfg["data"].get("output_dir", "outputs")) / "checkpoint.pt"
    csv_path = Path(cfg["data"].get("sample_csv", "outputs/sample_transactions.csv"))
    if not ckpt_path.is_file() or not csv_path.is_file():
        return None, cfg
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    tcfg, gcfg = cfg["transformer"], cfg["gnn"]
    tf = TemporalRiskTransformer(
        num_locations=int(ckpt["num_locations"]),
        d_model=int(tcfg["d_model"]),
        nhead=int(tcfg["nhead"]),
        num_layers=int(tcfg["num_layers"]),
        dim_feedforward=int(tcfg["dim_feedforward"]),
        dropout=float(tcfg["dropout"]),
    )
    gnn = RelationalRiskGNN(
        num_nodes=int(ckpt["num_nodes"]),
        hidden_channels=int(gcfg["hidden_channels"]),
        out_channels=int(gcfg["out_channels"]),
        num_layers=int(gcfg["num_layers"]),
        dropout=float(gcfg["dropout"]),
        model_type=str(gcfg.get("model_type", "graphsage")),
    )
    fusion = FusionRiskModel(gnn_dim=int(gcfg["out_channels"]), hidden_dim=int(cfg["fusion"]["hidden_dim"]))
    tf.load_state_dict(ckpt["tf"])
    gnn.load_state_dict(ckpt["gnn"])
    fusion.load_state_dict(ckpt["fusion"])
    tf.eval()
    gnn.eval()
    fusion.eval()
    return (tf, gnn, fusion, ckpt, df), cfg


def heuristic_risk(amount: float, location: str, device: str) -> float:
    z = min(1.0, amount / 5000.0) * 1.2
    if any(x in location.upper() for x in ("UK", "DE", "SG")):
        z += 0.15
    if device.lower().startswith("dev_09") or device.lower().startswith("dev_08"):
        z += 0.1
    return float(1.0 / (1.0 + np.exp(-(z - 0.8))))


def _encode_candidate_step(
    amount: float,
    location: str,
    vocab: list[str],
    am_mean: float,
    am_std: float,
) -> np.ndarray:
    """Create one feature row compatible with transformer preprocessing."""
    # Use the same normalization stats as the user history we built `X` with.
    amount_norm = float((np.log1p(max(amount, 0.0)) - am_mean) / am_std)

    now = datetime.now(timezone.utc)
    hour_norm = float((now.hour + now.minute / 60.0) / 24.0)
    dow = now.weekday()
    dow_sin = float(np.sin(2 * np.pi * dow / 7.0))
    dow_cos = float(np.cos(2 * np.pi * dow / 7.0))
    loc_idx = float(vocab.index(location) if location in vocab else 0)
    return np.array([amount_norm, hour_norm, dow_sin, dow_cos, loc_idx], dtype=np.float32)


def main():
    st.set_page_config(page_title="Aegis Risk Dashboard", layout="wide")
    st.title("Aegis — Dual-Engine Risk Dashboard")
    rag, cfg = load_rag_with_cfg()
    stack, _ = load_trained_stack()

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (USD)", min_value=0.0, value=420.0, step=1.0)
        location = st.text_input("Location code", value="US-NY")
    with col2:
        device = st.text_input("Device id", value="dev_0001")
        merchant = st.text_input("Merchant id", value="merch_0005")

    user_id = st.text_input("User id (must exist in training CSV for model score)", value="user_0001")

    if st.button("Score & explain"):
        summary = (
            f"User {user_id} transacted ${amount:.2f} at merchant {merchant} "
            f"from device {device} in {location}."
        )
        with st.spinner("RAG explanation..."):
            explanation, lat, meta = rag.explain(summary)

        risk = heuristic_risk(amount, location, device)
        resilience_val = None
        tf_risk = None
        resilience_var_mean = None
        mode = "Heuristic risk (run `python main.py` to enable trained models)"

        if stack is not None:
            tf, gnn, fusion, ckpt, df = stack
            user_map = ckpt["user_map"]
            vocab = ckpt["vocab"]
            seq_len = int(ckpt.get("seq_len", int(cfg["transformer"]["seq_len"])))
            um, dm, mm, nn = build_node_mappings(df)
            data = build_pyg_data(df, nn, um, dm, mm)
            edge_index = data.edge_index
            if user_id in user_map:
                X, mask, _, _, _ = filter_sequences_by_users(df, seq_len, {str(user_id)}, vocab)
                if X.size(0) > 0:
                    x = X[-1:].clone()
                    m = mask[-1:].clone()
                    # Inject the current dashboard input into the last timestep.
                    user_hist = df[df["user_id"] == user_id].copy()
                    user_hist_amounts = user_hist["amount"].astype(float).values
                    am_log_user = np.log1p(user_hist_amounts)
                    am_mean = float(am_log_user.mean())
                    am_std = float(am_log_user.std() + 1e-6)
                    x[0, -1, :] = torch.from_numpy(
                        _encode_candidate_step(amount, location, vocab, am_mean, am_std)
                    )
                    m[0, -1] = True
                    with torch.no_grad():
                        tf_logits = tf.forward_logit(x, m)
                        tf_p = torch.sigmoid(tf_logits)
                        emb = gnn(edge_index, return_logits=False)
                        uid = user_map[user_id]
                        fused = fusion(tf_logits, emb[uid : uid + 1])
                        risk = float(fused.item())
                        tf_risk = float(tf_p.item())
                        # Lightweight online adjustment for unseen entities.
                        user_hist = df[df["user_id"] == user_id]
                        unseen_device = bool(device not in set(user_hist["device_id"].astype(str).tolist()))
                        unseen_merchant = bool(merchant not in set(user_hist["merchant_id"].astype(str).tolist()))
                        if unseen_device:
                            risk += 0.08
                        if unseen_merchant:
                            risk += 0.06
                        risk = float(np.clip(risk, 0.0, 1.0))
                        mode = "Trained Transformer + GNN + Fusion"
                        ra = ResilienceAnalyzer(num_scenarios=12)
                        res = ra.run(
                            lambda xx, mm_: tf.forward_logit(xx, mm_).detach().cpu().numpy(),
                            x,
                            m,
                            plot_path=None,
                        )
                        resilience_val = float(res["mean_resilience"])
                        resilience_var_mean = float(np.mean(res["prediction_variance"]))
                else:
                    st.warning("No sequence for this user in CSV — using heuristic risk.")
            else:
                st.warning("User id not in graph — using heuristic risk.")

        st.metric("Risk score (0–1)", f"{risk:.8f}", help=mode)
        if tf_risk is not None:
            st.metric("Transformer-only risk (0–1)", f"{tf_risk:.8f}")
        if resilience_val is not None:
            st.metric("Resilience (mean factor)", f"{resilience_val:.10f}")
        if resilience_var_mean is not None:
            st.caption(f"Mean variance (logit space): {resilience_var_mean:.6e}")
        st.subheader("Explanation")
        st.write(explanation)
        st.caption(f"RAG latency: {lat:.2f}s | budget: {cfg['rag']['latency_budget_seconds']}s")
        st.json({k: meta[k] for k in ("latency_seconds", "scores") if k in meta})


if __name__ == "__main__":
    main()
