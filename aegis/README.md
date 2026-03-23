# Aegis: Dual-Engine AI for Resilient Risk Detection

Production-oriented research codebase that combines **temporal Transformers**, **Graph Neural Networks (PyTorch Geometric)**, and **RAG** (FAISS + lightweight generation) for behavioral risk scoring, explanations, and robustness analysis.

## Architecture (ASCII)

```
                    ┌─────────────────────┐
                    │ Synthetic + tabular   │
                    │ transactions        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────────┐           ┌──────────────────┐
   │ Transformer          │           │ GNN (GraphSAGE/  │
   │ (sequence: amount,   │           │  GAT) user–device│
   │  time, location)     │           │  –merchant graph │
   └──────────┬───────────┘           └────────┬─────────┘
              │                                 │
              └────────────┬────────────────────┘
                           ▼
                 ┌──────────────────┐
                 │ Fusion MLP       │
                 │ → final risk     │
                 └────────┬─────────┘
                          │
     ┌────────────────────┼────────────────────────┐
     ▼                    ▼                        ▼
┌─────────┐        ┌─────────────┐        ┌────────────────┐
│ AUC eval│        │ RAG explain │        │ Resilience test│
│ (TF/GNN │        │ FAISS + HF  │        │ variance-based │
│ /Fused) │        │ / OpenAI    │        │ + PNG plot     │
└─────────┘        └─────────────┘        └────────────────┘
```

## Setup

```bash
cd aegis
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

**PyTorch + CUDA (optional):** install the wheel matching your CUDA from [pytorch.org](https://pytorch.org) before `torch-geometric`.

## Run end-to-end

```bash
cd aegis
python main.py
```

This will:

1. Generate synthetic transactions + risk narratives under `outputs/`.
2. Train the Transformer, GNN, and fusion head.
3. Print **AUC-ROC** for Transformer-only, GNN-only (user-level), and fused scores.
4. Run **RAG** on a sample transaction (downloads small HF models on first run).
5. Run **resilience** stress tests and save `outputs/resilience_plot.png`.
6. Save `outputs/checkpoint.pt` for the optional Streamlit app.

## Run pieces separately

| Task | Command |
|------|---------|
| Full pipeline | `python main.py` |
| Unit tests | From repo root (`Aegis/`): `python -m pytest tests -q` |
| Dashboard (bonus) | `streamlit run streamlit_app.py` (from `aegis/`) |

## Configuration

Edit `config.yaml` for seeds, model sizes, RAG models, latency budget (default 13s), and output paths.

Set `OPENAI_API_KEY` to use OpenAI for explanations instead of local `distilgpt2`.

**Latency:** The first RAG call may download `sentence-transformers` and `distilgpt2` weights and can exceed the default 13s budget; subsequent runs are much faster. Greedy decoding and `max_new_tokens` in `config.yaml` keep typical CPU runs under budget after caches are warm.

## Example outputs

**AUC (illustrative — your numbers will vary):**

```
Transformer AUC: 0.78–0.92
GNN (user-level) AUC: 0.72–0.88
Fused AUC: 0.80–0.93
```

**Sample RAG explanation (local HF):**

> The model may describe unusual amount, device, or merchant patterns aligned with retrieved fraud narratives.

**Resilience:**

- Mean **resilience factor** printed to console (`1 / (1 + variance)` over perturbations).
- Plot: `outputs/resilience_plot.png` — left: mean prediction vs stress scenario; right: per-sample resilience.

## Project layout

```
aegis/
├── models/           # Transformer + GNN
├── fusion/           # Fusion MLP
├── rag/              # FAISS retriever + generator + pipeline
├── resilience/       # Stress tests + plot
├── data/             # Synthetic data + preprocessing
├── evaluation/       # Metrics
├── utils/            # Logging + YAML
├── config.yaml
├── main.py
├── requirements.txt
└── streamlit_app.py  # Optional UI
```

## License

Use and modify for research and internal systems per your organization’s policy.
