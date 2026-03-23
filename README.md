# Aegis
Aegis: Dual-Engine AI for Resilient Risk Detection
End-to-end research prototype for adaptive transaction risk scoring & explainability

Goal
Detect anomalous/ fraud-like behavior in transaction streams
Produce interpretable risk insights (RAG narrative)
Measure model resilience under worst-case perturbations
1) Core idea
Aegis builds a hybrid risk detection engine using:
Sequence modeling (temporal Transformer)
Graph representation learning (GNN on user-device-merchant graph)
Fusion head to combine strengths
Retrieval-Augmented Generation (RAG) for human-readable explanation
Variance-based resilience testing on model scores

2) Data layer (data)
Synthetic transaction simulation (synthetic_generator.py)
user histories, timestamps, amounts, categories, geo
plant risk events via known “fraud narrative” rules
Preprocessing (preprocess.py)
split train/val/test by user
compute embedding-ready features, chronological sequences
Output:
sample CSV + risk narrative JSON
graph edges for GNN
sequence datasets for Transformer

3) Modeling layer
Transformer path (transformer.py)
per-user transaction sequences (time+amount+text features)
temporal attention and classification head
outputs per-transaction risk score
GNN path (gnn.py)
constructs heterogeneous graph: user ↔ device, user ↔ merchant
uses GraphSAGE / GAT
user-level embedding + risk score
can capture cross-entity propagation patterns
Fusion path (fusion_model.py)
takes Transformer and GNN score vectors
MLP fusion to final risk output
learns weighted interaction + correction

4) RAG explanation (rag)
Retriever (retriever.py)
FAISS + sentence-transformers
semantic search over risk_narratives.json
Generator (generator.py)
transformers text-generation model (distilgpt2 or OpenAI)
condition on retrieved narratives + transaction context
Pipeline (rag_pipeline.py)
end-to-end narrative for a selected sample
measures latency vs config threshold

5) Resilience testing (resilience_test.py)
Stress variants:
amount inflation/deflation
time shifts
merchant/device swaps
For each sample, compute:
score variance + conservative resilience factor 1/(1+var)
Visual results:
per-sample robustness
aggregated stability plot

6) Evaluation (metrics.py)
AUC-ROC for:
Transformer-only
GNN-only
Fused model
Other metrics: precision, recall, aggregated risk calibration
In full pipeline (main.py):
print comparative AUC with logs

7) Execution semantics (main.py)
Generate and save dataset (+ narrative library).
Fit transformer model.
Build graph + fit GNN.
Train fusion module.
Evaluate with held-out users.
RAG one sample + report latency.
Run resilience evaluations and save plot.
Store checkpoint for UI/re-use.

8) Produced artifacts
outputs/sample_transactions.csv
outputs/risk_narratives.json
outputs/checkpoint.pt
outputs/resilience_plot.png
console metrics + explanation text
