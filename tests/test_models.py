import torch

from models.gnn import RelationalRiskGNN
from models.transformer import TemporalRiskTransformer


def test_transformer_forward():
    m = TemporalRiskTransformer(num_locations=16, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    x = torch.randn(2, 10, 5)
    x[:, :, 4] = torch.randint(0, 16, (2, 10)).float()
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = m(x, mask)
    assert out.shape == (2,)
    assert (out >= 0).all() and (out <= 1).all()


def test_gnn_forward():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    m = RelationalRiskGNN(num_nodes=5, hidden_channels=16, out_channels=8, num_layers=2, model_type="graphsage")
    emb = m(edge_index, return_logits=False)
    assert emb.shape == (5, 8)
    logits = m(edge_index, return_logits=True)
    assert logits.shape == (5, 2)
