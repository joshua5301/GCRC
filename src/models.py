import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList([GCNConv(nin, nhid), GCNConv(nhid, nout)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.layers[0](x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(self.layers[1](x, edge_index), dim=1)


@torch.no_grad()
def accuracy(model, graph, mask=None):
    model.eval()
    pred, y = model(graph.x, graph.edge_index).argmax(1), graph.y
    if mask is not None:
        pred, y = pred[mask], y[mask]
    return 100 * (pred == y).double().mean().item()
