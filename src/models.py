import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList([GCNConv(nin, nhid), GCNConv(nhid, nout)])
        self.dropout = dropout

    def forward(self, x, adj=None):
        for i, layer in enumerate(self.layers):
            x = layer.lin(x)
            if adj is not None:
                x = torch.sparse.mm(adj, x)
            x = x + layer.bias
            if i == 0:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


@torch.no_grad()
def accuracy(model, graph, mask=None):
    model.eval()
    pred, y = model(graph.x, graph.adj).argmax(1), graph.y
    if mask is not None:
        pred, y = pred[mask], y[mask]
    return 100 * (pred == y).double().mean().item()
