import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.ops import Propagate


class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, dropout=0.5, layers=2):
        super().__init__()
        sizes = [nin] + [nhid] * (layers - 1) + [nout]
        self.layers = torch.nn.ModuleList([GCNConv(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
        self.dropout = dropout

    def forward(self, x, adj=None):
        for i, layer in enumerate(self.layers):
            x = layer.lin(x)
            if adj is not None:
                x = Propagate.apply(x, adj)
            x = x + layer.bias
            if i < len(self.layers) - 1:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


@torch.no_grad()
def evaluate(model, graph, mask=None):
    model.eval()
    log_p, y = model(graph.x, graph.adj), graph.y
    if mask is not None:
        log_p, y = log_p[mask], y[mask]
    return 100 * (log_p.argmax(1) == y).double().mean().item(), F.nll_loss(log_p, y).item()


def accuracy(model, graph, mask=None):
    return evaluate(model, graph, mask)[0]
