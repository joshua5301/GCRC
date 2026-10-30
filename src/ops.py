import os

import torch


def make_deterministic():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Adjacency:
    def __init__(self, row, col, value, nodes):
        order = torch.argsort(row * nodes + col)
        self.row, self.col, self.value, self.nodes = row[order], col[order], value[order], nodes

    def t(self):
        return Adjacency(self.col, self.row, self.value, self.nodes)


def spmm(adj, x, chunk=1 << 21):
    out = x.new_zeros(adj.nodes, x.shape[1])
    for start in range(0, len(adj.row), chunk):
        edges = slice(start, start + chunk)
        out.index_put_((adj.row[edges],), adj.value[edges, None].to(x) * x[adj.col[edges]], accumulate=True)
    return out


class Propagate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, adj):
        ctx.adj = adj
        return spmm(adj, x)

    @staticmethod
    def backward(ctx, gradient):
        return spmm(ctx.adj.t(), gradient), None
