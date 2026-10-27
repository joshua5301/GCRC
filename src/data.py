import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import subgraph

BUDGET = {
    ('cora', 0.013): 35, ('cora', 0.026): 70, ('cora', 0.052): 140,
    ('citeseer', 0.009): 30, ('citeseer', 0.018): 60, ('citeseer', 0.036): 120,
    ('arxiv', 0.0005): 90, ('arxiv', 0.0025): 454, ('arxiv', 0.005): 909,
    ('flickr', 0.001): 44, ('flickr', 0.005): 223, ('flickr', 0.01): 446,
    ('reddit', 0.0005): 77, ('reddit', 0.001): 153, ('reddit', 0.002): 307,
}


def load_dataset(name, root, device):
    if name in ('cora', 'citeseer'):
        graph = with_adj(Planetoid(root, name, transform=T.NormalizeFeatures())[0].to(device))
        return graph, (graph, graph.val_mask), (graph, graph.test_mask)
    folder = Path(root) / ('ogbn-arxiv' if name == 'arxiv' else name) / 'raw'
    adj = sp.load_npz(folder / 'adj_full.npz')
    if name == 'arxiv':
        adj = adj + adj.T
        adj[adj > 1] = 1
    nodes = adj.shape[0]
    role = json.loads((folder / 'role.json').read_text())
    masks = {}
    for split, key in (('train', 'tr'), ('val', 'va'), ('test', 'te')):
        masks[split] = torch.zeros(nodes, dtype=torch.bool)
        masks[split][role[key]] = True
    labels = np.zeros(nodes, dtype=int)
    for node, label in json.loads((folder / 'class_map.json').read_text()).items():
        labels[int(node)] = label
    features = np.load(folder / 'feats.npy')
    features = StandardScaler().fit(features[role['tr']]).transform(features)
    graph = T.ToUndirected()(Data(x=torch.FloatTensor(features), edge_index=torch.LongTensor(np.array(adj.nonzero())),
                                  y=torch.LongTensor(labels - labels.min()), train_mask=masks['train'],
                                  val_mask=masks['val'], test_mask=masks['test']))
    if name == 'arxiv':
        graph = with_adj(graph.to(device))
        return graph, (graph, graph.val_mask), (graph, graph.test_mask)
    train, val, test = [with_adj(induced(graph, masks[split]).to(device)) for split in ('train', 'val', 'test')]
    train.train_mask = torch.ones(train.num_nodes, dtype=torch.bool, device=device)
    return train, (val, None), (test, None)


def with_adj(graph):
    edge_index, weight = gcn_norm(graph.edge_index, None, graph.num_nodes, dtype=graph.x.dtype)
    graph.adj = torch.sparse_coo_tensor(edge_index.flip(0), weight, (graph.num_nodes, graph.num_nodes)).coalesce().to_sparse_csr()
    return graph


def induced(graph, mask):
    edge_index, _ = subgraph(mask, graph.edge_index, relabel_nodes=True)
    return Data(x=graph.x[mask], y=graph.y[mask], edge_index=edge_index)


def propagate(graph, hops=2):
    nodes = graph.num_nodes
    adj = sp.csr_matrix((np.ones(graph.edge_index.shape[1]), graph.edge_index.cpu().numpy()), shape=(nodes, nodes))
    if adj[0, 0] == 0:
        adj = adj + sp.eye(nodes)
    degree = np.power(np.asarray(adj.sum(1)).flatten(), -0.5)
    degree[np.isinf(degree)] = 0
    adj = (sp.diags(degree) @ adj @ sp.diags(degree)).tocoo().astype(np.float32)
    adj = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((adj.row, adj.col))), torch.FloatTensor(adj.data),
                                  (nodes, nodes)).to(graph.x.device)
    x = graph.x
    for _ in range(hops):
        x = torch.sparse.mm(adj, x)
    return x
