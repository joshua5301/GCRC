import torch


def rms_normalize(x):
    x = x - x.mean(0)
    return x / x.square().sum(1).mean().sqrt().clamp(min=1e-30)


def cell_means(x, assign, cluster_num):
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(x.dtype)
    return x.new_zeros(cluster_num, x.shape[1]).index_add_(0, assign, x) / counts[:, None]


def sse(block):
    return (block - block.mean(0)).square().sum()


def var_part(z, cluster_num):
    groups, costs = [torch.arange(len(z), device=z.device)], [sse(z)]
    while len(groups) < cluster_num:
        i = int(torch.stack(costs).argmax())
        ids = groups[i]
        centered = z[ids] - z[ids].mean(0)
        projection = centered[:, centered.square().sum(0).argmax()]
        left = projection <= 0
        if left.all() or not left.any():
            left = torch.zeros(len(ids), dtype=torch.bool, device=z.device)
            left[torch.argsort(projection, stable=True)[:len(ids) // 2]] = True
        groups[i] = ids[left]
        groups.append(ids[~left])
        costs[i] = sse(z[groups[i]])
        costs.append(sse(z[groups[-1]]))
    assign = torch.empty(len(z), dtype=torch.long, device=z.device)
    for c, ids in enumerate(groups):
        assign[ids] = c
    return assign


def lloyd(z, assign, cluster_num, max_iter=300):
    sq_norm = z.square().sum(1, keepdim=True)
    for _ in range(max_iter):
        centers = cell_means(z, assign, cluster_num)
        dist, new_assign = (sq_norm - 2 * z @ centers.T + centers.square().sum(1)).min(1)
        empty = (torch.bincount(new_assign, minlength=cluster_num) == 0).nonzero().flatten()
        if len(empty):  # move empty cells to the points farthest from their centers, as sklearn does
            new_assign[dist.topk(len(empty)).indices] = empty
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
    return assign


def partition(H, y_pred, cluster_num, lam):
    H, y_pred = H.double(), y_pred.double()
    x = rms_normalize(H / H.norm(dim=1, keepdim=True).clamp(min=1e-12))
    z = torch.cat([x, lam ** 0.5 * (y_pred - y_pred.mean(0))], 1)
    assign = lloyd(z, var_part(z, cluster_num), cluster_num)
    return cell_means(H, assign, cluster_num), cell_means(y_pred, assign, cluster_num)
