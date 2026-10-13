import torch


def rms_normalize(x):
    x = x - x.mean(0)
    return x / x.square().sum(1).mean().sqrt().clamp(min=1e-30)


def cell_means(x, assign, cluster_num):
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(x.dtype)
    return x.new_zeros(cluster_num, x.shape[1]).index_add_(0, assign, x) / counts[:, None]


def normalize_features(H, norm='rms'):
    # row: unit-L2 rows (direction only), col: per-dimension z-score, rms: raw; all then centered and scaled to unit RMS norm
    if norm == 'row':
        H = H / H.norm(dim=1, keepdim=True).clamp(min=1e-12)
    elif norm == 'col':
        H = (H - H.mean(0)) / H.std(0).clamp(min=1e-12)
    elif norm != 'rms':
        raise ValueError(f'unknown feature norm {norm}')
    return rms_normalize(H)


def joint_features(H, y_pred, lam, norm='rms'):
    return torch.cat([normalize_features(H, norm), lam ** 0.5 * (y_pred - y_pred.mean(0))], 1)


def sq_dist(z, centers):
    return (z.square().sum(1, keepdim=True) - 2 * z @ centers.T + centers.square().sum(1)).clamp(min=0)


def cost(dx2, dq2, lam_c, lam):
    # pointwise objective ||dx||^2 + lam_c ||dx|| ||dq|| + lam ||dq||^2, from squared feature/label distances
    return dx2 + lam_c * (dx2 * dq2).sqrt() + lam * dq2


def energy(x, q, assign, cluster_num, lam_c, lam):
    ux, uq = cell_means(x, assign, cluster_num), cell_means(q, assign, cluster_num)
    dx2 = (x - ux[assign]).square().sum(1)
    dq2 = (q - uq[assign]).square().sum(1)
    return float(cost(dx2, dq2, lam_c, lam).sum()), ux, uq


def var_part(x, q, cluster_num, lam_c, lam):
    # deterministic greedy bisection on the objective: split the cluster contributing most to it; candidate cuts
    # are the highest-variance coordinate of x and of q inside that cluster, thresholded at the cluster mean;
    # keep the cut that lowers the objective more
    def cell_cost(ids):
        dx2 = (x[ids] - x[ids].mean(0)).square().sum(1)
        dq2 = (q[ids] - q[ids].mean(0)).square().sum(1)
        return float(cost(dx2, dq2, lam_c, lam).sum())
    groups = [torch.arange(len(x), device=x.device)]
    costs = [cell_cost(groups[0])]
    while len(groups) < cluster_num:
        i = max((j for j in range(len(groups)) if len(groups[j]) > 1), key=lambda j: costs[j])
        ids = groups[i]
        best = None
        for z in (x, q):
            column = z[ids][:, z[ids].var(0).argmax()]
            left = column <= column.mean()
            if left.all() or not left.any():
                continue
            parts = (ids[left], ids[~left])
            part_costs = (cell_cost(parts[0]), cell_cost(parts[1]))
            if best is None or sum(part_costs) < sum(best[1]):
                best = (parts, part_costs)
        if best is None:  # every candidate coordinate is constant in this cluster: cut it in half arbitrarily
            parts = (ids[:len(ids) // 2], ids[len(ids) // 2:])
            best = (parts, (cell_cost(parts[0]), cell_cost(parts[1])))
        (groups[i], right), (costs[i], right_cost) = best
        groups.append(right)
        costs.append(right_cost)
    assign = torch.empty(len(x), dtype=torch.long, device=x.device)
    for c, ids in enumerate(groups):
        assign[ids] = c
    return assign


def lloyd(x, q, assign, cluster_num, lam_c, lam, max_iter=100):
    # exact-cost assignment + mean centers; means do not minimize the cross term, so the objective is
    # guarded: stop as soon as an update would raise it
    e, ux, uq = energy(x, q, assign, cluster_num, lam_c, lam)
    for _ in range(max_iter):
        dist, new_assign = cost(sq_dist(x, ux), sq_dist(q, uq), lam_c, lam).min(1)
        empty = (torch.bincount(new_assign, minlength=cluster_num) == 0).nonzero().flatten()
        if len(empty):  # move empty cells to the points farthest from their centers, as sklearn does
            new_assign[dist.topk(len(empty)).indices] = empty
        if torch.equal(new_assign, assign):
            break
        new_e, new_ux, new_uq = energy(x, q, new_assign, cluster_num, lam_c, lam)
        if new_e > e:
            break
        assign, e, ux, uq = new_assign, new_e, new_ux, new_uq
    return assign


def partition(H, y_pred, cluster_num, lam_c, lam, norm='rms'):
    H, y_pred = H.double(), y_pred.double()
    x, q = normalize_features(H, norm), y_pred - y_pred.mean(0)
    assign = lloyd(x, q, var_part(x, q, cluster_num, lam_c, lam), cluster_num, lam_c, lam)
    return cell_means(H, assign, cluster_num), cell_means(y_pred, assign, cluster_num), assign
