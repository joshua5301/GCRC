import torch


def rms_normalize(h):
    offset = h.mean(0)
    scale = (h - offset).square().sum(1).mean().sqrt().clamp_min(1e-30)
    return (h - offset) / scale, offset, scale


def cell_means(values, assign, cells):
    counts = torch.bincount(assign, minlength=cells).to(values)
    return values.new_zeros(cells, values.shape[1]).index_add_(0, assign, values) / counts[:, None]


def class_budget(labels, cells, classes):
    counts = torch.bincount(labels, minlength=classes).double()
    raw = cells * counts / counts.sum()
    budget = torch.where(counts > 0, raw.floor().clamp_min(1), torch.zeros_like(raw))
    order = torch.argsort(raw - raw.floor(), descending=True)
    i = 0
    while budget.sum() < cells:
        c = order[i % classes]
        if counts[c] > budget[c]:
            budget[c] += 1
        i += 1
    while budget.sum() > cells:
        budget[budget.argmax()] -= 1
    return budget.long().tolist()


def var_part(z, cells):
    def scatter(ids):
        return float((z[ids] - z[ids].mean(0)).square().sum()) if len(ids) > 1 else -1.0

    groups = [torch.arange(len(z), device=z.device)]
    scatters = [scatter(groups[0])]
    while len(groups) < cells:
        i = max(range(len(groups)), key=lambda j: scatters[j])
        ids = groups[i]
        centered = z[ids] - z[ids].mean(0)
        projection = centered[:, centered.square().sum(0).argmax()]
        left = projection <= 0
        if left.all() or not left.any():
            left = torch.zeros_like(left)
            left[torch.argsort(projection, stable=True)[:len(ids) // 2]] = True
        groups[i] = ids[left]
        groups.append(ids[~left])
        scatters[i] = scatter(groups[i])
        scatters.append(scatter(groups[-1]))
    assign = torch.empty(len(z), dtype=torch.long, device=z.device)
    for j, ids in enumerate(groups):
        assign[ids] = j
    return assign


def initial_assignment(z, labels, cells, classes):
    assign = torch.empty(len(z), dtype=torch.long, device=z.device)
    start = 0
    for c, budget in enumerate(class_budget(labels, cells, classes)):
        if budget == 0:
            continue
        ids = torch.where(labels == c)[0]
        assign[ids] = var_part(z[ids], budget) + start
        start += budget
    return assign


def lloyd(z, assign, cells, max_iter=100, block=1024):
    def energy(a):
        return float((z - cell_means(z, a, cells)[a]).square().sum(1).mean())

    current = energy(assign)
    for _ in range(max_iter):
        centers = cell_means(z, assign, cells)
        new, kept = assign.clone(), z.new_empty(len(z))
        for start in range(0, len(z), block):
            end = min(start + block, len(z))
            cost = torch.cdist(z[start:end], centers).square()
            best, target = cost.min(1)
            old = cost.gather(1, assign[start:end, None]).squeeze(1)
            new[start:end] = torch.where(old <= best, assign[start:end], target)
            kept[start:end] = torch.minimum(old, best)
        counts = torch.bincount(new, minlength=cells)
        for empty in torch.where(counts == 0)[0].tolist():
            donor = int(kept.masked_fill(counts[new] <= 1, -torch.inf).argmax())
            counts[new[donor]] -= 1
            new[donor] = empty
            counts[empty] += 1
            kept[donor] = -torch.inf
        if torch.equal(new, assign):
            break
        updated = energy(new)
        if updated >= current:
            break
        assign, current = new, updated
    return assign


def cluster(h, q, labels, cells, lam):
    x, _, _ = rms_normalize(h)
    z = torch.cat((x, lam ** 0.5 * (q - q.mean(0))), 1)
    return lloyd(z, initial_assignment(z, labels, cells, q.shape[1]), cells)
