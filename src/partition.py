import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def rms_normalize(x):
    x = x - x.mean(0)
    return x / x.square().sum(1).mean().sqrt().clamp(min=1e-30)


def cell_means(x, assign, cluster_num):
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(x.dtype)
    return x.new_zeros(cluster_num, x.shape[1]).index_add_(0, assign, x) / counts[:, None]


def sse(block):
    return ((block - block.mean(0)) ** 2).sum()


def var_part(z, cluster_num):
    groups, costs = [np.arange(len(z))], [sse(z)]
    while len(groups) < cluster_num:
        i = int(np.argmax(costs))
        ids = groups[i]
        centered = z[ids] - z[ids].mean(0)
        projection = centered[:, (centered ** 2).sum(0).argmax()]
        left = projection <= 0
        if left.all() or not left.any():
            left = np.zeros(len(ids), dtype=bool)
            left[np.argsort(projection, kind='stable')[:len(ids) // 2]] = True
        groups[i] = ids[left]
        groups.append(ids[~left])
        costs[i] = sse(z[groups[i]])
        costs.append(sse(z[groups[-1]]))
    return np.stack([z[ids].mean(0) for ids in groups])


def initial_centers(z, cluster_num, init):
    if init == 'pca':
        return PCA(n_components=cluster_num).fit(z).components_
    if init == 'var-part':
        return var_part(z, cluster_num)
    return init


def partition(H, y_pred, cluster_num, lam, init='k-means++', seed=0):
    H, y_pred = H.double(), y_pred.double()
    x = rms_normalize(H / H.norm(dim=1, keepdim=True).clamp(min=1e-12))
    z = torch.cat([x, lam ** 0.5 * (y_pred - y_pred.mean(0))], 1).cpu().numpy()
    kmeans = KMeans(cluster_num, init=initial_centers(z, cluster_num, init), n_init=1, random_state=seed).fit(z)
    assign = torch.from_numpy(kmeans.labels_).long().to(H.device)
    return cell_means(H, assign, cluster_num), cell_means(y_pred, assign, cluster_num)
