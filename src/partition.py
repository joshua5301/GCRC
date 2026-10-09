import torch
from sklearn.cluster import KMeans


def rms_normalize(x):
    x = x - x.mean(0)
    return x / x.square().sum(1).mean().sqrt().clamp(min=1e-30)


def cell_means(x, assign, cluster_num):
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(x.dtype)
    return x.new_zeros(cluster_num, x.shape[1]).index_add_(0, assign, x) / counts[:, None]


def partition(H, y_pred, cluster_num, lam, seed=0):
    H, y_pred = H.double(), y_pred.double()
    x = rms_normalize(H / H.norm(dim=1, keepdim=True).clamp(min=1e-12))
    z = torch.cat([x, lam ** 0.5 * (y_pred - y_pred.mean(0))], 1)
    kmeans = KMeans(cluster_num, init='k-means++', n_init=1, random_state=seed).fit(z.cpu().numpy())
    assign = torch.from_numpy(kmeans.labels_).long().to(H.device)
    return cell_means(H, assign, cluster_num), cell_means(y_pred, assign, cluster_num)
