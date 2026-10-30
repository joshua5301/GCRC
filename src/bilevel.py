import math

import torch

from src.clustering import rms_normalize


def augment(x):
    return torch.cat((x, x.new_ones(len(x), 1)), 1)


def head_objective(x, labels, theta, penalty):
    return -(labels * (x @ theta.T).log_softmax(1)).sum(1).mean() + penalty * theta.square().sum() / 2


def head_gradient(x, labels, theta, penalty):
    return ((x @ theta.T).softmax(1) - labels).T @ x / len(x) + penalty * theta


def head_hessian(x, theta, penalty):
    p = (x @ theta.T).softmax(1)

    def multiply(w):
        d = x @ w.T
        return (p * (d - (p * d).sum(1, keepdim=True))).T @ x / len(x) + penalty * w

    return multiply, (p * (1 - p)).T @ x.square() / len(x) + penalty


def conjugate_gradient(multiply, rhs, diagonal, initial=None, rtol=1e-3, max_iter=2048):
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    r = rhs - multiply(x)
    z = r / diagonal
    p, rz, target = z, (r * z).sum(), rtol * rhs.norm()
    for _ in range(max_iter):
        if r.norm() <= target:
            break
        ap = multiply(p)
        alpha = rz / (p * ap).sum()
        x = x + alpha * p
        r = r - alpha * ap
        z = r / diagonal
        rz, previous = (r * z).sum(), rz
        p = z + rz / previous * p
    return x


def fit_head(x, labels, theta, penalty, tol=1e-5, max_iter=200):
    for _ in range(max_iter):
        gradient = head_gradient(x, labels, theta, penalty)
        g_max = float(gradient.abs().max())
        if g_max <= tol:
            break
        multiply, diagonal = head_hessian(x, theta, penalty)
        direction = conjugate_gradient(multiply, gradient, diagonal, rtol=min(0.1, max(1e-4, g_max ** 0.5)))
        objective, decrease = head_objective(x, labels, theta, penalty), (gradient * direction).sum()
        step = 1.0
        while head_objective(x, labels, theta - step * direction, penalty) > objective - 1e-4 * step * decrease and step > 1e-8:
            step /= 2
        theta = theta - step * direction
    return theta


def outer_loss(x, q, theta):
    log_p = (x @ theta.T).log_softmax(1)
    return float(-(q * log_p).sum(1).mean()), (log_p.exp() - q).T @ x / len(x)


def bilevel(h, q, assign, cells, penalty, steps, rank=16, lr=0.1, leak=1e-6, seed=0):
    z, offset, scale = rms_normalize(h)
    nodes, rows = augment(z), torch.arange(len(z), device=z.device)
    own_logit = math.log((1 - (cells - 1) * leak) / leak)
    generator = torch.Generator(device=z.device).manual_seed(seed)
    u = z.new_zeros(len(z), rank).requires_grad_()
    v = torch.randn(cells, rank, generator=generator, device=z.device, dtype=z.dtype).requires_grad_()
    optimizer = torch.optim.Adam([u, v], lr=lr, eps=1e-12)
    theta, vector, scale0 = z.new_zeros(q.shape[1], z.shape[1] + 1), None, None
    for step in range(steps + 1):
        logits = u @ v.T / math.sqrt(rank)
        logits[rows, assign] += own_logit
        p = logits.softmax(1)
        mass = p.sum(0)[:, None]
        centers, labels = p.T @ z / mass, p.T @ q / mass
        if step == steps:
            return centers.detach() * scale + offset, labels.detach()
        x, y = augment(centers.detach()), labels.detach()
        theta = fit_head(x, y, theta, penalty)
        value, outer_gradient = outer_loss(nodes, q, theta)
        scale0 = scale0 or value
        multiply, diagonal = head_hessian(x, theta, penalty)
        vector = conjugate_gradient(multiply, outer_gradient, diagonal, vector)
        optimizer.zero_grad()
        (-(head_gradient(augment(centers), labels, theta, penalty) * vector).sum() / scale0).backward()
        optimizer.step()
