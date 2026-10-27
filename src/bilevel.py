import math

import torch

from src.clustering import rms_normalize


def soft_assignment(u, v, assign, own_logit):
    logits = u @ v.T / math.sqrt(u.shape[1])
    logits[torch.arange(len(u), device=u.device), assign] += own_logit
    return logits.softmax(1)


class SoftMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, assign, own_logit, material, chunk):
        ctx.save_for_backward(u, v, assign, material)
        ctx.own_logit, ctx.chunk = own_logit, chunk
        moments = material.new_zeros(len(v), material.shape[1])
        for start in range(0, len(u), chunk):
            end = start + chunk
            moments += soft_assignment(u[start:end], v, assign[start:end], own_logit).T @ material[start:end]
        return moments / len(u)

    @staticmethod
    def backward(ctx, gradient):
        u, v, assign, material = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        for start in range(0, len(u), ctx.chunk):
            end = start + ctx.chunk
            p = soft_assignment(u[start:end], v, assign[start:end], ctx.own_logit)
            direction = material[start:end] @ gradient.T
            block = p * (direction - (p * direction).sum(1, keepdim=True)) / (len(u) * math.sqrt(u.shape[1]))
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        return du, dv, None, None, None, None


def decode(moments, dim):
    mass = moments[:, :1]
    return moments[:, 1:dim + 1] / mass, moments[:, dim + 1:] / mass


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


def bilevel(h, q, assign, cells, penalty, steps, rank=16, lr=0.1, own=0.99, seed=0, chunk=4096):
    z, offset, scale = rms_normalize(h)
    material, nodes = torch.cat((z.new_ones(len(z), 1), z, q), 1), augment(z)
    own_logit = math.log(own / (1 - own) * (cells - 1))
    generator = torch.Generator(device=z.device).manual_seed(seed)
    u = z.new_zeros(len(z), rank).requires_grad_()
    v = torch.randn(cells, rank, generator=generator, device=z.device, dtype=z.dtype).requires_grad_()
    optimizer = torch.optim.Adam([u, v], lr=lr, eps=1e-12)
    theta, vector, scale0 = z.new_zeros(q.shape[1], z.shape[1] + 1), None, None
    for step in range(steps + 1):
        moments = SoftMoments.apply(u, v, assign, own_logit, material, chunk)
        centers, labels = decode(moments.detach(), z.shape[1])
        if step == steps:
            return centers * scale + offset, labels
        x = augment(centers)
        theta = fit_head(x, labels, theta, penalty)
        value, outer_gradient = outer_loss(nodes, q, theta)
        scale0 = scale0 or value
        multiply, diagonal = head_hessian(x, theta, penalty)
        vector = conjugate_gradient(multiply, outer_gradient, diagonal, vector)
        variable = moments.detach().requires_grad_()
        c, l = decode(variable, z.shape[1])
        direction, = torch.autograd.grad(-(head_gradient(augment(c), l, theta, penalty) * vector).sum(), variable)
        optimizer.zero_grad()
        moments.backward(direction / scale0)
        optimizer.step()
