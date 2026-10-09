import math

import torch

from src.partition import cell_means, joint_features


# ---------- soft assignment P = softmax(U V^T / (tau sqrt(rank))), initialized from the stage-1 distances ----------

@torch.no_grad()
def distance_svd(inputs, centers, rank):
    # -||x_i - c_j||^2 = 2 x_i.c_j - ||c_j||^2 - ||x_i||^2; the last term does not change the row softmax
    left = torch.cat((2 * inputs, torch.ones_like(inputs[:, :1])), 1)
    right = torch.cat((centers, -centers.square().sum(1, keepdim=True)), 1)
    right = right - right.mean(0)
    ql, rl = torch.linalg.qr(left)
    qr, rr = torch.linalg.qr(right)
    a, s, bt = torch.linalg.svd(rl @ rr.T, full_matrices=False)
    scale = s[:rank].sqrt() * rank ** 0.25
    return (ql @ a[:, :rank]) * scale, (qr @ bt[:rank].T) * scale


class SoftMoments(torch.autograd.Function):
    # moments[j] = (1/N) sum_i P_ij material_i, computed in chunks so P (N x M) is never stored
    @staticmethod
    def forward(ctx, u, v, material, chunk_size):
        ctx.save_for_backward(u, v, material)
        ctx.chunk_size = chunk_size
        result = material.new_zeros(len(v), material.shape[1])
        for start in range(0, len(u), chunk_size):
            p = (u[start:start + chunk_size] @ v.T).softmax(1)
            result += p.T @ material[start:start + chunk_size] / len(u)
        return result

    @staticmethod
    def backward(ctx, gradient):
        u, v, material = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            p = (u[start:end] @ v.T).softmax(1)
            direction = material[start:end] @ gradient.T / len(u)
            block = p * (direction - (p * direction).sum(1, keepdim=True))
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        return du, dv, None, None


# ---------- inner problem: linear softmax head on the condensed nodes (uniform weights, ridge penalty) ----------

def augment(x):
    return torch.cat((x, torch.ones_like(x[:, :1])), 1)


def decode(moments, dim):
    mass = moments[:, :1]
    return moments[:, 1:dim + 1] / mass, moments[:, dim + 1:] / mass


def head_loss(x, labels, theta, penalty):
    return -(labels * (x @ theta.T).log_softmax(1)).sum(1).mean() + penalty * theta.square().sum() / 2


def head_gradient(x, labels, theta, penalty):
    error = labels.sum(1, keepdim=True) * (x @ theta.T).softmax(1) - labels
    return error.T @ x / len(x) + penalty * theta


def fit_head(x, labels, theta, penalty, tol=1e-5, newton_steps=8, max_iter=2000):
    # warm-started Newton-CG with Armijo backtracking (one or two steps once the assignment moves slowly),
    # L-BFGS only when Newton does not reach the tolerance
    theta = theta.detach().clone()
    with torch.no_grad():
        for _ in range(newton_steps):
            gradient = head_gradient(x, labels, theta, penalty)
            grad_max = float(gradient.abs().max())
            if grad_max <= tol:
                return theta, grad_max
            direction = solve_hessian(x, theta, penalty, gradient, rtol=min(0.1, max(1e-4, grad_max ** 0.5)), max_iter=512)
            decrease = float((gradient * direction).sum())
            if not decrease > 0:
                break
            loss = head_loss(x, labels, theta, penalty)
            for k in range(24):
                proposal = theta - 0.5 ** k * direction
                if head_loss(x, labels, proposal, penalty) <= loss - 1e-4 * 0.5 ** k * decrease:
                    theta = proposal
                    break
            else:
                break
    theta.requires_grad_()
    optimizer = torch.optim.LBFGS([theta], max_iter=max_iter, tolerance_grad=tol, tolerance_change=0, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = head_loss(x, labels, theta, penalty)
        loss.backward()
        return loss

    optimizer.step(closure)
    theta = theta.detach()
    return theta, float(head_gradient(x, labels, theta, penalty).abs().max())


@torch.no_grad()
def solve_hessian(x, theta, penalty, rhs, initial=None, rtol=1e-3, max_iter=2048):
    # preconditioned CG for H s = rhs, H = Hessian of head_loss at theta
    p = (x @ theta.T).softmax(1)
    diagonal = (p * (1 - p)).T @ x.square() / len(x) + penalty

    def multiply(s):
        direction = x @ s.T
        return (p * (direction - (p * direction).sum(1, keepdim=True))).T @ x / len(x) + penalty * s

    s = torch.zeros_like(rhs) if initial is None else initial.clone()
    residual = rhs - multiply(s)
    z = residual / diagonal
    d, rz = z.clone(), (residual * z).sum()
    target = rtol * rhs.norm()
    for _ in range(max_iter):
        if residual.norm() <= target:
            break
        image = multiply(d)
        alpha = rz / (d * image).sum()
        s += alpha * d
        residual -= alpha * image
        z = residual / diagonal
        rz, rz_old = (residual * z).sum(), rz
        d = z + rz / rz_old * d
    return s


# ---------- outer problem: CE of the head on every original node against the teacher ----------

@torch.no_grad()
def outer_loss(x, q, theta, chunk_size):
    value, gradient = 0.0, torch.zeros_like(theta)
    for start in range(0, len(x), chunk_size):
        xb, qb = x[start:start + chunk_size], q[start:start + chunk_size]
        log_p = (xb @ theta.T).log_softmax(1)
        value -= float((qb * log_p).sum()) / len(x)
        gradient += (log_p.exp() - qb).T @ xb / len(x)
    return value, gradient


def bilevel(H, y_pred, assign, cluster_num, lam, rank=16, tau=0.1, penalty=1e-5, steps=500, lr=0.01,
            chunk_size=4096, log_every=50):
    """Stage 2: start from the stage-1 hard partition and learn a soft assignment that minimizes the original risk
    of a linear head trained on the condensed nodes (implicit differentiation through the inner optimum).
    Returns the condensed features (weighted means of raw H) and labels at the step with the lowest outer loss."""
    H, q = H.double(), y_pred.double()
    offset = H.mean(0)
    scale = (H - offset).square().sum(1).mean().sqrt()
    z = (H - offset) / scale                      # feature space of the linear head
    x_outer = augment(z)
    material = torch.cat((torch.ones_like(z[:, :1]), z, q), 1)
    inputs = joint_features(H, q, lam)            # stage-1 clustering space
    u, v = distance_svd(inputs, cell_means(inputs, assign, cluster_num), rank)
    u, v = u.requires_grad_(), v.requires_grad_()
    optimizer = torch.optim.Adam([u, v], lr=lr, eps=1e-12)
    theta, vector, best, history = torch.zeros(q.shape[1], z.shape[1] + 1, dtype=z.dtype, device=z.device), None, None, []
    for step in range(steps + 1):
        moments = SoftMoments.apply(u / (tau * math.sqrt(rank)), v, material, chunk_size)
        centers, labels = decode(moments.detach(), z.shape[1])
        theta, inner_grad = fit_head(augment(centers), labels, theta, penalty)
        value, outer_gradient = outer_loss(x_outer, q, theta, chunk_size)
        if step == 0:
            loss_scale = value
        history.append(dict(step=step, outer_loss=value, inner_grad=inner_grad, min_mass=float(moments[:, 0].detach().min()) * cluster_num))
        if best is None or value < best[0]:
            best = (value, step, centers, labels)
        if step % log_every == 0 or step == steps:
            print(f'stage2 step {step:4d}: outer CE {value:.5f}  inner grad {inner_grad:.1e}  min mass {history[-1]["min_mass"]:.3f}')
        if step == steps:
            break
        # d outer / d moments = -d/d moments [ grad_theta inner(moments, theta) . H^{-1} grad_theta outer ]
        vector = solve_hessian(augment(centers), theta, penalty, outer_gradient, vector)
        variable = moments.detach().requires_grad_()
        c, l = decode(variable, z.shape[1])
        (direction,) = torch.autograd.grad(-(head_gradient(augment(c), l, theta, penalty) * vector).sum(), variable)
        optimizer.zero_grad()
        moments.backward(direction / loss_scale)
        optimizer.step()
    value, step, centers, labels = best
    print(f'stage2 best step {step}: outer CE {value:.5f} (step 0: {history[0]["outer_loss"]:.5f})')
    return centers * scale + offset, labels, history
