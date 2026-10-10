import torch
from torch.func import grad, jvp

from src.partition import cell_means, joint_features


@torch.no_grad()
def svd_factors(inputs, centers, rank):
    # rank-r factorization of the stage-1 logits -||x_i - c_j||^2 (the row constant -||x_i||^2 is dropped)
    left = torch.cat((2 * inputs, torch.ones_like(inputs[:, :1])), 1)
    right = torch.cat((centers, -centers.square().sum(1, keepdim=True)), 1)
    right = right - right.mean(0)
    ql, rl = torch.linalg.qr(left)
    qr, rr = torch.linalg.qr(right)
    a, s, bt = torch.linalg.svd(rl @ rr.T, full_matrices=False)
    scale = s[:rank].sqrt() * rank ** 0.25  # logits = U V^T / sqrt(rank), as in GRIP
    return (ql @ a[:, :rank]) * scale, (qr @ bt[:rank].T) * scale


def augment(x):
    return torch.cat((x, torch.ones_like(x[:, :1])), 1)


def ce(theta, x, labels):
    return -(labels * (augment(x) @ theta.T).log_softmax(1)).sum(1).mean()


def inner_loss(theta, x, labels, penalty):
    return ce(theta, x, labels) + penalty * theta.square().sum() / 2


@torch.no_grad()
def cg(matvec, b, x=None, rtol=1e-3, max_iter=2048):
    x = torch.zeros_like(b) if x is None else x.clone()
    r = b - matvec(x)
    p, rr = r.clone(), (r * r).sum()
    for _ in range(max_iter):
        if rr.sqrt() <= rtol * b.norm():
            break
        Ap = matvec(p)
        alpha = rr / (p * Ap).sum()
        x, r = x + alpha * p, r - alpha * Ap
        rr, rr_old = (r * r).sum(), rr
        p = r + rr / rr_old * p
    return x


def fit_head(theta, x, labels, penalty, tol=1e-5, max_steps=50):
    # warm-started Newton-CG with Armijo backtracking
    gradient = grad(inner_loss)
    for _ in range(max_steps):
        g = gradient(theta, x, labels, penalty)
        if g.abs().max() <= tol:
            break
        hvp = lambda v: jvp(lambda t: gradient(t, x, labels, penalty), (theta,), (v,))[1]
        direction = cg(hvp, g, rtol=min(0.1, max(1e-4, float(g.abs().max()) ** 0.5)), max_iter=512)
        loss, decrease = inner_loss(theta, x, labels, penalty), (g * direction).sum()
        step = 1.0
        while inner_loss(theta - step * direction, x, labels, penalty) > loss - 1e-4 * step * decrease and step > 1e-7:
            step /= 2
        theta = theta - step * direction
    return theta.detach(), float(g.abs().max())


def adam(grad_fn, x, steps, lr):
    # a fixed number of Adam steps from x (RCIG-style approximate solve; fresh moment estimates each call)
    x = x.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        x.grad = grad_fn(x.detach())
        optimizer.step()
    return x.detach()


def bilevel(H, y_pred, assign, cluster_num, lam, rank=16, tau=0.03, penalty=1e-5, steps=500, lr=0.01, solver='exact',
            inner_steps=20, inner_lr=0.01, hinv_steps=20, hinv_lr=0.1, checkpoint_every=None, log_every=50):
    """Stage 2: soft assignment P = softmax(U V^T / (tau sqrt(rank))), initialized from the stage-1 partition, trained so that a linear
    head fitted on the condensed nodes minimizes the teacher CE on all original nodes (implicit differentiation).
    solver='exact': inner head by Newton-CG to convergence, H^{-1} g by CG (warm-started).
    solver='adam': RCIG-style, inner_steps Adam steps warm-started from the previous head, H^{-1} g by hinv_steps Adam steps
    on 1/2 s^T H s - g^T s from zero.
    Returns the condensed features (P-weighted means of raw H) and labels at the step with the lowest outer CE, and a list of
    (step, features, labels) every checkpoint_every steps and at the last step (empty if checkpoint_every is None)."""
    H, q = H.double(), y_pred.double()
    offset, scale = H.mean(0), (H - H.mean(0)).square().sum(1).mean().sqrt()
    z = (H - offset) / scale                        # feature space of the linear head
    inputs = joint_features(H, q, lam)              # stage-1 clustering space
    u, v = svd_factors(inputs, cell_means(inputs, assign, cluster_num), rank)
    u, v = u.requires_grad_(), v.requires_grad_()
    optimizer = torch.optim.Adam([u, v], lr=lr, eps=1e-12)
    outer_grad = grad(ce)
    theta, s, best, checkpoints = z.new_zeros(q.shape[1], z.shape[1] + 1), None, None, []
    for step in range(steps + 1):
        p = (u @ v.T / (tau * rank ** 0.5)).softmax(1)
        mass = p.sum(0)[:, None]
        x, labels = p.T @ z / mass, p.T @ q / mass  # condensed nodes, differentiable in (u, v)
        g_inner = grad(inner_loss)
        if solver == 'exact':
            theta, inner_grad = fit_head(theta, x.detach(), labels.detach(), penalty)
        else:
            theta = adam(lambda t: g_inner(t, x.detach(), labels.detach(), penalty), theta, inner_steps, inner_lr)
            inner_grad = float(g_inner(theta, x.detach(), labels.detach(), penalty).abs().max())
        with torch.no_grad():
            value = float(ce(theta, z, q))
        if step == 0:
            loss_scale = value
        if best is None or value < best[0]:
            best = (value, step, x.detach(), labels.detach())
        if checkpoint_every and (step % checkpoint_every == 0 or step == steps):
            checkpoints.append((step, x.detach() * scale + offset, labels.detach()))
        if step % log_every == 0 or step == steps:
            print(f'stage2 step {step:4d}: outer CE {value:.5f}  inner grad {inner_grad:.1e}  min mass {float(mass.detach().min()) * cluster_num / len(z):.3f}')
        if step == steps:
            break
        # d outer/d(x, labels) = -d/d(x, labels) [ grad_theta inner . H^{-1} grad_theta outer ]
        hvp = lambda w: jvp(lambda t: g_inner(t, x.detach(), labels.detach(), penalty), (theta,), (w,))[1]
        g_outer = outer_grad(theta, z, q)
        if solver == 'exact':
            s = cg(hvp, g_outer, s)
        else:
            s = adam(lambda w: hvp(w) - g_outer, torch.zeros_like(g_outer), hinv_steps, hinv_lr)
        optimizer.zero_grad()
        (-(g_inner(theta, x, labels, penalty) * s).sum() / loss_scale).backward()
        optimizer.step()
    value, step, x, labels = best
    print(f'stage2 best step {step}: outer CE {value:.5f} (step 0: {loss_scale:.5f})')
    return x * scale + offset, labels, checkpoints
