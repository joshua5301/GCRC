import torch
from torch.func import grad, grad_and_value, jvp

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


def minimize(value_and_grad, x, tol=1e-5, max_iter=1000):
    # torch L-BFGS (strong Wolfe line search) from x until the max-abs gradient is <= tol
    x = x.detach().clone().requires_grad_()
    optimizer = torch.optim.LBFGS([x], lr=1, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-14,
                                  history_size=20, line_search_fn='strong_wolfe')
    def closure():
        value, x.grad = value_and_grad(x.detach())
        return value
    optimizer.step(closure)
    return x.detach()


def bilevel(H, y_pred, assign, cluster_num, lam, rank=16, tau=0.03, penalty=1e-5, steps=500, lr=0.01,
            checkpoint_every=None, callback=None, log_every=50):
    """Stage 2: soft assignment P = softmax(U V^T / (tau sqrt(rank))), initialized from the stage-1 partition, trained so that a linear
    head fitted on the condensed nodes minimizes the teacher CE on all original nodes (implicit differentiation).
    The inner head and H^{-1} g (as the minimizer of 1/2 s^T H s - g^T s) are both solved to convergence by L-BFGS, warm-started.
    Every checkpoint_every steps (and at the last step) calls callback(step, features, labels); a True return stops early.
    Returns the condensed features (P-weighted means of raw H) and labels at the step with the lowest outer CE."""
    H, q = H.double(), y_pred.double()
    offset, scale = H.mean(0), (H - H.mean(0)).square().sum(1).mean().sqrt()
    z = (H - offset) / scale                        # feature space of the linear head
    inputs = joint_features(H, q, lam)              # stage-1 clustering space
    u, v = svd_factors(inputs, cell_means(inputs, assign, cluster_num), rank)
    u, v = u.requires_grad_(), v.requires_grad_()
    optimizer = torch.optim.Adam([u, v], lr=lr, eps=1e-12)
    g_inner = grad(inner_loss)
    theta, s, best = z.new_zeros(q.shape[1], z.shape[1] + 1), None, None
    for step in range(steps + 1):
        p = (u @ v.T / (tau * rank ** 0.5)).softmax(1)
        mass = p.sum(0)[:, None]
        x, labels = p.T @ z / mass, p.T @ q / mass  # condensed nodes, differentiable in (u, v)
        xd, ld = x.detach(), labels.detach()
        theta = minimize(lambda t: grad_and_value(inner_loss)(t, xd, ld, penalty)[::-1], theta)
        with torch.no_grad():
            value = float(ce(theta, z, q))
        if step == 0:
            loss_scale = value
        if best is None or value < best[0]:
            best = (value, step, xd, ld)
        if step % log_every == 0 or step == steps:
            print(f'stage2 step {step:4d}: outer CE {value:.5f}  inner grad {float(g_inner(theta, xd, ld, penalty).abs().max()):.1e}  '
                  f'min mass {float(mass.detach().min()) * cluster_num / len(z):.3f}')
        if callback and (step % checkpoint_every == 0 or step == steps) and callback(step, xd * scale + offset, ld):
            print(f'stage2 early stop at step {step}')
            break
        if step == steps:
            break
        # d outer/d(x, labels) = -d/d(x, labels) [ grad_theta inner . H^{-1} grad_theta outer ]
        hvp = lambda w: jvp(lambda t: g_inner(t, xd, ld, penalty), (theta,), (w,))[1]
        g_outer = grad(ce)(theta, z, q)
        def quadratic(w):
            hw = hvp(w)
            return (w * hw).sum() / 2 - (g_outer * w).sum(), hw - g_outer
        s = minimize(quadratic, torch.zeros_like(g_outer) if s is None else s, tol=1e-3 * float(g_outer.abs().max()))
        optimizer.zero_grad()
        (-(g_inner(theta, x, labels, penalty) * s).sum() / loss_scale).backward()
        optimizer.step()
    value, step, x, labels = best
    print(f'stage2 best step {step}: outer CE {value:.5f} (step 0: {loss_scale:.5f})')
    return x * scale + offset, labels
