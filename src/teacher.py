import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.models import GCN, accuracy, evaluate


def train_teacher(args, graph, val, test, seed, wd, dropout, layers=2, hidden=None):
    seed_everything(seed)
    model = GCN(graph.num_features, hidden or args.hidden, int(graph.y.max()) + 1, dropout, layers).to(graph.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    best, best_state = None, None
    for epoch in range(1, args.epochs + 1):
        if epoch == args.epochs // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=wd)
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(graph.x, graph.adj)[graph.train_mask], graph.y[graph.train_mask]).backward()
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        acc, ce = evaluate(model, *val)
        score = -acc if args.teacher_ckpt == 'acc' else ce
        if best is None or score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.adj).double()
    val_acc, val_ce = evaluate(model, *val)
    return dict(logits=logits, val_acc=val_acc, val_ce=val_ce, test_acc=accuracy(model, *test))


def train_linear_teacher(graph, h, val, test, wd, iterations=500):
    weight = h.new_zeros(h.shape[1], int(graph.y.max()) + 1, requires_grad=True)
    bias = h.new_zeros(weight.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=iterations, tolerance_grad=1e-10, tolerance_change=1e-14,
                                  line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(h[graph.train_mask] @ weight + bias, graph.y[graph.train_mask]) + wd / 2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        logits = h @ weight + bias
    scores = {}
    for split, (_, mask) in (('val', val), ('test', test)):
        scores[f'{split}_acc'] = 100 * (logits[mask].argmax(1) == graph.y[mask]).double().mean().item()
        scores[f'{split}_ce'] = F.cross_entropy(logits[mask], graph.y[mask]).item()
    return dict(logits=logits, val_acc=scores['val_acc'], val_ce=scores['val_ce'], test_acc=scores['test_acc'])
