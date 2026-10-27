import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.models import GCN, accuracy, evaluate


def train_teacher(args, graph, val, test, seed, wd, dropout):
    seed_everything(seed)
    model = GCN(graph.num_features, args.hidden, int(graph.y.max()) + 1, dropout).to(graph.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        if epoch == args.epochs // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=wd)
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(graph.x, graph.adj)[graph.train_mask], graph.y[graph.train_mask]).backward()
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        acc = accuracy(model, *val)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.adj).double()
    val_acc, val_ce = evaluate(model, *val)
    return dict(logits=logits, val_acc=val_acc, val_ce=val_ce, test_acc=accuracy(model, *test))
