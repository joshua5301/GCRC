import torch
from torch_geometric import seed_everything

from src.models import GCN, accuracy, evaluate


def train_student(args, x, y, val, test, seed):
    seed_everything(seed)
    model = GCN(x.shape[1], args.hidden, y.shape[1], args.dropout).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_test, best_ce = -1.0, 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.epochs // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
        model.train()
        optimizer.zero_grad()
        (-(y * model(x)).sum(1).mean()).backward()
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        val_acc, val_ce = evaluate(model, *val)
        if val_acc > best_val:
            best_val, best_test, best_ce = val_acc, accuracy(model, *test), val_ce
    return best_val, best_test, best_ce
