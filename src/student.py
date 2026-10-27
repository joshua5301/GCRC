import torch
from torch_geometric import seed_everything

from src.models import GCN, accuracy


def train_student(args, x, y, val, test, seed):
    seed_everything(seed)
    model = GCN(x.shape[1], args.hidden, y.shape[1], args.dropout).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_test = -1.0, 0.0
    for epoch in range(1, args.student_epochs + 1):
        if epoch == args.student_epochs // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
        model.train()
        optimizer.zero_grad()
        (-(y * model(x)).sum(1).mean()).backward()
        optimizer.step()
        if epoch % args.eval_every and epoch != args.student_epochs:
            continue
        val_acc = accuracy(model, *val)
        if val_acc > best_val:
            best_val, best_test = val_acc, accuracy(model, *test)
    return best_val, best_test
