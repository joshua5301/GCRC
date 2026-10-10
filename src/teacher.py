import os

from src.models import *


def score(model, data, mask, criterion):
    out = model(data)[mask]
    if criterion == 'ce':
        return -F.nll_loss(out, data.y[mask]).item()
    return (out.argmax(1) == data.y[mask]).float().mean().item()


def train_teacher(args, data, data_val=None, data_test=None):
    seed_everything(args.seed)
    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.teacher_dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    val_data, val_mask = (data_val, slice(None)) if data_val is not None else (data, data.val_mask)
    best_score, best_state = -float('inf'), None
    for epoch in range(1, args.epoch + 1):
        if epoch == args.epoch // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
        model.train()
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epoch:
            continue
        model.eval()
        with torch.no_grad():
            val_score = score(model, val_data, val_mask, args.teacher_criterion)
        if val_score > best_score:
            best_score, best_state = val_score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = score(model, data_test, slice(None), 'acc') if data_test is not None else \
            score(model, data, data.test_mask, 'acc') if 'test_mask' in data else float('nan')
        return dict(log_prob=model(data).double(), val_score=best_score, test_acc=test_acc)


def teacher_path(args):
    return f'{args.teacher_dir}/{args.dataset_name}_d{args.teacher_dropout:g}_{args.teacher_criterion}_s{args.seed}.pt'


def get_teacher_labels(args, data, data_val=None, data_test=None):
    # with --teacher_dir, a teacher trained earlier (same dataset, dropout, criterion, seed) is loaded instead of retrained
    path = teacher_path(args) if args.teacher_dir else None
    loaded = bool(path) and os.path.exists(path)
    if loaded:
        teacher = torch.load(path, map_location=args.device)
    else:
        teacher = train_teacher(args, data, data_val, data_test)
        if path:
            os.makedirs(args.teacher_dir, exist_ok=True)
            torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in teacher.items()}, f'{path}.{os.getpid()}.tmp')
            os.replace(f'{path}.{os.getpid()}.tmp', path)
    print(f'teacher dropout {args.teacher_dropout:g}: val {100 * teacher["val_score"]:.2f}  test {100 * teacher["test_acc"]:.2f}'
          + ('  (loaded)' if loaded else ''))
    return F.softmax(teacher['log_prob'].to(args.device) / args.T, dim=1)
