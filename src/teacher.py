from src.models import *


def score(model, data, mask, criterion):
    out = model(data)[mask]
    if criterion == 'ce':
        return -F.nll_loss(out, data.y[mask]).item()
    return (out.argmax(1) == data.y[mask]).float().mean().item()


def get_teacher_labels(args, data, data_val=None):
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
        return F.softmax(model(data).double() / args.T, dim=1)
