from src.hyperparams import *
from src.models import *
from src.utils import *
from src.module import *
from src.dataloader import *
from src.teacher import get_teacher_labels
from src.partition import partition
from src.bilevel import bilevel

args = get_hyperparams()
args = device_setting(args)
seed_everything(args.seed)

args = override_to_best_hyperparams(args)
print(f'{args.dataset_name} r={args.ratio:g}: teacher dropout={args.teacher_dropout:g} criterion={args.teacher_criterion} '
      f'T={args.T:g} lam={args.lam:g} | student lr={args.lr:g} wd={args.weight_decay:g} dropouts={args.dropouts}')

## data
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

## condensation
begin = time.time()
budget_node_num = budget(args)
H0, H1, H2 = conv_graph_multi(args, data)
X = H2
t_prop = time.time()

y_pred = get_teacher_labels(args, data, data_val, data_test)
t_teacher = time.time()
if data_val is None:
    print(f'teacher test acc: {100 * (y_pred.argmax(1)[data.test_mask] == data.y[data.test_mask]).double().mean():.2f}%')
X_cond, y_cond, assign = partition(X, y_pred, budget_node_num, args.lam)
t_stage1 = time.time()

def evaluate(X_cond, y_cond, repeat, verbose=True):
    X_cond, y_cond, n = X_cond.float(), y_cond.float(), len(X_cond)
    graph = Data(x=X_cond, y=y_cond, edge_index=torch.eye(n).nonzero().t().to(X_cond.device),
                 edge_attr=torch.ones(n, device=X_cond.device), train_mask=torch.ones(n, dtype=torch.bool, device=X_cond.device)).to(args.device)
    results = {}
    for dropout in [float(v) for v in str(args.dropouts).split(',')]:
        runs = []
        for _ in range(repeat):
            model = GCN(data.num_features, args.n_dim, args.num_class, 2, dropout).to(args.device)
            runs.append(model_training(model, args, data, graph, data_val, data_test))
        val, test = np.mean([v for v, _ in runs]), [t for _, t in runs]
        results[dropout] = (val, np.mean(test), np.std(test, ddof=1) if len(test) > 1 else 0.0)
        if verbose:
            print(f'-- dropout {dropout:g}: val {100 * val:.2f}  test {100 * np.mean(test):.2f} +- {100 * results[dropout][2]:.2f}')
    return results

search = []
def on_checkpoint(step, X_step, y_step):
    # student val search over the stage 2 checkpoints (test recorded too, selection by val only); True = stop early
    results = evaluate(X_step, y_step, args.search_repeat, verbose=False)
    dropout = max(results, key=lambda d: results[d][0])
    search.append((results[dropout][0], step, X_step, y_step))
    print(f'checkpoint step {step}: val {100 * results[dropout][0]:.2f}  test {100 * results[dropout][1]:.2f}  (dropout {dropout:g})', flush=True)
    best = max(range(len(search)), key=lambda i: search[i][0])
    return args.patience is not None and len(search) - 1 - best >= args.patience

if args.stage2_steps is not None:
    X_cond, y_cond = bilevel(X, y_pred, assign, budget_node_num, args.lam, args.rank, args.tau, args.penalty,
                             args.stage2_steps, args.stage2_lr, args.stage2_solver, args.inner_steps, args.inner_lr,
                             args.hinv_steps, args.hinv_lr, args.checkpoint_every, on_checkpoint if args.checkpoint_every else None)
if search:
    _, step, X_cond, y_cond = max(search, key=lambda c: c[0])
    print(f'checkpoint selected by val: step {step}')
args.cond_time = time.time() - begin
print(f'condensed: {len(X_cond)} nodes (budget {budget_node_num})  time {args.cond_time:.2f} s')
print(f'time: propagation {t_prop - begin:.2f} s, teacher {t_teacher - t_prop:.2f} s, stage 1 {t_stage1 - t_teacher:.2f} s, '
      f'stage 2 {time.time() - t_stage1:.2f} s' + (' (incl. checkpoint student search)' if search else ''))

## student
results = evaluate(X_cond, y_cond, args.repeat)
best = max(results, key=lambda d: results[d][0])
print(f'== {args.dataset_name} r={args.ratio:g}: {100 * results[best][1]:.2f} +- {100 * results[best][2]:.2f}  (dropout {best:g}, val {100 * results[best][0]:.2f})')
