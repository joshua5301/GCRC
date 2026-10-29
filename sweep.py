import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.bilevel import bilevel
from src.clustering import cell_means, cluster
from src.data import BUDGET, load_dataset, propagate
from src.hyperparams import get_parser
from src.ops import make_deterministic
from src.student import train_student
from src.teacher import train_linear_teacher, train_teacher

TEACHER_WDS, TEACHER_DROPOUTS, TEACHER_SEEDS = [0.0, 3e-5, 1e-4, 3e-4, 5e-4], [0.5, 0.7, 0.9], [0, 42]
TEMPERATURES, LAMS, PENALTIES = [0.1, 0.2, 0.5, 1.0, 2.0], [0.0, 0.1, 0.3, 1.0, 3.0, 10.0], [1e-4, 3e-4, 1e-3]
STEPS, EVERY = 200, 10

parser = get_parser()
parser.add_argument('--stage', choices=['teachers', 'grid', 'summary'], required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--lams', type=float, nargs='*', default=LAMS)
parser.add_argument('--select', choices=['acc', 'ce', 'calib'], default='acc')
parser.add_argument('--pools', type=str, nargs='*', default=None)
parser.add_argument('--teacher_model', choices=['gcn', 'sgc'], default='gcn')
parser.add_argument('--teacher_wds', type=float, nargs='*', default=TEACHER_WDS)
parser.add_argument('--teacher_seeds', type=int, nargs='*', default=TEACHER_SEEDS)
parser.add_argument('--teacher_layers', type=int, nargs='*', default=[2])
parser.add_argument('--teacher_hiddens', type=int, nargs='*', default=[256])
parser.add_argument('--teacher', type=str, default=None)
parser.add_argument('--joint', action='store_true')
parser.add_argument('--ensemble', type=int, default=0)
parser.add_argument('--stage1_only', action='store_true')
parser.add_argument('--every', type=int, default=EVERY, help='stage 2 checkpoint interval; 0 = last step only')
args = parser.parse_args()
make_deterministic()
out = Path(args.out)
ensemble = f'_k{args.ensemble}' if args.ensemble else ''
ratio_dir = out / (f'teacher_{args.teacher}' if args.teacher else f'select_{args.select}{ensemble}') / f'ratio_{args.ratio:g}'


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    torch.save(obj, tmp) if path.suffix == '.pt' else obj.to_csv(tmp, index=False)
    os.replace(tmp, path)


def teacher_name(wd, dropout, seed, layers=2, hidden=256):
    arch = '' if (layers, hidden) == (2, 256) else f'L{layers}_h{hidden}_'
    return f'{arch}wd{wd:g}_d{dropout:g}_s{seed}'


def calibrated_teacher():
    rows = []
    for pool in map(Path, args.pools or [args.out]):
        for name in pd.read_csv(pool / 'teachers.csv').teacher:
            logits = torch.load(pool / 'teachers' / f'{name}.pt').to(device)
            for T in TEMPERATURES:
                ce = F.cross_entropy(logits[graph.val_mask] / T, graph.y[graph.val_mask]).item()
                rows.append(dict(pool=str(pool), teacher=name, T=T, val_ce=ce))
    table = pd.DataFrame(rows)
    save(table, ratio_dir.parent / 'calibration.csv')
    return table.sort_values('val_ce', kind='stable').iloc[0]


def ensemble_teacher():
    names = pd.read_csv(out / 'teachers.csv').teacher
    configs = sorted({name.rsplit('_s', 1)[0] for name in names})
    rows, members = [], {}
    for config in configs:
        seeds = sorted(int(name.rsplit('_s', 1)[1]) for name in names if name.rsplit('_s', 1)[0] == config)[:args.ensemble]
        logits = torch.stack([torch.load(out / 'teachers' / f'{config}_s{seed}.pt').to(device) for seed in seeds]).mean(0)
        members[config] = logits
        acc = 100 * (logits[graph.val_mask].argmax(1) == graph.y[graph.val_mask]).double().mean().item()
        ce = F.cross_entropy(logits[graph.val_mask], graph.y[graph.val_mask]).item()
        test = 100 * (logits[graph.test_mask].argmax(1) == graph.y[graph.test_mask]).double().mean().item()
        rows.append(dict(teacher=f'{config}_k{args.ensemble}', seeds=' '.join(map(str, seeds)), val_acc=acc, val_ce=ce, test_acc=test))
    table = pd.DataFrame(rows)
    save(table, ratio_dir.parent / 'ensembles.csv')
    if args.select == 'ce':
        pick = table.sort_values('val_ce', kind='stable').iloc[0]
    else:
        pick = table.sort_values(['val_acc', 'val_ce'], ascending=[False, True], kind='stable').iloc[0]
    return pick, members[pick.teacher.rsplit('_k', 1)[0]]


def picked_teacher():
    table = pd.read_csv(out / 'teachers.csv')
    if args.select == 'ce':
        return table.sort_values('val_ce', kind='stable').iloc[0]
    return table.sort_values(['val_acc', 'val_ce'], ascending=[False, True], kind='stable').iloc[0]


def search(x, y, val, test):
    scores = np.array([train_student(args, x.float(), y.float(), val, test, args.seed + i) for i in range(args.repeat)])
    return scores[:, 0].mean(), scores[:, 1].mean()


if args.stage == 'summary':
    files = out.glob(f'teacher_*/ratio_{args.ratio:g}/T*_lam*.csv') if args.joint else ratio_dir.glob('T*_lam*.csv')
    if args.joint:
        ratio_dir = out / 'joint' / f'ratio_{args.ratio:g}'
    rows = pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True)
    rows['val'] = rows.search_val.round(2)  # ties at 2 decimals -> larger step
    rows['val_rank'] = rows.val.rank(ascending=False, method='min').astype(int)
    rows['test_rank'] = rows.search_test.rank(ascending=False, method='min').astype(int)
    stage1 = rows[rows.step < 0]
    picks = {'stage1': stage1.sort_values(['val', 'step'], ascending=False).iloc[0],
             'joint_val_pick': rows.sort_values(['val', 'step'], ascending=False).iloc[0],
             'oracle': rows.sort_values(['search_test', 'search_val'], ascending=False).iloc[0]}
    summary = pd.DataFrame([dict(phase=phase, **pick) for phase, pick in picks.items()]).assign(points=len(rows))
    save(summary, ratio_dir / 'summary.csv')
    print(summary.to_string(index=False))
    raise SystemExit

device = torch.device('cpu' if args.gpu < 0 else f'cuda:{args.gpu}')
graph, val, test = load_dataset(args.dataset, args.data_dir, device)

if args.stage == 'teachers' and args.teacher_model == 'sgc':
    h = propagate(graph).double()
    rows = []
    for wd in args.teacher_wds:
        name = f'sgc_wd{wd:g}'
        teacher = train_linear_teacher(graph, h, val, test, wd)
        save(teacher['logits'].cpu(), out / 'teachers' / f'{name}.pt')
        rows.append(dict(teacher=name, wd=wd, val_acc=teacher['val_acc'], val_ce=teacher['val_ce'], test_acc=teacher['test_acc']))
        print(rows[-1], flush=True)
    save(pd.DataFrame(rows), out / 'teachers.csv')
    raise SystemExit

if args.stage == 'teachers':
    rows = []
    for layers, hidden, wd, dropout, seed in itertools.product(args.teacher_layers, args.teacher_hiddens, args.teacher_wds,
                                                               TEACHER_DROPOUTS, args.teacher_seeds):
        name = teacher_name(wd, dropout, seed, layers, hidden)
        teacher = train_teacher(args, graph, val, test, seed, wd, dropout, layers, hidden)
        save(teacher['logits'].cpu(), out / 'teachers' / f'{name}.pt')
        rows.append(dict(teacher=name, layers=layers, hidden=hidden, wd=wd, dropout=dropout, seed=seed, val_acc=teacher['val_acc'], val_ce=teacher['val_ce'],
                         test_acc=teacher['test_acc']))
        print(rows[-1], flush=True)
    save(pd.DataFrame(rows), out / 'teachers.csv')
    raise SystemExit

if args.select == 'calib':
    teacher = calibrated_teacher()
    args.T = float(teacher['T'])
    logits = torch.load(Path(teacher.pool) / 'teachers' / f'{teacher.teacher}.pt').to(device)
elif args.ensemble:
    teacher, logits = ensemble_teacher()
elif args.teacher:
    teacher = pd.Series(dict(teacher=args.teacher))
    path = out / 'teachers' / f'{args.teacher}.pt'
    if not path.exists():  # committed teachers (teachers/<dataset>/)
        path = Path(__file__).parent / 'teachers' / args.dataset / f'{args.teacher}.pt'
    logits = torch.load(path).to(device)
else:
    teacher = picked_teacher()
    logits = torch.load(out / 'teachers' / f'{teacher.teacher}.pt').to(device)
h = propagate(graph).double()
cells = BUDGET[(args.dataset, args.ratio)]
q = (logits / args.T).softmax(1)
for lam in args.lams:
    path = ratio_dir / f'T{args.T:g}_lam{lam:g}.csv'
    if path.exists():
        continue
    assign = cluster(h, q, logits.argmax(1), cells, lam)
    point = dict(teacher=teacher.teacher, T=args.T, lam=lam)
    rows = [dict(point, penalty=np.nan, step=-1, **dict(zip(['search_val', 'search_test'],
                                                           search(cell_means(h, assign, cells), cell_means(q, assign, cells), val, test))))]
    print(rows[-1], flush=True)
    for penalty in [] if args.stage1_only else PENALTIES:
        for step, (x, y) in bilevel(h, q, assign, cells, penalty, STEPS, args.rank, args.stage2_lr, every=args.every or None).items():
            rows.append(dict(point, penalty=penalty, step=step, **dict(zip(['search_val', 'search_test'], search(x, y, val, test)))))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), path)
