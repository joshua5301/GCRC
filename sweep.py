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
parser.add_argument('--teacher', type=str, default=None)
parser.add_argument('--joint', action='store_true')
args = parser.parse_args()
make_deterministic()
out = Path(args.out)
ratio_dir = out / (f'teacher_{args.teacher}' if args.teacher else f'select_{args.select}') / f'ratio_{args.ratio:g}'


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    torch.save(obj, tmp) if path.suffix == '.pt' else obj.to_csv(tmp, index=False)
    os.replace(tmp, path)


def teacher_name(wd, dropout, seed):
    return f'wd{wd:g}_d{dropout:g}_s{seed}'


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
    rows['val_rank'] = rows.search_val.rank(ascending=False, method='min').astype(int)
    rows['test_rank'] = rows.search_test.rank(ascending=False, method='min').astype(int)
    stage1 = rows[rows.step < 0]
    picks = {'stage1': stage1.sort_values(['search_val', 'step'], ascending=False).iloc[0],
             'joint_val_pick': rows.sort_values(['search_val', 'step'], ascending=False).iloc[0],
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
    for wd, dropout, seed in itertools.product(TEACHER_WDS, TEACHER_DROPOUTS, TEACHER_SEEDS):
        name = teacher_name(wd, dropout, seed)
        teacher = train_teacher(args, graph, val, test, seed, wd, dropout)
        save(teacher['logits'].cpu(), out / 'teachers' / f'{name}.pt')
        rows.append(dict(teacher=name, wd=wd, dropout=dropout, seed=seed, val_acc=teacher['val_acc'], val_ce=teacher['val_ce'],
                         test_acc=teacher['test_acc']))
        print(rows[-1], flush=True)
    save(pd.DataFrame(rows), out / 'teachers.csv')
    raise SystemExit

if args.select == 'calib':
    teacher = calibrated_teacher()
    args.T = float(teacher['T'])
    logits = torch.load(Path(teacher.pool) / 'teachers' / f'{teacher.teacher}.pt').to(device)
elif args.teacher:
    teacher = pd.Series(dict(teacher=args.teacher))
    logits = torch.load(out / 'teachers' / f'{args.teacher}.pt').to(device)
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
    for penalty in PENALTIES:
        for step, (x, y) in bilevel(h, q, assign, cells, penalty, STEPS, args.rank, args.stage2_lr, every=EVERY).items():
            rows.append(dict(point, penalty=penalty, step=step, **dict(zip(['search_val', 'search_test'], search(x, y, val, test)))))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), path)
