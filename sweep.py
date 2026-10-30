import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.bilevel import bilevel
from src.clustering import cell_means, cluster
from src.data import BUDGET, load_dataset, propagate
from src.hyperparams import get_parser
from src.ops import make_deterministic
from src.student import train_student
from src.teacher import train_teacher

LAYERS, WDS, DROPOUTS = [2, 3, 4], [0.0, 3e-5, 1e-4, 3e-4, 5e-4], [0.5, 0.7, 0.9]
TEMPERATURES, LAMS, PENALTIES = [0.1, 0.2, 0.5, 1.0, 2.0], [0.1, 0.3, 1.0, 3.0], [1e-4, 3e-4, 1e-3]

parser = get_parser()
parser.add_argument('--stage', choices=['teachers', 'stage1', 'stage2'], required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--temperatures', type=float, nargs='*', default=TEMPERATURES)
parser.add_argument('--lams', type=float, nargs='*', default=LAMS)
args = parser.parse_args()
make_deterministic()
device = torch.device('cpu' if args.gpu < 0 else f'cuda:{args.gpu}')
out = Path(args.out) / args.dataset


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    torch.save(obj, tmp) if path.suffix == '.pt' else obj.to_csv(tmp, index=False)
    os.replace(tmp, path)


def teacher_name(layers, wd, dropout, seed):
    return f'{"" if layers == 2 else f"L{layers}_h{args.hidden}_"}wd{wd:g}_d{dropout:g}_s{seed}'


def picked_teacher():
    if args.teacher:
        return args.teacher
    table = pd.read_csv(out / 'teachers.csv').assign(val=lambda d: d.val_acc.round(2))
    return table.sort_values(['val', 'val_ce'], ascending=[False, True], kind='stable').teacher.iloc[0]


def load_teacher(name):
    path = out / 'teachers' / f'{name}.pt'
    return torch.load(path if path.exists() else Path(args.teacher_dir) / args.dataset / f'{name}.pt').to(device)


def search(x, y):
    scores = np.array([train_student(args, x.float(), y.float(), val, test, args.seed + i) for i in range(args.repeat)])
    return dict(zip(['val', 'test', 'val_ce'], scores.mean(0)))


graph, val, test = load_dataset(args.dataset, args.data_dir, device)

if args.stage == 'teachers':
    rows = []
    for layers, wd, dropout in itertools.product(LAYERS, WDS, DROPOUTS):
        name = teacher_name(layers, wd, dropout, args.seed)
        teacher = train_teacher(args, graph, val, test, args.seed, wd, dropout, layers, args.hidden)
        save(teacher['logits'].cpu(), out / 'teachers' / f'{name}.pt')
        rows.append(dict(teacher=name, layers=layers, wd=wd, dropout=dropout, seed=args.seed,
                         **{k: teacher[k] for k in ('val_acc', 'val_ce', 'test_acc')}))
        print(rows[-1], flush=True)
    save(pd.DataFrame(rows), out / 'teachers.csv')
    print('picked', picked_teacher())
    raise SystemExit

name = picked_teacher()
logits = load_teacher(name)
h = propagate(graph).double()
cells = BUDGET[(args.dataset, args.ratio)]
ratio_dir = out / name / f'ratio_{args.ratio:g}'


def condense(T, lam):
    q = (logits / T).softmax(1)
    return q, cluster(h, q, logits.argmax(1), cells, lam)


if args.stage == 'stage1':
    for T, lam in itertools.product(args.temperatures, args.lams):
        path = ratio_dir / f'T{T:g}_lam{lam:g}.csv'
        if path.exists():
            continue
        q, assign = condense(T, lam)
        row = dict(T=T, lam=lam, **search(cell_means(h, assign, cells), cell_means(q, assign, cells)))
        save(pd.DataFrame([row]), path)
        print(row, flush=True)
    raise SystemExit

stage1 = pd.concat([pd.read_csv(ratio_dir / f'T{T:g}_lam{lam:g}.csv') for T, lam in itertools.product(args.temperatures, args.lams)])
stage1 = stage1.assign(rounded=stage1.val.round(2)).sort_values(['rounded', 'val_ce'], ascending=[False, True], kind='stable')
T, lam = stage1.iloc[0][['T', 'lam']]
q, assign = condense(T, lam)
rows = [dict(stage1.iloc[0][['T', 'lam', 'val', 'test', 'val_ce']], penalty=np.nan)]
for penalty in PENALTIES:
    rows.append(dict(T=T, lam=lam, penalty=penalty, **search(*bilevel(h, q, assign, cells, penalty, args.steps, args.rank, args.stage2_lr, args.leak))))
    print(rows[-1], flush=True)
stage2 = pd.DataFrame(rows)
save(stage2, ratio_dir / 'stage2.csv')
pick = stage2.assign(rounded=stage2.val.round(2), stage=stage2.penalty.notna()).sort_values(['rounded', 'stage'], ascending=False, kind='stable')
print(pick.drop(columns=['rounded', 'stage']).to_string(index=False))
save(pick.iloc[:1].drop(columns=['rounded', 'stage']), ratio_dir / 'summary.csv')
