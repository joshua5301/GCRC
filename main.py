import time

import numpy as np
import torch

from src.bilevel import bilevel
from src.clustering import cell_means, cluster
from src.data import BUDGET, load_dataset, propagate
from src.hyperparams import get_args
from src.student import train_student
from src.teacher import train_teacher

args = get_args()
device = torch.device('cpu' if args.gpu < 0 else f'cuda:{args.gpu}')
print(vars(args))

graph, val, test = load_dataset(args.dataset, args.data_dir, device)
cells = BUDGET[(args.dataset, args.ratio)]

start = time.time()
h = propagate(graph).double()
logits, teacher_val, teacher_test = train_teacher(args, graph, val, test)
print(f'teacher: val {teacher_val:.2f}  test {teacher_test:.2f}')
q = (logits / args.T).softmax(1)
assign = cluster(h, q, logits.argmax(1), cells, args.lam)
if args.steps < 0:
    x_cond, y_cond = cell_means(h, assign, cells), cell_means(q, assign, cells)
else:
    x_cond, y_cond = bilevel(h, q, assign, cells, args.penalty, args.steps, args.rank, args.stage2_lr)
print(f'condensed {graph.num_nodes} -> {cells} nodes in {time.time() - start:.1f} s')

results = np.array([train_student(args, x_cond.float(), y_cond.float(), val, test, args.seed + i) for i in range(args.repeat)])
std = results[:, 1].std(ddof=1) if args.repeat > 1 else 0.0
print(f'{args.dataset} {args.ratio:g}: test {results[:, 1].mean():.2f} +- {std:.2f}  (val {results[:, 0].mean():.2f})')
