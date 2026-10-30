from pathlib import Path

import numpy as np
import torch

from src.bilevel import bilevel
from src.clustering import cell_means, cluster
from src.data import BUDGET, load_dataset, propagate
from src.hyperparams import get_args
from src.ops import make_deterministic
from src.student import train_student

args = get_args()
make_deterministic()
device = torch.device('cpu' if args.gpu < 0 else f'cuda:{args.gpu}')
print(vars(args))

graph, val, test = load_dataset(args.dataset, args.data_dir, device)
cells = BUDGET[(args.dataset, args.ratio)]
h = propagate(graph).double()
logits = torch.load(Path(args.teacher_dir) / args.dataset / f'{args.teacher}.pt').to(device)
q = (logits / args.T).softmax(1)
assign = cluster(h, q, logits.argmax(1), cells, args.lam)
if args.penalty is None:
    x, y = cell_means(h, assign, cells), cell_means(q, assign, cells)
else:
    x, y = bilevel(h, q, assign, cells, args.penalty, args.steps, args.rank, args.stage2_lr, args.leak)

scores = np.array([train_student(args, x.float(), y.float(), val, test, args.seed + i) for i in range(args.repeat)])
print(f'{args.dataset} {args.ratio:g}: test {scores[:, 1].mean():.2f} +- {scores[:, 1].std():.2f}  val {scores[:, 0].mean():.2f}')
