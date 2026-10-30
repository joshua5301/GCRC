import argparse

BEST = {
    ('cora', 0.013): dict(teacher='L4_h256_wd0_d0.9_s0', T=0.5, lam=0.1, penalty=1e-3),
    ('cora', 0.026): dict(teacher='L4_h256_wd0_d0.9_s0', T=0.2, lam=1.0, penalty=3e-4),
    ('cora', 0.052): dict(teacher='L4_h256_wd0_d0.9_s0', T=1.0, lam=0.3, penalty=1e-4),
    ('citeseer', 0.009): dict(teacher='wd0.0005_d0.7_s0', T=1.0, lam=1.0, penalty=None),
    ('citeseer', 0.018): dict(teacher='wd0.0005_d0.7_s0', T=0.5, lam=0.1, penalty=None),
    ('citeseer', 0.036): dict(teacher='wd0.0005_d0.7_s0', T=1.0, lam=0.1, penalty=None),
    ('arxiv', 0.0005): dict(teacher='L4_h256_wd3e-05_d0.5_s0', T=0.2, lam=0.3, penalty=3e-4),
    ('arxiv', 0.0025): dict(teacher='L4_h256_wd3e-05_d0.5_s0', T=0.5, lam=0.1, penalty=1e-4),
    ('arxiv', 0.005): dict(teacher='L4_h256_wd3e-05_d0.5_s0', T=0.2, lam=0.1, penalty=1e-4),
    ('flickr', 0.001): dict(teacher='L4_h256_wd0.0005_d0.9_s0', T=0.5, lam=0.1, penalty=None),
    ('flickr', 0.005): dict(teacher='L4_h256_wd0.0005_d0.9_s0', T=0.5, lam=0.3, penalty=None),
    ('flickr', 0.01): dict(teacher='L4_h256_wd0.0005_d0.9_s0', T=0.1, lam=0.3, penalty=1e-4),
    ('reddit', 0.0005): dict(teacher='wd0.0001_d0.7_s0', T=2.0, lam=0.3, penalty=None),
    ('reddit', 0.001): dict(teacher='wd0.0001_d0.7_s0', T=2.0, lam=0.1, penalty=None),
    ('reddit', 0.002): dict(teacher='wd0.0001_d0.7_s0', T=2.0, lam=0.3, penalty=None),
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'arxiv', 'flickr', 'reddit'])
    parser.add_argument('--ratio', type=float, default=0.013)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--teacher', type=str, default=None)
    parser.add_argument('--teacher_dir', type=str, default='./teachers/')
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--penalty', type=float, default=None)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--stage2_lr', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=1e-6)
    return parser


def get_args():
    parser = get_parser()
    parser.add_argument('--stage1_only', action='store_true')
    args = parser.parse_args()
    best = BEST[(args.dataset, args.ratio)]
    for name in ('teacher', 'T', 'lam'):
        if getattr(args, name) is None:
            setattr(args, name, best[name])
    if args.penalty is None:
        args.penalty = best['penalty']
    if args.stage1_only:
        args.penalty = None
    return args
