import argparse

BEST = {
    ('cora', 0.013): dict(teacher_wd=1e-4, teacher_dropout=0.9, teacher_seed=42, T=0.1, lam=0.0, penalty=3e-4, steps=20),
    ('cora', 0.026): dict(teacher_wd=1e-4, teacher_dropout=0.9, teacher_seed=42, T=0.2, lam=0.1, penalty=1e-4, steps=140),
    ('cora', 0.052): dict(teacher_wd=1e-4, teacher_dropout=0.9, teacher_seed=42, T=2.0, lam=1.0, penalty=1e-3, steps=190),
    ('citeseer', 0.009): dict(teacher_wd=5e-4, teacher_dropout=0.9, teacher_seed=0, T=0.1, lam=1.0, penalty=None, steps=-1),
    ('citeseer', 0.018): dict(teacher_wd=5e-4, teacher_dropout=0.9, teacher_seed=0, T=0.1, lam=0.0, penalty=None, steps=-1),
    ('citeseer', 0.036): dict(teacher_wd=5e-4, teacher_dropout=0.9, teacher_seed=0, T=0.1, lam=1.0, penalty=None, steps=-1),
    ('arxiv', 0.0005): dict(teacher_wd=3e-5, teacher_dropout=0.5, teacher_seed=42, T=0.5, lam=0.1, penalty=1e-4, steps=160),
    ('arxiv', 0.0025): dict(teacher_wd=3e-5, teacher_dropout=0.5, teacher_seed=42, T=0.2, lam=0.3, penalty=1e-4, steps=130),
    ('arxiv', 0.005): dict(teacher_wd=3e-5, teacher_dropout=0.5, teacher_seed=42, T=0.2, lam=0.3, penalty=1e-4, steps=150),
    ('flickr', 0.001): dict(teacher_wd=5e-4, teacher_dropout=0.9, teacher_seed=0, T=0.5, lam=0.3, penalty=1e-4, steps=40),
    ('flickr', 0.005): dict(teacher_wd=5e-4, teacher_dropout=0.9, teacher_seed=0, T=0.5, lam=1.0, penalty=1e-3, steps=140),
    ('flickr', 0.01): dict(teacher_wd=5e-4, teacher_dropout=0.9, teacher_seed=0, T=1.0, lam=1.0, penalty=1e-4, steps=60),
    ('reddit', 0.0005): dict(teacher_wd=1e-4, teacher_dropout=0.7, teacher_seed=42, T=2.0, lam=10.0, penalty=1e-4, steps=80),
    ('reddit', 0.001): dict(teacher_wd=1e-4, teacher_dropout=0.7, teacher_seed=42, T=2.0, lam=0.3, penalty=1e-4, steps=70),
    ('reddit', 0.002): dict(teacher_wd=1e-4, teacher_dropout=0.7, teacher_seed=42, T=2.0, lam=0.3, penalty=1e-3, steps=0),
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'arxiv', 'flickr', 'reddit'])
    parser.add_argument('--ratio', type=float, default=0.013)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--gpu', type=int, default=0, help='-1 for cpu')
    parser.add_argument('--seed', type=int, default=0, help='student i uses seed + i')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--student_epochs', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--teacher_wd', type=float, default=None)
    parser.add_argument('--teacher_dropout', type=float, default=None)
    parser.add_argument('--teacher_seed', type=int, default=None)
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--penalty', type=float, default=None)
    parser.add_argument('--steps', type=int, default=None, help='-1 for stage 1 only')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--stage2_lr', type=float, default=0.1)
    return parser


def get_args():
    args = get_parser().parse_args()
    for name, value in BEST[(args.dataset, args.ratio)].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args
