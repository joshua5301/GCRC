import argparse

def get_hyperparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="reddit", help='cora, citeseer, arxiv, flickr, reddit')
    parser.add_argument('--ratio', type=float, default=0.001)
    parser.add_argument('--raw_data_dir', type=str, default="./data/")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10, help='evaluation interval')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_dim', type=int, default=256)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropouts', type=str, default=None, help='comma-separated dropout values')
    parser.add_argument('--teacher_dropout', type=float, default=None, help='dropout of the 2-layer GCN teacher')
    parser.add_argument('--teacher_criterion', type=str, default='acc', help='teacher early stopping by val acc or ce')
    parser.add_argument('--T', type=float, default=None, help='teacher softmax temperature')
    parser.add_argument('--lam', type=float, default=None, help='weight of label variance in joint clustering')
    parser.add_argument('--stage2_steps', type=int, default=None, help='bilevel stage 2 Adam steps (unset = stage 1 only, 0 = soft initial assignment)')
    parser.add_argument('--rank', type=int, default=16, help='rank of the stage 2 soft assignment U V^T')
    parser.add_argument('--tau', type=float, default=0.03, help='softmax temperature of the stage 2 assignment')
    parser.add_argument('--penalty', type=float, default=1e-5, help='ridge penalty of the stage 2 inner linear head')
    parser.add_argument('--stage2_lr', type=float, default=0.01, help='Adam learning rate of stage 2')

    args = parser.parse_args()
    return args

BEST_HYPERPARAMS_DICT = {
    # dataset    ratio     t_dropout, T,    lam,   dropout
    ('cora',     0.013):  (0.5,       1.0,  0.1,   '0.9'),
    ('cora',     0.026):  (0.9,       1.0,  0.01,  '0.9'),
    ('cora',     0.052):  (0.5,       2.0,  0.01,  '0.9'),
    ('citeseer', 0.009):  (0.1,       0.2,  10.0,  '0.5'),
    ('citeseer', 0.018):  (0.5,       0.05, 0.01,  '0.5'),
    ('citeseer', 0.036):  (0.5,       1.0,  0.1,   '0.5'),
    ('arxiv',    0.0005): (0.1,       1.0,  1.0,   '0.5'),
    ('arxiv',    0.0025): (0.1,       0.05, 0.1,   '0.5'),
    ('arxiv',    0.005):  (0.1,       0.05, 0.1,   '0.5'),
    ('flickr',   0.001):  (0.5,       0.2,  0.1,   '0.5'),
    ('flickr',   0.005):  (0.5,       0.2,  0.1,   '0.5'),
    ('flickr',   0.01):   (0.5,       2.0,  1.0,   '0.5'),
    ('reddit',   0.0005): (0.5,       2.0,  0.1,   '0.5'),
    ('reddit',   0.001):  (0.1,       2.0,  0.01,  '0.5'),
    ('reddit',   0.002):  (0.5,       2.0,  0.01,  '0.5'),
}

def override_to_best_hyperparams(args):
    best = BEST_HYPERPARAMS_DICT[(args.dataset_name, args.ratio)]
    for name, value in zip(['teacher_dropout', 'T', 'lam', 'dropouts'], best):
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args
