# GCRC: Graph Condensation via Risk-minimizing Clustering

## Pipeline

1. Teacher: GCN (hidden 256, lr 0.01, 1000 epochs, lr x0.1 at epoch 500), checkpoint with the highest validation accuracy. Pool of 45 teachers, layers {2, 3, 4} x weight decay {0, 3e-5, 1e-4, 3e-4, 5e-4} x dropout {0.5, 0.7, 0.9}, seed 0; the teacher with the highest validation accuracy (2 decimals) is used, ties broken by lower validation cross-entropy.
2. Stage 1: clustering of the nodes on `[rms(S^2 X), sqrt(lam) (q - mean q)]` with `q = softmax(teacher logits / T)`.
   Initialization: cells split across teacher-predicted classes in proportion to class size (at least one per class), Var-Part inside each class, then Lloyd iterations.
   Condensed features are cell means of `S^2 X`, condensed labels are cell means of `q`, and the condensed adjacency is the identity.
3. Stage 2: the hard assignment is relaxed to `softmax(s onehot + U V^T / sqrt(16))` (every other cell has probability 1e-6 at the start, `U = 0`)
   and `U, V` are trained with Adam (lr 0.1, 200 steps) so that a ridge-regularized linear head fitted on the condensed nodes minimizes the teacher cross-entropy on all original nodes.
4. Evaluation: 2-layer GCN student (hidden 256, dropout 0.5, lr 0.01, weight decay 5e-4, 1000 epochs, lr x0.1 at epoch 500), trained on the condensed graph with soft labels; test accuracy at the epoch with the best validation accuracy (checked every 10 epochs), averaged over student seeds 0-4.

## Selection

Validation accuracy of the five students, rounded to 2 decimals.

1. `(T, lam)` over T {0.1, 0.2, 0.5, 1, 2} x lam {0.1, 0.3, 1, 3} by stage 1 validation accuracy, ties broken by lower validation cross-entropy.
2. Stage 2 at that point with penalty {1e-4, 3e-4, 1e-3}; the stage 1 result and the three stage 2 results are compared by validation accuracy, ties go to stage 2.

## Usage

```
python main.py --dataset cora --ratio 0.013
python main.py --dataset cora --ratio 0.013 --train_teacher
bash run.sh [gpu] [data_dir]
```

By default the teacher is loaded from `teachers/<dataset>/`; `--train_teacher` trains it from scratch with the same configuration (`TEACHERS` in `src/hyperparams.py`).
The other hyperparameters come from `BEST` in `src/hyperparams.py`; any of them can be overridden from the command line (`--stage1_only` skips stage 2).
Cora and Citeseer are downloaded by PyG. Ogbn-arxiv, Flickr and Reddit are read from `data_dir/{ogbn-arxiv,flickr,reddit}/raw/` in the GraphSAINT format (`adj_full.npz`, `feats.npy`, `class_map.json`, `role.json`); Flickr and Reddit are inductive.

## Selected hyperparameters

| dataset | ratio | teacher | T | lam | penalty | test |
|---|---|---|---|---|---|---|
| cora | 0.013 / 0.026 / 0.052 | L4, wd 0, dropout 0.9 | 0.5 / 0.2 / 1 | 0.1 / 1 / 0.3 | 1e-3 / 3e-4 / 1e-4 | 83.56 / 83.78 / 83.44 |
| citeseer | 0.009 / 0.018 / 0.036 | L2, wd 5e-4, dropout 0.7 | 1 / 0.5 / 1 | 1 / 0.1 / 0.1 | - / - / - | 74.28 / 74.48 / 74.98 |
| arxiv | 0.0005 / 0.0025 / 0.005 | L4, wd 3e-5, dropout 0.5 | 0.2 / 0.5 / 0.2 | 0.3 / 0.1 / 0.1 | 3e-4 / 1e-4 / 1e-4 | 68.16 / 69.87 / 70.28 |
| flickr | 0.001 / 0.005 / 0.01 | L4, wd 3e-4, dropout 0.9 | 0.5 / 0.5 / 0.5 | 0.3 / 0.3 / 0.1 | - / - / - | 47.44 / 47.18 / 47.24 |
| reddit | 0.0005 / 0.001 / 0.002 | L2, wd 1e-4, dropout 0.7 | 2 / 2 / 2 | 0.3 / 0.1 / 0.3 | - / - / - | 92.36 / 92.39 / 92.82 |

`-` means stage 1 was selected; stage 2 was not run for Citeseer. The teachers in `teachers/` and the test accuracies were obtained on an RTX PRO 6000 Blackwell Server Edition (RunPod); other GPUs give slightly different numbers.
