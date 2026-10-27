# GCRC: Graph Condensation via Risk-minimizing Clustering

## Pipeline

1. Teacher: 2-layer GCN (hidden 256, lr 0.01, 1000 epochs, lr x0.1 at epoch 500), checkpoint with the highest validation accuracy.
2. Stage 1: clustering of the nodes on `[rms(S^2 X), sqrt(lam) (q - mean q)]` with `q = softmax(teacher logits / T)`.
   Initialization: cells split across teacher-predicted classes in proportion to class size (at least one per class), Var-Part inside each class, then Lloyd iterations.
   Condensed features are cell means of `S^2 X`, condensed labels are cell means of `q`, and the condensed adjacency is the identity.
3. Stage 2: the hard assignment is relaxed to `softmax(s onehot + U V^T / sqrt(16))` (own-cell probability 0.99, `U = 0` at the start)
   and `U, V` are trained with Adam (lr 0.1) so that a ridge-regularized linear head fitted on the condensed nodes minimizes the teacher cross-entropy on all original nodes.
4. Evaluation: 2-layer GCN student (hidden 256, dropout 0.5, lr 0.01, weight decay 5e-4, 1000 epochs, lr x0.1 at epoch 500), trained on the condensed graph with soft labels; test accuracy at the epoch with the best validation accuracy (checked every 10 epochs), averaged over the student seeds 0-4 that were used for the validation selection.

## Usage

```
python main.py --dataset cora --ratio 0.013
bash run.sh [gpu] [data_dir]
```

Every dataset/ratio uses the hyperparameters in `src/hyperparams.py` (`BEST`), selected by validation accuracy; any of them can be overridden from the command line (`--steps -1` runs stage 1 only).
Cora and Citeseer are downloaded by PyG. Ogbn-arxiv, Flickr and Reddit are read from `data_dir/{ogbn-arxiv,flickr,reddit}/raw/` in the GraphSAINT format (`adj_full.npz`, `feats.npy`, `class_map.json`, `role.json`); Flickr and Reddit are inductive.

## Selected hyperparameters

| dataset | ratio | teacher (wd, dropout, seed) | T | lam | penalty | stage 2 steps |
|---|---|---|---|---|---|---|
| cora | 0.013 / 0.026 / 0.052 | 1e-4, 0.9, 42 | 0.1 / 0.2 / 2 | 0 / 0.1 / 1 | 3e-4 / 1e-4 / 1e-3 | 20 / 140 / 190 |
| citeseer | 0.009 / 0.018 / 0.036 | 5e-4, 0.9, 0 | 0.1 / 0.1 / 0.1 | 1 / 0 / 1 | - | stage 1 only |
| arxiv | 0.0005 / 0.0025 / 0.005 | 3e-5, 0.5, 42 | 0.5 / 0.2 / 0.2 | 0.1 / 0.3 / 0.3 | 1e-4 / 1e-4 / 1e-4 | 160 / 130 / 150 |
| flickr | 0.001 / 0.005 / 0.01 | 5e-4, 0.9, 0 | 0.5 / 0.5 / 1 | 0.3 / 1 / 1 | 1e-4 / 1e-3 / 1e-4 | 40 / 140 / 60 |
| reddit | 0.0005 / 0.001 / 0.002 | 1e-4, 0.7, 42 | 2 / 2 / 2 | 10 / 0.3 / 0.3 | 1e-4 / 1e-4 / 1e-3 | 80 / 70 / 0 |
