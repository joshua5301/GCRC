#!/bin/bash
gpu=${1:-0}
data_dir=${2:-./data/}

for r in 0.013 0.026 0.052; do python main.py --gpu $gpu --data_dir $data_dir --dataset cora --ratio $r; done
for r in 0.009 0.018 0.036; do python main.py --gpu $gpu --data_dir $data_dir --dataset citeseer --ratio $r; done
for r in 0.0005 0.0025 0.005; do python main.py --gpu $gpu --data_dir $data_dir --dataset arxiv --ratio $r; done
for r in 0.001 0.005 0.01; do python main.py --gpu $gpu --data_dir $data_dir --dataset flickr --ratio $r; done
for r in 0.0005 0.001 0.002; do python main.py --gpu $gpu --data_dir $data_dir --dataset reddit --ratio $r; done
