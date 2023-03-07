#!/bin/bash
# shellcheck disable=SC2164

conda env list
source activate pytorch-1.8-py37

cd "/mnt/YuHe/full_resnet50"
python test_model.py  --is_left --device "cuda:1" --model_path "/mnt/YuHe/data/SDYD/left/classify/weight/v1.17-2023-02-17.pt" --valid_dir "/mnt/YuHe/data/val_data/left/cur-val-data-classify/valid-data"  --threshold 0.85 --batch_size 64
