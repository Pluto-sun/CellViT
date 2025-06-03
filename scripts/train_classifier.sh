#!/bin/bash

# 基础模型训练
python models/classification/train.py \
    --data SAHU \
    --root_path dataset/ \
    --seq_len 72 \
    --step 1 \
    --gaf_method summation \
    --model_type base \
    --num_classes 3 \
    --drop_rate 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path_rate 0.1 \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --log_dir logs/classification/base \
    --save_dir checkpoints/classification/base

# ViT-256模型训练（需要预训练模型）
# python models/classification/train.py \
#     --data SAHU \
#     --root_path dataset/ \
#     --seq_len 72 \
#     --step 1 \
#     --gaf_method summation \
#     --model_type 256 \
#     --pretrained_path path/to/vit256_model.pth \
#     --num_classes 3 \
#     --drop_rate 0.1 \
#     --attn_drop_rate 0.1 \
#     --drop_path_rate 0.1 \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --weight_decay 1e-4 \
#     --log_dir logs/classification/vit256 \
#     --save_dir checkpoints/classification/vit256

# SAM模型训练（需要预训练模型）
# python models/classification/train.py \
#     --data SAHU \
#     --root_path dataset/ \
#     --seq_len 72 \
#     --step 1 \
#     --gaf_method summation \
#     --model_type sam \
#     --pretrained_path path/to/sam_model.pth \
#     --vit_structure SAM-B \
#     --num_classes 3 \
#     --drop_rate 0.1 \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --weight_decay 1e-4 \
#     --log_dir logs/classification/sam \
#     --save_dir checkpoints/classification/sam 