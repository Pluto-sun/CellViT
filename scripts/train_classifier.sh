#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 训练基础模型（仅使用encoder）
PYTHONPATH=. python models/classification/train.py \
    --data SAHU \
    --root_path dataset/SAHU/ \
    --win_size 72 \
    --step 72 \
    --gaf_method summation \
    --model_type base \
    --model_arch encoder \
    --num_classes 3 \
    --drop_rate 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path_rate 0.1 \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --log_dir logs/classification/encoder_base \
    --save_dir checkpoints/classification/encoder_base

# 训练基础模型（使用encoder和decoder）
PYTHONPATH=. python models/classification/train.py \
    --data SAHU \
    --root_path dataset/SAHU/ \
    --win_size 72 \
    --step 72 \
    --gaf_method summation \
    --model_type base \
    --model_arch decoder \
    --num_classes 3 \
    --drop_rate 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path_rate 0.1 \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --log_dir logs/classification/decoder_base \
    --save_dir checkpoints/classification/decoder_base

# ViT-256模型训练（仅使用encoder，需要预训练模型）
# PYTHONPATH=. python models/classification/train.py \
#     --data SAHU \
#     --root_path dataset/ \
#     --win_size 72 \
#     --step 1 \
#     --gaf_method summation \
#     --model_type 256 \
#     --model_arch encoder \
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
#     --log_dir logs/classification/encoder_vit256 \
#     --save_dir checkpoints/classification/encoder_vit256

# ViT-256模型训练（使用encoder和decoder，需要预训练模型）
# PYTHONPATH=. python models/classification/train.py \
#     --data SAHU \
#     --root_path dataset/ \
#     --win_size 72 \
#     --step 1 \
#     --gaf_method summation \
#     --model_type 256 \
#     --model_arch decoder \
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
#     --log_dir logs/classification/decoder_vit256 \
#     --save_dir checkpoints/classification/decoder_vit256

# SAM模型训练（仅使用encoder，需要预训练模型）
# PYTHONPATH=. python models/classification/train.py \
#     --data SAHU \
#     --root_path dataset/ \
#     --win_size 72 \
#     --step 1 \
#     --gaf_method summation \
#     --model_type sam \
#     --model_arch encoder \
#     --pretrained_path path/to/sam_model.pth \
#     --vit_structure SAM-B \
#     --num_classes 3 \
#     --drop_rate 0.1 \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --weight_decay 1e-4 \
#     --log_dir logs/classification/encoder_sam \
#     --save_dir checkpoints/classification/encoder_sam

# SAM模型训练（使用encoder和decoder，需要预训练模型）
# PYTHONPATH=. python models/classification/train.py \
#     --data SAHU \
#     --root_path dataset/ \
#     --win_size 72 \
#     --step 1 \
#     --gaf_method summation \
#     --model_type sam \
#     --model_arch decoder \
#     --pretrained_path path/to/sam_model.pth \
#     --vit_structure SAM-B \
#     --num_classes 3 \
#     --drop_rate 0.1 \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --weight_decay 1e-4 \
#     --log_dir logs/classification/decoder_sam \
#     --save_dir checkpoints/classification/decoder_sam 