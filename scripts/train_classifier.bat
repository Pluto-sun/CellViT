@echo off
REM 训练基础模型（仅使用encoder）
python models/classification/train.py ^
    --data SAHU ^
    --root_path dataset/ ^
    --seq_len 72 ^
    --step 1 ^
    --gaf_method summation ^
    --model_type base ^
    --model_arch encoder ^
    --num_classes 3 ^
    --drop_rate 0.1 ^
    --attn_drop_rate 0.1 ^
    --drop_path_rate 0.1 ^
    --batch_size 32 ^
    --num_workers 4 ^
    --epochs 100 ^
    --lr 1e-4 ^
    --weight_decay 1e-4 ^
    --log_dir logs/classification/encoder_base ^
    --save_dir checkpoints/classification/encoder_base

REM 训练基础模型（使用encoder和decoder）
python models/classification/train.py ^
    --data SAHU ^
    --root_path dataset/ ^
    --seq_len 72 ^
    --step 1 ^
    --gaf_method summation ^
    --model_type base ^
    --model_arch decoder ^
    --num_classes 3 ^
    --drop_rate 0.1 ^
    --attn_drop_rate 0.1 ^
    --drop_path_rate 0.1 ^
    --batch_size 32 ^
    --num_workers 4 ^
    --epochs 100 ^
    --lr 1e-4 ^
    --weight_decay 1e-4 ^
    --log_dir logs/classification/decoder_base ^
    --save_dir checkpoints/classification/decoder_base

REM ViT-256模型训练（仅使用encoder，需要预训练模型）
REM python models/classification/train.py ^
REM     --data SAHU ^
REM     --root_path dataset/ ^
REM     --seq_len 72 ^
REM     --step 1 ^
REM     --gaf_method summation ^
REM     --model_type 256 ^
REM     --model_arch encoder ^
REM     --pretrained_path path/to/vit256_model.pth ^
REM     --num_classes 3 ^
REM     --drop_rate 0.1 ^
REM     --attn_drop_rate 0.1 ^
REM     --drop_path_rate 0.1 ^
REM     --batch_size 32 ^
REM     --num_workers 4 ^
REM     --epochs 100 ^
REM     --lr 1e-4 ^
REM     --weight_decay 1e-4 ^
REM     --log_dir logs/classification/encoder_vit256 ^
REM     --save_dir checkpoints/classification/encoder_vit256

REM ViT-256模型训练（使用encoder和decoder，需要预训练模型）
REM python models/classification/train.py ^
REM     --data SAHU ^
REM     --root_path dataset/ ^
REM     --seq_len 72 ^
REM     --step 1 ^
REM     --gaf_method summation ^
REM     --model_type 256 ^
REM     --model_arch decoder ^
REM     --pretrained_path path/to/vit256_model.pth ^
REM     --num_classes 3 ^
REM     --drop_rate 0.1 ^
REM     --attn_drop_rate 0.1 ^
REM     --drop_path_rate 0.1 ^
REM     --batch_size 32 ^
REM     --num_workers 4 ^
REM     --epochs 100 ^
REM     --lr 1e-4 ^
REM     --weight_decay 1e-4 ^
REM     --log_dir logs/classification/decoder_vit256 ^
REM     --save_dir checkpoints/classification/decoder_vit256

REM SAM模型训练（仅使用encoder，需要预训练模型）
REM python models/classification/train.py ^
REM     --data SAHU ^
REM     --root_path dataset/ ^
REM     --seq_len 72 ^
REM     --step 1 ^
REM     --gaf_method summation ^
REM     --model_type sam ^
REM     --model_arch encoder ^
REM     --pretrained_path path/to/sam_model.pth ^
REM     --vit_structure SAM-B ^
REM     --num_classes 3 ^
REM     --drop_rate 0.1 ^
REM     --batch_size 32 ^
REM     --num_workers 4 ^
REM     --epochs 100 ^
REM     --lr 1e-4 ^
REM     --weight_decay 1e-4 ^
REM     --log_dir logs/classification/encoder_sam ^
REM     --save_dir checkpoints/classification/encoder_sam

REM SAM模型训练（使用encoder和decoder，需要预训练模型）
REM python models/classification/train.py ^
REM     --data SAHU ^
REM     --root_path dataset/ ^
REM     --seq_len 72 ^
REM     --step 1 ^
REM     --gaf_method summation ^
REM     --model_type sam ^
REM     --model_arch decoder ^
REM     --pretrained_path path/to/sam_model.pth ^
REM     --vit_structure SAM-B ^
REM     --num_classes 3 ^
REM     --drop_rate 0.1 ^
REM     --batch_size 32 ^
REM     --num_workers 4 ^
REM     --epochs 100 ^
REM     --lr 1e-4 ^
REM     --weight_decay 1e-4 ^
REM     --log_dir logs/classification/decoder_sam ^
REM     --save_dir checkpoints/classification/decoder_sam

pause 