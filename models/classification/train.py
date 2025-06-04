import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from data_provider.data_factory import data_provider
from models.classification.cellvit_classifier_encoder import (
    CellViTEncoderClassifier,
    CellViT256EncoderClassifier,
    CellViTSAMEncoderClassifier
)
from models.classification.cellvit_classifier_with_decoder import (
    CellViTClassifierWithDecoder,
    CellViT256ClassifierWithDecoder,
    CellViTSAMClassifierWithDecoder
)

def get_args():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--data', type=str, default='SAHU', help='dataset type')
    parser.add_argument('--root_path', type=str, default='dataset/', help='root path of the data')
    parser.add_argument('--win_size', type=int, default=72, help='window size for time series')
    parser.add_argument('--step', type=int, default=1, help='step size for sliding window')
    parser.add_argument('--gaf_method', type=str, default='summation', help='GAF method')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='base', choices=['base', '256', 'sam'], help='model type')
    parser.add_argument('--model_arch', type=str, default='encoder', choices=['encoder', 'decoder'], help='model architecture')
    parser.add_argument('--pretrained_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--vit_structure', type=str, default='SAM-B', help='ViT structure for SAM model')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--input_channels', type=int, default=None, help='number of input channels (if None, will be determined from data)')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.1, help='attention dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='drop path rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs/classification', help='log directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/classification', help='save directory')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/accuracy', 100.*correct/total, global_step)
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Log to tensorboard
    writer.add_scalar('val/loss', total_loss/len(val_loader), epoch)
    writer.add_scalar('val/accuracy', 100.*correct/total, epoch)
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    args = get_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset, train_loader = data_provider(args, 'train')
    val_dataset, val_loader = data_provider(args, 'val')
    
    # 如果未指定输入通道数，从数据集中获取
    if args.input_channels is None:
        # 获取一个样本来确定通道数
        sample_data, _ = next(iter(train_loader))
        args.input_channels = sample_data.shape[1]  # [B, C, H, W] 中的 C
        print(f"自动检测到输入通道数: {args.input_channels}")
    
    # Initialize model
    if args.model_arch == 'encoder':
        if args.model_type == 'base':
            model = CellViTEncoderClassifier(
                num_classes=args.num_classes,
                embed_dim=384,
                input_channels=args.input_channels,  # 使用检测到的通道数
                depth=12,
                num_heads=6,
                extract_layers=[3, 6, 9, 12],
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                drop_path_rate=args.drop_path_rate
            )
        elif args.model_type == '256':
            model = CellViT256EncoderClassifier(
                model256_path=args.pretrained_path,
                num_classes=args.num_classes,
                input_channels=args.input_channels,  # 使用检测到的通道数
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                drop_path_rate=args.drop_path_rate
            )
            model.load_pretrained_encoder()
        elif args.model_type == 'sam':
            model = CellViTSAMEncoderClassifier(
                model_path=args.pretrained_path,
                num_classes=args.num_classes,
                input_channels=args.input_channels,  # 使用检测到的通道数
                vit_structure=args.vit_structure,
                drop_rate=args.drop_rate
            )
            model.load_pretrained_encoder()
    else:  # decoder architecture
        if args.model_type == 'base':
            model = CellViTClassifierWithDecoder(
                num_classes=args.num_classes,
                embed_dim=384,
                input_channels=args.input_channels,
                depth=12,
                num_heads=6,
                extract_layers=[3, 6, 9, 12],
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                drop_path_rate=args.drop_path_rate
            )
        elif args.model_type == '256':
            model = CellViT256ClassifierWithDecoder(
                model256_path=args.pretrained_path,
                num_classes=args.num_classes,
                input_channels=args.input_channels,
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                drop_path_rate=args.drop_path_rate
            )
            model.load_pretrained_encoder()
        elif args.model_type == 'sam':
            model = CellViTSAMClassifierWithDecoder(
                model_path=args.pretrained_path,
                num_classes=args.num_classes,
                input_channels=args.input_channels,
                vit_structure=args.vit_structure,
                drop_rate=args.drop_rate
            )
            model.load_pretrained_encoder()
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Tensorboard writer
    log_dir = os.path.join(args.log_dir, f"{args.model_arch}_{args.model_type}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f"{args.model_arch}_{args.model_type}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            print(f'Saved best model with accuracy: {best_acc:.2f}%')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f"{args.model_arch}_{args.model_type}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, save_path)
    
    writer.close()

if __name__ == '__main__':
    main() 