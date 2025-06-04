from typing import List, Union
from pathlib import Path
import torch
import torch.nn as nn
from functools import partial

from models.segmentation.cell_segmentation.utils import ViTCellViT, ViTCellViTDeit
from models.classification.cellvit_classifier import CellViTClassifier, CellViT256Classifier, CellViTSAMClassifier

class CellViTClassifierWithDecoder(CellViTClassifier):
    """CellViT-based classifier using both encoder and decoder features
    
    This version uses both the global class token and decoder features for classification,
    which can capture both global and local features.
    
    Args:
        num_classes (int): Number of output classes
        embed_dim (int): Embedding dimension of backbone ViT
        input_channels (int): Number of input channels
        depth (int): Depth of the backbone ViT
        num_heads (int): Number of heads of the backbone ViT
        extract_layers (List[int]): List of Transformer Blocks whose outputs should be returned
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone ViT. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone ViT. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection. Defaults to 0.
    """
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List[int],
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            input_channels=input_channels,
            depth=depth,
            num_heads=num_heads,
            extract_layers=extract_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Decoder for local features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 64, 512),  # Combined global and local features
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x (torch.Tensor): Input images in BCHW format
            
        Returns:
            torch.Tensor: Classification logits
        """
        # Get encoder outputs
        _, class_token, features = self.encoder(x)
        
        # Process local features through decoder
        local_features = self.decoder(features[-1])  # Use last layer features
        local_features = local_features.view(local_features.size(0), -1)  # Flatten
        
        # Combine global and local features
        combined_features = torch.cat([class_token, local_features], dim=1)
        
        # Process through classifier
        logits = self.classifier(combined_features)
        
        return logits

class CellViT256ClassifierWithDecoder(CellViT256Classifier):
    """CellViT classifier with ViT-256 backbone settings and decoder"""
    
    def __init__(
        self,
        model256_path: Union[Path, str],
        num_classes: int,
        input_channels: int = 3,  # 默认值为3，但允许修改
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        self.patch_size = 16
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.extract_layers = [3, 6, 9, 12]
        self.input_channels = input_channels  # 使用传入的通道数
        
        super().__init__(
            model256_path=model256_path,
            num_classes=num_classes,
            input_channels=input_channels,  # 传递通道数到父类
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Override classifier with decoder version
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, num_classes)
        )

class CellViTSAMClassifierWithDecoder(CellViTSAMClassifier):
    """CellViT classifier with SAM backbone settings and decoder"""
    
    def __init__(
        self,
        model_path: Union[Path, str],
        num_classes: int,
        input_channels: int = 3,  # 默认值为3，但允许修改
        vit_structure: str = "SAM-B",
        drop_rate: float = 0,
    ):
        self.input_channels = input_channels  # 使用传入的通道数
        
        super().__init__(
            model_path=model_path,
            num_classes=num_classes,
            input_channels=input_channels,  # 传递通道数到父类
            vit_structure=vit_structure,
            drop_rate=drop_rate,
        )
        
        # Override classifier with decoder version
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.prompt_embed_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.prompt_embed_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, num_classes)
        ) 