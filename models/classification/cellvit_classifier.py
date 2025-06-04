from typing import List, Union
from pathlib import Path
import torch
import torch.nn as nn
from functools import partial

from models.segmentation.cell_segmentation.utils import ViTCellViT, ViTCellViTDeit

class CellViTClassifier(nn.Module):
    """CellViT-based classifier for time series classification using GAF-transformed images
    
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
        super().__init__()
        
        self.patch_size = 16
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        # Vision Transformer backbone
        self.encoder = ViTCellViT(
            patch_size=self.patch_size,
            num_classes=0,  # No classification head in encoder
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
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
        # Get class token from encoder
        _, class_token, _ = self.encoder(x)
        
        # Pass through classification head
        logits = self.classifier(class_token)
        
        return logits

class CellViT256Classifier(CellViTClassifier):
    """CellViT classifier with ViT-256 backbone settings"""
    
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
            num_classes=num_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,  # 使用传入的通道数
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        self.model256_path = model256_path
        
    def load_pretrained_encoder(self):
        """Load pretrained ViT-256 encoder weights"""
        state_dict = torch.load(str(self.model256_path), map_location="cpu")
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")

class CellViTSAMClassifier(CellViTClassifier):
    """CellViT classifier with SAM backbone settings"""
    
    def __init__(
        self,
        model_path: Union[Path, str],
        num_classes: int,
        input_channels: int = 3,  # 默认值为3，但允许修改
        vit_structure: str = "SAM-B",
        drop_rate: float = 0,
    ):
        self.input_channels = input_channels  # 使用传入的通道数
        
        if vit_structure.upper() == "SAM-B":
            self.init_vit_b()
        elif vit_structure.upper() == "SAM-L":
            self.init_vit_l()
        elif vit_structure.upper() == "SAM-H":
            self.init_vit_h()
        else:
            raise NotImplementedError("Unknown ViT-SAM backbone structure")
            
        self.mlp_ratio = 4
        self.qkv_bias = True
        
        super().__init__(
            num_classes=num_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,  # 使用传入的通道数
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
        )
        
        self.model_path = model_path
        self.prompt_embed_dim = 256
        
        # Replace encoder with SAM-style encoder
        self.encoder = ViTCellViTDeit(
            extract_layers=self.extract_layers,
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
            in_chans=self.input_channels,  # 使用传入的通道数
        )
        
        # Update classifier input dimension
        self.classifier[0] = nn.Linear(self.prompt_embed_dim, 512)
        
    def load_pretrained_encoder(self):
        """Load pretrained SAM encoder weights"""
        state_dict = torch.load(str(self.model_path), map_location="cpu")
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")
        
    def init_vit_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]
        
    def init_vit_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]
        
    def init_vit_h(self):
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32] 