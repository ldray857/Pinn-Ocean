# -*- coding: utf-8 -*-
"""
Swin-Ocean-PINN Architecture
Coupling Shifted Window Self-Attention (Swin-Unet) with Physics-Informed
Continuous Depth Coordinate Representation for 3-D Ocean Thermohaline Reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_blocks import (
    PatchEmbed,
    SwinTransformerBlock,
    PatchMerging,
    to_2tuple
)


class SwinOceanPINN(nn.Module):
    """
    Swin-Ocean-PINN Model:
    1. 2-D Surface Multi-Source Dynamics Encoder: Shifted Window Attention
    2. Continuous Spatial-Vertical Fusion
    3. Physics-Informed Neural Network (PINN) Decoder Head:
       Maps [Token_feat, z] -> [T_hat, S_hat] with continuous differentiability.
    """
    def __init__(self, in_channels=8, embed_dim=96, window_size=4,
                 physics_hidden_dim=256, out_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.out_dim = out_dim

        # 1. Surface Dynamics Spatial Feature Extractor
        # Conv-based feature stem to preserve fine spatial details
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # 2. Hierarchical Swin Attention Blocks
        # For typical regional ocean grids (e.g. 40x40 to 64x64)
        self.stage1_block1 = SwinTransformerBlock(
            dim=embed_dim, input_resolution=(40, 40), num_heads=4,
            window_size=window_size, shift_size=0
        )
        self.stage1_block2 = SwinTransformerBlock(
            dim=embed_dim, input_resolution=(40, 40), num_heads=4,
            window_size=window_size, shift_size=window_size // 2
        )

        self.norm = nn.LayerNorm(embed_dim)

        # 3. Continuous Depth Coordinate PINN Decoder Head
        # Smooth Tanh activation guarantees high-order differentiability for Autograd
        self.physics_head = nn.Sequential(
            nn.Linear(embed_dim + 1, physics_hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(physics_hidden_dim, physics_hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(physics_hidden_dim, physics_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(physics_hidden_dim // 2, out_dim)
        )

    def forward(self, x, z, sample_idx=None):
        """
        Forward pass.
        
        Args:
            x: Sea surface multi-source features (B, 8, H, W)
            z: Continuous vertical depth coordinates (D,), requires_grad=True
            sample_idx: Optional 1-D Tensor of spatial indices for random sampling
            
        Returns:
            out: (B, 2, D, S) or (B, 2, D, H, W) reconstructed thermohaline field
        """
        B, C, H, W = x.shape
        D = z.shape[0]

        # 1. Surface Spatial Feature Extraction
        feat_map = self.stem(x)  # (B, embed_dim, H, W)

        # Dynamically adapt input resolution for Swin Blocks if needed
        if (H, W) != self.stage1_block1.input_resolution:
            self.stage1_block1.input_resolution = (H, W)
            self.stage1_block2.input_resolution = (H, W)

        # Flatten spatial dimensions into Token sequence: (B, H*W, embed_dim)
        tokens = feat_map.flatten(2).permute(0, 2, 1)

        # Swin Attention processing
        tokens = self.stage1_block1(tokens)
        tokens = self.stage1_block2(tokens)
        tokens = self.norm(tokens)

        # 2. Spatial Sampling branch for memory efficiency
        if sample_idx is not None:
            selected_tokens = tokens[:, sample_idx, :]  # (B, S, embed_dim)
            S = selected_tokens.shape[1]
        else:
            selected_tokens = tokens
            S = H * W

        # 3. Continuous Coordinate Fusion
        # Expand spatial tokens to (B, S, D, embed_dim)
        feat_expanded = selected_tokens.unsqueeze(2).expand(-1, -1, D, -1)
        # Expand continuous depth z to (B, S, D, 1)
        z_expanded = z.view(1, 1, D, 1).expand(B, S, D, 1)

        # Concatenate features and depth coordinate
        combined_coords = torch.cat([feat_expanded, z_expanded], dim=-1)  # (B, S, D, embed_dim + 1)

        # Pass through PINN Physics Head
        preds = self.physics_head(combined_coords)  # (B, S, D, 2)

        # Permute to oceanographic standard: (B, out_dim=2, D, S)
        preds = preds.permute(0, 3, 2, 1)

        if sample_idx is None:
            # Restore 3-D volumetric grid: (B, 2, D, H, W)
            preds = preds.view(B, self.out_dim, D, H, W)

        return preds
