# -*- coding: utf-8 -*-
"""
Baseline & Ablation Models for Pinn-Ocean Superiority Benchmark
Provides classical and deep learning comparison models:
1. TrilinearBaseline3D: Classical 3-D geophysical linear interpolation
2. PureDataCNN3D: Traditional multi-layer 2D-CNN + MLP decoder without physics
3. PureSwinAblation: Swin Transformer architecture without physics-informed constraints (MSE-only ablation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.interpolate import RegularGridInterpolator


class TrilinearBaseline3D:
    """
    Classical Geophysical 3-D Trilinear Interpolation Baseline.
    Represents the standard numerical interpolation approach used in GIS and oceanography.
    """
    def __init__(self, depths: np.ndarray, lats: np.ndarray, lons: np.ndarray):
        self.depths = np.asarray(depths, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.lons = np.asarray(lons, dtype=np.float64)

    def predict(
        self,
        coarse_field_3d: np.ndarray,
        target_depths: np.ndarray,
        target_lats: np.ndarray,
        target_lons: np.ndarray
    ) -> np.ndarray:
        """
        Linearly interpolates 3-D field across depth, latitude, and longitude.
        """
        rgi = RegularGridInterpolator(
            (self.depths, self.lats, self.lons),
            coarse_field_3d,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        mesh_z, mesh_y, mesh_x = np.meshgrid(target_depths, target_lats, target_lons, indexing='ij')
        pts = np.stack([mesh_z.ravel(), mesh_y.ravel(), mesh_x.ravel()], axis=-1)
        result = rgi(pts).reshape(len(target_depths), len(target_lats), len(target_lons))
        return result.astype(np.float32)


class PureDataCNN3D(nn.Module):
    """
    Standard Pure Data-Driven Convolutional Neural Network (No Physics).
    Represents conventional black-box CNN architectures widely applied in ocean remote sensing.
    Has limited local receptive field and lacks physical conservation / stability regularizations.
    """
    def __init__(self, in_channels: int = 8, hidden_dim: int = 96, out_dim: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # 4-layer 2D CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Depth MLP (linear projection of depth scalar)
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim)
        )

        # Output head (concatenated spatial feature + depth feature)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, sample_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, 8, H, W)
            z: (D,) in meters
        Returns:
            preds: (B, 2, D, H, W) or (B, 2, D, S)
        """
        B, C, H, W = x.shape
        D = z.shape[0] if z.dim() == 1 else z.shape[2]

        feat_2d = self.encoder(x)  # (B, hidden_dim, H, W)
        tokens = feat_2d.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)

        if sample_idx is not None:
            tokens = tokens[:, sample_idx, :]
        S = tokens.shape[1]

        # Process depth (normalized by 1000m)
        z_norm = (z.view(1, 1, D, 1) / 1000.0).expand(B, S, D, 1)
        z_feat = self.depth_mlp(z_norm)  # (B, S, D, hidden_dim)

        # Broadcast tokens to depth
        tokens_exp = tokens.unsqueeze(2).expand(B, S, D, self.hidden_dim)

        # Concatenate
        fused = torch.cat([tokens_exp, z_feat], dim=-1)  # (B, S, D, 2*hidden_dim)
        out = self.head(fused)  # (B, S, D, 2)
        preds = out.permute(0, 3, 2, 1)  # (B, 2, D, S)

        if sample_idx is None:
            preds = preds.view(B, self.out_dim, D, H, W)

        return preds


class PureSwinAblation(nn.Module):
    """
    Swin Transformer without Physical Constraints (Ablation Benchmark).
    Shares the Swin shifted-window spatial self-attention and Fourier depth representation
    with Swin-Ocean-PINN, but was optimized strictly under unconstrained empirical MSE data loss.
    Serves as the rigorous ablation baseline to demonstrate the precise marginal benefit
    of the active TEOS-10 buoyancy frequency and stratification stability constraints.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor, z: torch.Tensor, sample_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.base_model(x, z, sample_idx=sample_idx)
