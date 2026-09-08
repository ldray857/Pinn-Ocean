# -*- coding: utf-8 -*-
"""
Swin-Ocean-PINN Architecture
Coupling Shifted Window Self-Attention (Swin-Unet) with Physics-Informed
Continuous Depth Coordinate Representation for 3-D Ocean Thermohaline Reconstruction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_blocks import (
    PatchEmbed,
    SwinTransformerBlock,
    PatchMerging,
    to_2tuple
)


class DepthFourierEmbedding(nn.Module):
    """
    Multi-Scale Fourier Coordinate Embedding for Ocean Vertical Profiles.
    Overcomes coordinate spectral bias by expanding raw continuous vertical depth z (in meters)
    into a rich harmonic representation.
    Combines:
    - Linear normalized depth: z / z_max in [0, 1]
    - Oceanographic logarithmic depth: log1p(z / z_scale_log) / log(1 + z_max / z_scale_log) in [0, 1]
    - Multi-scale harmonic Fourier frequency bands: sin/cos across K octaves
    """
    def __init__(self, num_freqs=8, out_dim=64, z_max=1000.0, z_scale_log=15.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = out_dim
        self.z_max = z_max
        self.z_scale_log = z_scale_log

        # Frequencies: 2^0, 2^1, ..., 2^(num_freqs-1)
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

        # Raw features: z_lin (1) + z_log (1) + 2*K (sines/cosines of z_lin) + 2*K (sines/cosines of z_log)
        in_features = 2 + 4 * num_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, z):
        """
        Args:
            z: Depth in meters (positive downwards), shape (..., 1) or (D,)
        Returns:
            embed: Embedded depth coordinates with shape (..., out_dim)
        """
        if z.shape[-1] != 1:
            z_in = z.unsqueeze(-1)
        else:
            z_in = z

        z_safe = torch.clamp(z_in, min=0.0)
        z_lin = z_safe / self.z_max
        denom = math.log(1.0 + self.z_max / self.z_scale_log)
        z_log = torch.log1p(z_safe / self.z_scale_log) / denom

        pi = math.pi
        args_lin = z_lin * self.freq_bands * pi
        args_log = z_log * self.freq_bands * pi

        sin_lin = torch.sin(args_lin)
        cos_lin = torch.cos(args_lin)
        sin_log = torch.sin(args_log)
        cos_log = torch.cos(args_log)

        raw_feat = torch.cat([z_lin, z_log, sin_lin, cos_lin, sin_log, cos_log], dim=-1)
        return self.proj(raw_feat)


class SwinOceanPINN(nn.Module):
    """
    Coupled Swin-Ocean-PINN Architecture:
    1. 2-D Surface Multi-Source Dynamics Encoder: Shifted Window Attention (Swin-Unet)
    2. DeepONet Operator Branch-Trunk Coordinate Fusion (No Dropout, Smooth SiLU)
    3. Multi-Scale Fourier Coordinate Depth Embedding
    4. Decoupled Dual-Branch Decoders:
       - Dedicated Temperature Head (thermocline monotonic structure)
       - Dedicated Salinity Head (S-shaped halocline: subtropical maximum, intermediate minimum)
    """
    def __init__(self, in_channels=8, embed_dim=96, window_size=4,
                 physics_hidden_dim=256, depth_embed_dim=64, out_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.physics_hidden_dim = physics_hidden_dim
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

        # 3. Multi-Scale Fourier Depth Embedding Module
        self.depth_embedder = DepthFourierEmbedding(
            num_freqs=8, out_dim=depth_embed_dim, z_max=1000.0, z_scale_log=15.0
        )

        # 4. DeepONet Operator Fusion Network
        # Branch Network (Processes Sea Surface Tokens)
        self.branch_net = nn.Sequential(
            nn.Linear(embed_dim, physics_hidden_dim),
            nn.SiLU(),
            nn.Linear(physics_hidden_dim, physics_hidden_dim)
        )

        # Trunk Network (Processes Depth Coordinate Features)
        self.trunk_net = nn.Sequential(
            nn.Linear(depth_embed_dim, physics_hidden_dim),
            nn.SiLU(),
            nn.Linear(physics_hidden_dim, physics_hidden_dim)
        )

        # Bilinear & Residual Operator Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(physics_hidden_dim, physics_hidden_dim),
            nn.SiLU()
        )

        # 5. Decoupled Dual-Branch Oceanographic Heads
        # Dedicated Temperature Head (Monotonic Thermocline)
        self.temp_head = nn.Sequential(
            nn.Linear(physics_hidden_dim, physics_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(physics_hidden_dim // 2, 1)
        )

        # Dedicated Salinity Head (Higher capacity for non-monotonic S-shaped halocline)
        self.sal_head = nn.Sequential(
            nn.Linear(physics_hidden_dim, physics_hidden_dim),
            nn.SiLU(),
            nn.Linear(physics_hidden_dim, physics_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(physics_hidden_dim // 2, 1)
        )

    def forward(self, x, z, sample_idx=None):
        """
        Forward pass.
        
        Args:
            x: Sea surface multi-source features (B, 8, H, W)
            z: Continuous vertical depth coordinates in meters (D,) or (B, S, D, 1), requires_grad=True
            sample_idx: Optional 1-D Tensor of spatial indices for random sampling
            
        Returns:
            preds: (B, 2, D, S) or (B, 2, D, H, W) reconstructed thermohaline field
        """
        B, C, H, W = x.shape
        D = z.shape[2] if z.dim() == 4 else z.shape[0]

        # 1. Surface Spatial Feature Extraction
        feat_map = self.stem(x)  # (B, embed_dim, H, W)

        # Dynamically adapt input resolution for Swin Blocks if needed
        if (H, W) != self.stage1_block1.input_resolution:
            self.stage1_block1.input_resolution = (H, W)
            self.stage1_block2.input_resolution = (H, W)

        # Flatten spatial dimensions into Token sequence: (B, H*W, embed_dim)
        tokens = feat_map.flatten(2).permute(0, 2, 1)

        # Swin Attention processing with explicit input resolution
        tokens = self.stage1_block1(tokens, input_resolution=(H, W))
        tokens = self.stage1_block2(tokens, input_resolution=(H, W))
        tokens = self.norm(tokens)

        # 2. Spatial Sampling branch for memory efficiency
        if sample_idx is not None:
            selected_tokens = tokens[:, sample_idx, :]  # (B, S, embed_dim)
            S = selected_tokens.shape[1]
        else:
            selected_tokens = tokens
            S = H * W

        # 3. DeepONet Branch & Trunk Computation
        # Branch features from surface tokens: (B, S, physics_hidden_dim) -> (B, S, 1, physics_hidden_dim)
        h_branch = self.branch_net(selected_tokens).unsqueeze(2)

        # Trunk features from continuous depth coordinates
        if z.dim() == 1:
            feat_z = self.depth_embedder(z.view(1, 1, D, 1))  # (1, 1, D, depth_embed_dim)
            h_trunk = self.trunk_net(feat_z)                  # (1, 1, D, physics_hidden_dim)
        elif z.dim() == 4:
            if z.shape[0] != B or z.shape[1] != S:
                z_pts = z.expand(B, S, D, 1)
            else:
                z_pts = z
            feat_z = self.depth_embedder(z_pts)  # (B, S, D, depth_embed_dim)
            h_trunk = self.trunk_net(feat_z)     # (B, S, D, physics_hidden_dim)
        else:
            feat_z = self.depth_embedder(z.unsqueeze(-1))
            h_trunk = self.trunk_net(feat_z)

        # 4. Interactive Bilinear Fusion & Dual-Branch Predictions
        # Auto-chunk over spatial tokens when S is large to prevent VRAM spikes during full-grid inference
        if S > 2048:
            preds_list = []
            chunk_size = 2048
            for start in range(0, S, chunk_size):
                end = min(start + chunk_size, S)
                hb_chunk = h_branch[:, start:end, :, :]
                ht_chunk = h_trunk if h_trunk.shape[1] == 1 else h_trunk[:, start:end, :, :]
                fused_chunk = self.fusion_layer(hb_chunk * ht_chunk + hb_chunk + ht_chunk)
                tp_chunk = self.temp_head(fused_chunk)
                sp_chunk = self.sal_head(fused_chunk)
                p_chunk = torch.cat([tp_chunk, sp_chunk], dim=-1).permute(0, 3, 2, 1)
                preds_list.append(p_chunk)
            preds = torch.cat(preds_list, dim=-1)
        else:
            fused_state = self.fusion_layer(h_branch * h_trunk + h_branch + h_trunk)
            temp_pred = self.temp_head(fused_state)
            sal_pred = self.sal_head(fused_state)
            preds = torch.cat([temp_pred, sal_pred], dim=-1).permute(0, 3, 2, 1)

        if sample_idx is None:
            preds = preds.view(B, self.out_dim, D, H, W)

        return preds
