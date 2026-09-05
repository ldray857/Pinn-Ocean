# -*- coding: utf-8 -*-
"""
Configuration file for Pinn-Ocean (Swin-Ocean-PINN)
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    # Input surface dynamics channels:
    # [SST, SLA, SSS, Wind_U, Wind_V, Lon, Lat, Month] -> 8 channels
    in_channels: int = 8
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 4  # Window size for shifted-window attention
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    # PINN Decoder Head
    physics_hidden_dim: int = 256
    out_dim: int = 2  # [Potential Temperature, Practical Salinity]


@dataclass
class PhysicsConfig:
    # Monotonic temperature vertical gradient constraint
    temp_grad_threshold: float = 0.005  # threshold epsilon for dT/dz <= epsilon
    enable_temp_grad_loss: bool = True
    
    # Seawater stratification stability (d_rho / dz >= 0)
    enable_density_loss: bool = True
    
    # Adaptive multi-objective loss balance
    init_log_var_data: float = 0.0
    init_log_var_phy: float = 0.0
    adaptive_weighting: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 4
    num_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 8
    sampling_points: int = 800  # Latent sampling points per batch for VRAM efficiency
    clip_grad_norm: float = 5.0
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True


@dataclass
class DataConfig:
    sla_path: str = "../../实验一/data/input_sla_nwp.nc"
    gt_path: str = "../../实验一/data/gt_3d_temp_sal_nwp.nc"
    train_ratio: float = 0.8
    val_ratio: float = 0.15
    test_ratio: float = 0.05
