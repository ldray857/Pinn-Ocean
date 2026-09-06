# -*- coding: utf-8 -*-
"""
Configuration file for Pinn-Ocean (Swin-Ocean-PINN)
Defines model hyperparameters, physics loss weights, training configs, and Open Pacific dataset settings.
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
    # Continuous Depth PINN Decoder Head
    physics_hidden_dim: int = 256
    out_dim: int = 2  # Output variables: [Potential Temperature, Practical Salinity]


@dataclass
class PhysicsConfig:
    # Monotonic temperature vertical gradient constraint (dT/dz <= epsilon)
    temp_grad_threshold: float = 0.005  # threshold epsilon for dT/dz <= epsilon
    enable_temp_grad_loss: bool = True
    
    # Seawater stratification stability and anti-inversion constraint (d_rho / dz >= 0)
    enable_density_loss: bool = True
    
    # Adaptive multi-objective loss balance (homoscedastic uncertainty weighting)
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
    sampling_points: int = 800  # Spatial random sampling points per batch for VRAM efficiency
    clip_grad_norm: float = 5.0
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True


@dataclass
class DataConfig:
    # Local paths for Pacific Ocean datasets
    sla_path: str = "data/pacific_sla_2013_2021.nc"
    gt_path: str = "data/pacific_glorys_3d_temp_sal_2013_2021.nc"
    sst_path: str = "data/pacific_sst_2013_2021.nc"
    sss_path: str = "data/pacific_sss_2013_2021.nc"
    wind_path: str = "data/pacific_wind_2013_2021.nc"
    
    # Open Pacific bounding box (100% pure open ocean, zero land points)
    min_lon: float = 145.0
    max_lon: float = 165.0
    min_lat: float = 30.0
    max_lat: float = 40.0
    start_time: str = "2013-01-01"
    end_time: str = "2021-12-31"
    
    # Temporal train/val/test split ratios (72 months train, 24 months val, 12 months test)
    train_ratio: float = 72.0 / 108.0   # ~0.667
    val_ratio: float = 24.0 / 108.0     # ~0.222
    test_ratio: float = 12.0 / 108.0    # ~0.111



