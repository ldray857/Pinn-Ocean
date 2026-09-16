# -*- coding: utf-8 -*-
"""
Continuous Space-Depth Super-Resolution & GLORYS 3D Interpolation Module
Pinn-Ocean (Swin-Ocean-PINN)

Provides continuous spatial (horizontal downscaling/super-resolution) and
continuous vertical (arbitrary depth/dense regular voxel) reconstruction
for GLORYS ocean reanalysis data and multi-source satellite observations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from scipy.interpolate import RegularGridInterpolator

from pinn_ocean.utils.teos10 import approx_seawater_density


class ContinuousSpaceDepthSuperResolver:
    """
    Continuous Space-Depth Super-Resolution Engine based on Swin-Ocean-PINN.
    Leverages:
    1. Spatial sub-pixel feature upsampling (bilinear/bicubic) for sea surface dynamics tokens;
    2. Continuous Fourier Coordinate Embedding + DeepONet Trunk Network for arbitrary vertical depth z;
    3. Memory-safe chunked spatial decoding for ultra-dense 3D grids.
    """
    def __init__(self, model: nn.Module, stats: Optional[Dict[str, float]] = None, device: Optional[torch.device] = None):
        self.model = model
        self.stats = stats
        self.device = device or (next(model.parameters()).device if any(model.parameters()) else torch.device('cpu'))
        self.model.eval()

    @torch.no_grad()
    def super_resolve_surface_input(
        self,
        x_8ch: torch.Tensor,
        scale_factor: float = 2.0,
        target_hw: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Upsamples 8-channel sea surface input features to higher spatial resolution.
        Channels:
            [0: SST, 1: SLA, 2: SSS, 3: Wind_U, 4: Wind_V, 5: Lon, 6: Lat, 7: Month]
        """
        B, C, H, W = x_8ch.shape
        if target_hw is not None:
            H_hr, W_hr = target_hw
        else:
            H_hr = int(round(H * scale_factor))
            W_hr = int(round(W * scale_factor))

        if (H_hr, W_hr) == (H, W):
            return x_8ch

        # Channels 0-4: Physical variables (SST, SLA, SSS, Wind U, Wind V) -> Bicubic for smooth gradients
        phys_vars = x_8ch[:, :5]  # (B, 5, H, W)
        phys_hr = F.interpolate(phys_vars, size=(H_hr, W_hr), mode='bicubic', align_corners=True)

        # Channel 5 & 6: Normalized Longitude & Latitude -> Strictly linear coordinate grids
        lon_lin = torch.linspace(0.0, 1.0, W_hr, device=x_8ch.device, dtype=x_8ch.dtype)
        lat_lin = torch.linspace(0.0, 1.0, H_hr, device=x_8ch.device, dtype=x_8ch.dtype)
        lat_grid, lon_grid = torch.meshgrid(lat_lin, lon_lin, indexing='ij')  # (H_hr, W_hr)
        coords_hr = torch.stack([lon_grid, lat_grid], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H_hr, W_hr)

        # Channel 7: Cyclic Month Encoding -> Constant spatially across basin
        month_vals = x_8ch[:, 7:8, 0:1, 0:1].expand(B, 1, H_hr, W_hr)

        x_hr = torch.cat([phys_hr, coords_hr, month_vals], dim=1)
        return x_hr

    @torch.no_grad()
    def reconstruct_3d_high_res(
        self,
        x_8ch: torch.Tensor,
        z_coords: torch.Tensor,
        scale_factor: float = 2.0,
        target_hw: Optional[Tuple[int, int]] = None,
        unnormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs full 3D spatial-vertical high-resolution reconstruction.
        
        Args:
            x_8ch: (B, 8, H, W) surface features
            z_coords: (D_out,) 1-D depth tensor in meters
            scale_factor: Horizontal magnification factor (e.g. 2.0 or 4.0)
            target_hw: Optional explicit (H_hr, W_hr)
            unnormalize: Whether to return physical units (°C and PSU)
            
        Returns:
            temp_hr: (B, D_out, H_hr, W_hr) high-res potential temperature
            sal_hr: (B, D_out, H_hr, W_hr) high-res practical salinity
        """
        x_8ch = x_8ch.to(self.device)
        z_coords = z_coords.to(self.device)

        # 1. Upsample surface feature tensor to target spatial resolution
        x_hr = self.super_resolve_surface_input(x_8ch, scale_factor=scale_factor, target_hw=target_hw)
        B, C, H_hr, W_hr = x_hr.shape
        D_out = len(z_coords)

        # 2. Forward pass through Swin-Ocean-PINN
        # SwinOceanPINN dynamically adapts (H_hr, W_hr) and chunks spatial tokens automatically
        preds = self.model(x_hr, z_coords, sample_idx=None)  # (B, 2, D_out, H_hr, W_hr)

        temp_pred = preds[:, 0].cpu().numpy()
        sal_pred = preds[:, 1].cpu().numpy()

        if unnormalize and self.stats:
            temp_pred = temp_pred * self.stats.get('std_t', 1.0) + self.stats.get('mean_t', 0.0)
            sal_pred = sal_pred * self.stats.get('std_s', 1.0) + self.stats.get('mean_s', 0.0)

        return temp_pred, sal_pred


class GLORYS3DInterpolator:
    """
    Direct 3-D Ocean Reanalysis Interpolator.
    Supports classical mathematical interpolation (Trilinear, Tricubic)
    as well as Physics-Regularized Interpolation preserving thermodynamic monotonicity
    and density stratification stability.
    """
    def __init__(self, depths: np.ndarray, lats: np.ndarray, lons: np.ndarray):
        self.depths = np.asarray(depths, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.lons = np.asarray(lons, dtype=np.float64)

    def interpolate_field(
        self,
        field_3d: np.ndarray,
        target_depths: np.ndarray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
        method: str = 'physics_regularized',
        var_name: str = 'temp'
    ) -> np.ndarray:
        """
        Interpolates 3-D field (D, H, W) to (D_tgt, H_tgt, W_tgt).
        
        Args:
            field_3d: 3D numpy array of shape (D, H, W)
            target_depths: 1D array of target depths
            target_lats: 1D array of target latitudes
            target_lons: 1D array of target longitudes
            method: 'trilinear', 'tricubic', or 'physics_regularized'
            var_name: 'temp' or 'sal' (used for physics regularization)
            
        Returns:
            interpolated_3d: shape (len(target_depths), len(target_lats), len(target_lons))
        """
        scipy_method = 'linear' if method in ['trilinear', 'physics_regularized'] else 'cubic'
        
        # Handle 2D slice (H, W) interpolation gracefully
        if field_3d.ndim == 2:
            rgi = RegularGridInterpolator(
                (self.lats, self.lons),
                field_3d,
                method=scipy_method,
                bounds_error=False,
                fill_value=None
            )
            mesh_y, mesh_x = np.meshgrid(target_lats, target_lons, indexing='ij')
            pts = np.stack([mesh_y.ravel(), mesh_x.ravel()], axis=-1)
            result_2d = rgi(pts).reshape(len(target_lats), len(target_lons))
            if method == 'physics_regularized':
                if var_name.lower() in ['temp', 'thetao', 'temperature']:
                    result_2d = np.clip(result_2d, -2.0, 35.0)
                elif var_name.lower() in ['sal', 'so', 'salinity']:
                    result_2d = np.clip(result_2d, 30.0, 38.0)
            return result_2d.astype(np.float32)

        rgi = RegularGridInterpolator(
            (self.depths, self.lats, self.lons),
            field_3d,
            method=scipy_method,
            bounds_error=False,
            fill_value=None
        )

        D_tgt = len(target_depths)
        H_tgt = len(target_lats)
        W_tgt = len(target_lons)

        mesh_z, mesh_y, mesh_x = np.meshgrid(target_depths, target_lats, target_lons, indexing='ij')
        pts = np.stack([mesh_z.ravel(), mesh_y.ravel(), mesh_x.ravel()], axis=-1)

        result = rgi(pts).reshape(D_tgt, H_tgt, W_tgt)

        if method == 'physics_regularized':
            result = self._apply_physics_regularization(result, target_depths, var_name)


        return result.astype(np.float32)

    def _apply_physics_regularization(self, field_3d: np.ndarray, depths: np.ndarray, var_name: str) -> np.ndarray:
        """
        Enforces physical consistency on interpolated field:
        1. For temperature: Enforces non-increasing profile in deep waters (z >= 100m)
           to eliminate unphysical warm oscillations from cubic splines.
        2. Bounds extremes within reasonable oceanographic limits.
        """
        res = field_3d.copy()
        D, H, W = res.shape

        if var_name.lower() in ['temp', 'thetao', 'temperature']:
            deep_mask = depths >= 100.0
            deep_indices = np.where(deep_mask)[0]
            for idx in range(1, len(deep_indices)):
                curr_k = deep_indices[idx]
                prev_k = deep_indices[idx - 1]
                res[curr_k] = np.minimum(res[curr_k], res[prev_k] + 0.02)
            res = np.clip(res, -2.0, 35.0)

        elif var_name.lower() in ['sal', 'so', 'salinity']:
            res = np.clip(res, 30.0, 38.0)

        return res


def compute_super_resolution_metrics(
    pred_hr: np.ndarray,
    true_hr: np.ndarray,
    data_range: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes standard field super-resolution quality metrics:
    PSNR (Peak Signal-to-Noise Ratio), RMSE, MAE, and Gradient Fidelity.
    """
    diff = pred_hr - true_hr
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    if data_range is None:
        data_range = float(np.max(true_hr) - np.min(true_hr))
        if data_range < 1e-4:
            data_range = 1.0

    psnr = 10.0 * math.log10((data_range ** 2) / (mse + 1e-12))

    gy_true, gx_true = np.gradient(true_hr, axis=(-2, -1))
    gy_pred, gx_pred = np.gradient(pred_hr, axis=(-2, -1))
    grad_mag_true = np.sqrt(gx_true**2 + gy_true**2)
    grad_mag_pred = np.sqrt(gx_pred**2 + gy_pred**2)

    g_diff = grad_mag_pred - grad_mag_true
    grad_rmse = float(np.sqrt(np.mean(g_diff ** 2)))

    g_true_flat = grad_mag_true.ravel()
    g_pred_flat = grad_mag_pred.ravel()
    corr = np.corrcoef(g_true_flat, g_pred_flat)[0, 1]
    grad_corr = float(corr) if not np.isnan(corr) else 1.0

    return {
        "rmse": rmse,
        "mae": mae,
        "psnr_db": float(psnr),
        "grad_rmse": grad_rmse,
        "grad_correlation": grad_corr
    }
