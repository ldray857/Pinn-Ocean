# -*- coding: utf-8 -*-
"""
Layer-by-Layer Subsurface Horizontal Depth Slices (50m Interval)
Generates high-resolution horizontal spatial maps of ocean temperature and salinity
at discrete 50-meter vertical intervals from sea surface to 1000m depth.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def interpolate_to_target_depths(data_3d, original_depths, target_depths):
    """
    Interpolate 3D volumetric field (D, H, W) along depth axis to exact target depths.
    """
    f = interp1d(original_depths, data_3d, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
    return f(target_depths)


def plot_depth_layers_grid(
    true_3d, pred_3d, lons, lats, original_depths,
    target_depths=np.arange(0, 1050, 50),
    var_name="temperature",
    save_path=None,
    export_individual_dir=None
):
    """
    Plots horizontal contour fields at 50m depth intervals.
    Generates:
    1. A consolidated publication-quality overview grid across depths.
    2. Optional individual 3-panel (True, Pred, Error) figures for each 50m depth layer.
    """
    true_interp = interpolate_to_target_depths(true_3d, original_depths, target_depths)
    pred_interp = interpolate_to_target_depths(pred_3d, original_depths, target_depths)
    error_interp = pred_interp - true_interp

    is_temp = "temp" in var_name.lower()
    unit = "°C" if is_temp else "PSU"
    title_var = "Potential Temperature" if is_temp else "Practical Salinity"
    cmap_field = "turbo" if is_temp else "viridis"
    cmap_err = "RdBu_r"

    vmin_field = np.nanpercentile(true_interp, 1)
    vmax_field = np.nanpercentile(true_interp, 99)
    err_limit = max(0.5 if is_temp else 0.1, np.nanpercentile(np.abs(error_interp), 98))

    # 1. Export individual layer figures for every 50m level (0, 50, 100, ..., 1000m)
    if export_individual_dir is not None:
        os.makedirs(export_individual_dir, exist_ok=True)
        for idx, z_val in enumerate(target_depths):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            
            # Ground Truth
            im0 = axes[0].pcolormesh(lons, lats, true_interp[idx], cmap=cmap_field, vmin=vmin_field, vmax=vmax_field, shading='auto')
            axes[0].set_title(f"GLORYS12V1 Truth @ {int(z_val)}m ({unit})", fontsize=13, fontweight='bold')
            axes[0].set_xlabel("Longitude (°E)", fontsize=11)
            axes[0].set_ylabel("Latitude (°N)", fontsize=11)
            plt.colorbar(im0, ax=axes[0], orientation='vertical', pad=0.02, shrink=0.85)

            # Prediction
            im1 = axes[1].pcolormesh(lons, lats, pred_interp[idx], cmap=cmap_field, vmin=vmin_field, vmax=vmax_field, shading='auto')
            axes[1].set_title(f"PINN Prediction @ {int(z_val)}m ({unit})", fontsize=13, fontweight='bold')
            axes[1].set_xlabel("Longitude (°E)", fontsize=11)
            plt.colorbar(im1, ax=axes[1], orientation='vertical', pad=0.02, shrink=0.85)

            # Error
            im2 = axes[2].pcolormesh(lons, lats, error_interp[idx], cmap=cmap_err, vmin=-err_limit, vmax=err_limit, shading='auto')
            rmse_layer = np.sqrt(np.mean(error_interp[idx]**2))
            axes[2].set_title(f"Residual (Pred - True) [RMSE: {rmse_layer:.3f}{unit}]", fontsize=13, fontweight='bold')
            axes[2].set_xlabel("Longitude (°E)", fontsize=11)
            plt.colorbar(im2, ax=axes[2], orientation='vertical', pad=0.02, shrink=0.85)

            fig.suptitle(f"Northwest Pacific {title_var} Horizontal Slice: Depth = {int(z_val)} m", fontsize=15, fontweight='bold')
            ind_path = os.path.join(export_individual_dir, f"{var_name}_depth_{int(z_val):04d}m.png")
            fig.savefig(ind_path, dpi=180)
            plt.close(fig)

    # 2. Consolidated multi-layer overview figure (Key representative 50m levels)
    key_depths = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
    n_rows = len(key_depths)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.2 * n_rows), constrained_layout=True)

    for r, z_val in enumerate(key_depths):
        idx = int(np.argmin(np.abs(target_depths - z_val)))
        
        # Ground Truth
        im0 = axes[r, 0].pcolormesh(lons, lats, true_interp[idx], cmap=cmap_field, vmin=vmin_field, vmax=vmax_field, shading='auto')
        axes[r, 0].set_ylabel(f"{int(z_val)}m\nLat (°N)", fontsize=11, fontweight='bold')
        if r == 0:
            axes[r, 0].set_title(f"GLORYS12V1 Truth ({unit})", fontsize=12, fontweight='bold')
        if r == n_rows - 1:
            axes[r, 0].set_xlabel("Longitude (°E)", fontsize=11)

        # PINN Pred
        im1 = axes[r, 1].pcolormesh(lons, lats, pred_interp[idx], cmap=cmap_field, vmin=vmin_field, vmax=vmax_field, shading='auto')
        if r == 0:
            axes[r, 1].set_title(f"PINN Prediction ({unit})", fontsize=12, fontweight='bold')
        if r == n_rows - 1:
            axes[r, 1].set_xlabel("Longitude (°E)", fontsize=11)

        # Error
        layer_rmse = np.sqrt(np.mean(error_interp[idx]**2))
        im2 = axes[r, 2].pcolormesh(lons, lats, error_interp[idx], cmap=cmap_err, vmin=-err_limit, vmax=err_limit, shading='auto')
        axes[r, 2].text(0.03, 0.88, f"RMSE: {layer_rmse:.3f}{unit}", transform=axes[r, 2].transAxes,
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        if r == 0:
            axes[r, 2].set_title(f"Residual (Pred - True)", fontsize=12, fontweight='bold')
        if r == n_rows - 1:
            axes[r, 2].set_xlabel("Longitude (°E)", fontsize=11)

    cbar_field = fig.colorbar(im0, ax=axes[:, :2], location='bottom', aspect=40, pad=0.02, shrink=0.6)
    cbar_field.set_label(f"{title_var} ({unit})", fontsize=11, fontweight='bold')

    cbar_err = fig.colorbar(im2, ax=axes[:, 2], location='bottom', aspect=20, pad=0.02, shrink=0.6)
    cbar_err.set_label(f"Residual Error ({unit})", fontsize=11, fontweight='bold')

    fig.suptitle(f"Northwest Pacific {title_var} Layer-by-Layer Subsurface Evaluation (50m Intervals)", fontsize=16, fontweight='bold', y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None
