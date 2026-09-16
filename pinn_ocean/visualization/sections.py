# -*- coding: utf-8 -*-
"""
Vertical Transect and Layer-wise Metrics Scientific Visualization Module for Pinn-Ocean
1. Full-depth 0-1000m vertical continuous transect along Kuroshio Extension (35°N)
2. Vertical profile curves of layer-by-layer RMSE(z), MAE(z), and R^2(z)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_vertical_section(
    true_3d, pred_3d, lons, lats, depths,
    slice_type="lat", slice_val=35.0,
    var_name="temperature",
    save_path="result/pic/fig2_vertical_section_35n.png"
):
    """
    Plots a continuous 2D vertical transect section (0-1000m):
    Panel 1: GLORYS12V1 Reference (真值)
    Panel 2: Swin-Ocean-PINN Reconstructed (重构)
    Panel 3: Reconstruction Error (重构误差: Pred - True)
    """
    if slice_type == "lat":
        idx = int(np.argmin(np.abs(lats - slice_val)))
        actual_coord = float(lats[idx])
        section_true = true_3d[:, idx, :]  # (D, W)
        section_pred = pred_3d[:, idx, :]
        x_coords = lons
        x_label = "经度 Longitude (°E)"
        slice_desc = f"纬向截面 (Lat = {actual_coord:.2f}°N, 黑潮延伸体主轴)"
    else:
        idx = int(np.argmin(np.abs(lons - slice_val)))
        actual_coord = float(lons[idx])
        section_true = true_3d[:, :, idx]  # (D, H)
        section_pred = pred_3d[:, :, idx]
        x_coords = lats
        x_label = "纬度 Latitude (°N)"
        slice_desc = f"经向截面 (Lon = {actual_coord:.2f}°E)"

    section_err = section_pred - section_true  # Signed error
    X, Z = np.meshgrid(x_coords, depths)

    if var_name == "temperature":
        cmap_field = cmocean.cm.thermal
        unit_str = "°C"
        var_label = "位温 Potential Temperature"
        levels = np.linspace(2.0, 28.0, 27)
        err_levels = np.linspace(-3.0, 3.0, 25)
    else:
        cmap_field = cmocean.cm.haline
        unit_str = "PSU"
        var_label = "实用盐度 Practical Salinity"
        levels = np.linspace(33.8, 35.2, 29)
        err_levels = np.linspace(-0.35, 0.35, 29)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True, dpi=300)

    # 1. Ground Truth Section
    cf0 = axes[0].contourf(X, Z, section_true, levels=levels, cmap=cmap_field, extend='both')
    cs0 = axes[0].contour(X, Z, section_true, levels=levels[::2], colors='k', linewidths=0.5, alpha=0.5)
    axes[0].clabel(cs0, inline=True, fontsize=8, fmt='%.1f')
    axes[0].set_title(f"GLORYS12V1 真值 - {var_label} 垂直大断面 ({slice_desc})", fontsize=11.5, fontweight='bold')
    axes[0].set_ylabel("水深 Depth (m)", fontsize=10.5)
    axes[0].invert_yaxis()
    cbar0 = fig.colorbar(cf0, ax=axes[0], orientation='vertical', shrink=0.92, pad=0.02)
    cbar0.set_label(f"{var_label} ({unit_str})", fontsize=9.5)

    # 2. Swin-Ocean-PINN Reconstructed Section
    cf1 = axes[1].contourf(X, Z, section_pred, levels=levels, cmap=cmap_field, extend='both')
    cs1 = axes[1].contour(X, Z, section_pred, levels=levels[::2], colors='k', linewidths=0.5, alpha=0.5)
    axes[1].clabel(cs1, inline=True, fontsize=8, fmt='%.1f')
    axes[1].set_title(f"Swin-Ocean-PINN 重构 - {var_label} 垂直大断面", fontsize=11.5, fontweight='bold')
    axes[1].set_ylabel("水深 Depth (m)", fontsize=10.5)
    cbar1 = fig.colorbar(cf1, ax=axes[1], orientation='vertical', shrink=0.92, pad=0.02)
    cbar1.set_label(f"{var_label} ({unit_str})", fontsize=9.5)

    # 3. Residual Error Section
    cf2 = axes[2].contourf(X, Z, section_err, levels=err_levels, cmap=cmocean.cm.balance, extend='both')
    axes[2].set_title(f"反演残差分布 (Pred - Truth) - {var_label}", fontsize=11.5, fontweight='bold')
    axes[2].set_xlabel(x_label, fontsize=10.5)
    axes[2].set_ylabel("水深 Depth (m)", fontsize=10.5)
    cbar2 = fig.colorbar(cf2, ax=axes[2], orientation='vertical', shrink=0.92, pad=0.02)
    cbar2.set_label(f"残差偏差 Bias ({unit_str})", fontsize=9.5)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_layer_metrics_profile(
    layer_t, layer_s, depths,
    save_path="result/pic/fig3_layer_metrics_depth.png"
):
    """
    Plots vertical profiles of RMSE, MAE, and R^2 across water depth (0-1000m).
    Highlights key ocean regimes: Mixed Layer, Main Thermocline, Deep Layer.
    """
    z = np.array(depths)
    rmse_t = np.array(layer_t['rmse'])
    mae_t = np.array(layer_t['mae'])
    r2_t = np.array(layer_t['r2'])

    rmse_s = np.array(layer_s['rmse'])
    mae_s = np.array(layer_s['mae'])
    r2_s = np.array(layer_s['r2'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True, dpi=300)

    # Shaded ocean regimes
    for ax in axes:
        ax.axhspan(0, 100, color='#3498db', alpha=0.08, label='混合层 (0-100m)')
        ax.axhspan(100, 400, color='#e67e22', alpha=0.08, label='主跃层 (100-400m)')
        ax.axhspan(400, 1000, color='#9b59b6', alpha=0.08, label='中深层 (400-1000m)')
        ax.invert_yaxis()
        ax.grid(True, linestyle=":", alpha=0.5)

    # 1. Temperature Errors vs Depth
    axes[0].plot(rmse_t, z, 'r-o', lw=2.2, ms=4.5, label='RMSE (°C)')
    axes[0].plot(mae_t, z, 'm--s', lw=1.8, ms=4, label='MAE (°C)')
    axes[0].set_title("温度重建误差随深度分布", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("误差数值 (°C)", fontsize=11)
    axes[0].set_ylabel("水深 Depth (m)", fontsize=11)
    axes[0].legend(loc='lower right', fontsize=10)

    # 2. Salinity Errors vs Depth
    axes[1].plot(rmse_s, z, 'b-o', lw=2.2, ms=4.5, label='RMSE (PSU)')
    axes[1].plot(mae_s, z, 'c--s', lw=1.8, ms=4, label='MAE (PSU)')
    axes[1].set_title("盐度重建误差随深度分布", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("误差数值 (PSU)", fontsize=11)
    axes[1].legend(loc='lower right', fontsize=10)

    # 3. Coefficient of Determination R^2 vs Depth
    axes[2].plot(r2_t, z, 'r-^', lw=2.2, ms=5, label='温度 $R^2$')
    axes[2].plot(r2_s, z, 'b-v', lw=2.2, ms=5, label='盐度 $R^2$')
    axes[2].set_title("决定系数 $R^2$ 随深度保真度", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("决定系数 $R^2$", fontsize=11)
    axes[2].set_xlim(-0.05, 1.05)
    axes[2].legend(loc='lower left', fontsize=10)

    fig.suptitle("Swin-Ocean-PINN 太平洋三维次表层温盐逐层垂直反演精度与衰减剖面 (0-1000m)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path
