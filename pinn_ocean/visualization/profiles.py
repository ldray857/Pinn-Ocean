# -*- coding: utf-8 -*-
"""
Vertical Profile Comparison Plotting Module for Pinn-Ocean
Plots representative station temperature and salinity profiles (0-1000m).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Configure high-quality publication styling and Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_vertical_profiles(
    true_t, pred_t, true_s, pred_s, depths,
    save_path="result/pic/fig1_profile_comparison.png",
    station_coord=None,
    station_label=None
):
    """
    Plots vertical profiles comparing ground truth and PINN reconstruction.
    
    Args:
        true_t, pred_t: (D, H, W) 3D temperature grids
        true_s, pred_s: (D, H, W) 3D salinity grids
        depths: (D,) 1D depth coordinate array
        save_path: output image filepath
        station_coord: optional (h_idx, w_idx) tuple; defaults to domain center
        station_label: optional string with geographic coordinates, e.g. '154.00°E, 34.33°N'
    """
    D, H, W = true_t.shape
    if station_coord is None:
        h_idx, w_idx = H // 2, W // 2
    else:
        h_idx, w_idx = station_coord

    t_prof_true = true_t[:, h_idx, w_idx]
    t_prof_pred = pred_t[:, h_idx, w_idx]
    s_prof_true = true_s[:, h_idx, w_idx]
    s_prof_pred = pred_s[:, h_idx, w_idx]

    # Calculate station profile metrics
    t_rmse = float(np.sqrt(np.mean((t_prof_pred - t_prof_true) ** 2)))
    s_rmse = float(np.sqrt(np.mean((s_prof_pred - s_prof_true) ** 2)))
    r_t = float(np.corrcoef(t_prof_pred, t_prof_true)[0, 1]) if np.std(t_prof_pred) > 1e-6 else 1.0
    r_s = float(np.corrcoef(s_prof_pred, s_prof_true)[0, 1]) if np.std(s_prof_pred) > 1e-6 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    sub_title_extra = f" [{station_label}]" if station_label else ""

    # 1. Temperature Vertical Profile
    axes[0].plot(t_prof_true, depths, 'k--', lw=2.2, label='GLORYS12V1 真值')
    axes[0].plot(t_prof_pred, depths, '#e74c3c', lw=2.8, label='Swin-Ocean-PINN 重构')
    axes[0].invert_yaxis()
    axes[0].set_title(f"代表站位温度垂直剖面重构 (0-1000m){sub_title_extra}", fontsize=12.5, fontweight='bold')
    axes[0].set_xlabel("位温 Potential Temperature (°C)", fontsize=11)
    axes[0].set_ylabel("水深 Depth (m)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10.5, loc='upper left')

    # Metric text box on temperature plot
    info_t = f"站位评估 (0-1000m):\n$R = {r_t:.4f}$\n$\\mathrm{{RMSE}} = {t_rmse:.2f}^\\circ\\mathrm{{C}}$"
    axes[0].text(
        0.05, 0.72, info_t,
        transform=axes[0].transAxes, fontsize=10.5,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
    )

    # 2. Salinity Vertical Profile
    axes[1].plot(s_prof_true, depths, 'k--', lw=2.2, label='GLORYS12V1 真值')
    axes[1].plot(s_prof_pred, depths, '#2980b9', lw=2.8, label='Swin-Ocean-PINN 重构')
    axes[1].invert_yaxis()
    axes[1].set_title(f"代表站位盐度垂直剖面重构 (0-1000m){sub_title_extra}", fontsize=12.5, fontweight='bold')
    axes[1].set_xlabel("实用盐度 Practical Salinity (PSU)", fontsize=11)
    axes[1].set_ylabel("水深 Depth (m)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10.5, loc='upper left')

    # Metric text box on salinity plot
    info_s = f"站位评估 (0-1000m):\n$R = {r_s:.4f}$\n$\\mathrm{{RMSE}} = {s_rmse:.4f}\\ \\mathrm{{PSU}}$"
    axes[1].text(
        0.05, 0.72, info_s,
        transform=axes[1].transAxes, fontsize=10.5,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
