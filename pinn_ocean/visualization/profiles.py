# -*- coding: utf-8 -*-
"""
Vertical Profile Comparison Plotting Module for Pinn-Ocean
Plots representative station temperature and salinity profiles (0-1000m).
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_vertical_profiles(
    true_t, pred_t, true_s, pred_s, depths,
    save_path="results/fig1_profile_comparison.png",
    station_coord=None
):
    """
    Plots vertical profiles comparing ground truth and PINN reconstruction.
    
    Args:
        true_t, pred_t: (D, H, W) 3D temperature grids
        true_s, pred_s: (D, H, W) 3D salinity grids
        depths: (D,) 1D depth coordinate array
        save_path: output image filepath
        station_coord: optional (h_idx, w_idx) tuple; defaults to domain center
    """
    D, H, W = true_t.shape
    if station_coord is None:
        h_idx, w_idx = H // 2, W // 2
    else:
        h_idx, w_idx = station_coord

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    # Temperature Vertical Profile
    axes[0].plot(true_t[:, h_idx, w_idx], depths, 'k--', lw=2.2, label='GLORYS12V1 真值')
    axes[0].plot(pred_t[:, h_idx, w_idx], depths, '#e74c3c', lw=2.8, label='Swin-Ocean-PINN 重构')
    axes[0].invert_yaxis()
    axes[0].set_title("代表站位温度垂直剖面重构 (0-1000m)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("位温 Potential Temperature (°C)", fontsize=11)
    axes[0].set_ylabel("水深 Depth (m)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=11, loc='lower left')

    # Salinity Vertical Profile
    axes[1].plot(true_s[:, h_idx, w_idx], depths, 'k--', lw=2.2, label='GLORYS12V1 真值')
    axes[1].plot(pred_s[:, h_idx, w_idx], depths, '#2980b9', lw=2.8, label='Swin-Ocean-PINN 重构')
    axes[1].invert_yaxis()
    axes[1].set_title("代表站位盐度垂直剖面重构 (0-1000m)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("实用盐度 Practical Salinity (PSU)", fontsize=11)
    axes[1].set_ylabel("水深 Depth (m)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=11, loc='lower left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
