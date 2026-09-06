# -*- coding: utf-8 -*-
"""
Full-Depth Prediction vs Truth Scatter Density (Hexbin) Module for Pinn-Ocean
Plots hexbin density distributions for temperature and salinity with R^2 and RMSE annotations.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from ..utils.metrics import calc_rmse, calc_r2


def plot_scatter_density(
    all_true_t, all_pred_t, all_true_s, all_pred_s,
    save_path="results/fig3_scatter_density.png",
    gridsize=55
):
    """
    Plots log-scale hexbin scatter density validation for temperature and salinity.
    
    Args:
        all_true_t, all_pred_t: 1D arrays of flat temperature samples
        all_true_s, all_pred_s: 1D arrays of flat salinity samples
        save_path: output image filepath
        gridsize: resolution of hexagonal binning
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- 1. Temperature Hexbin ---
    r2_t = calc_r2(all_pred_t, all_true_t)
    rmse_t = calc_rmse(all_pred_t, all_true_t)
    hb_t = axes[0].hexbin(all_true_t, all_pred_t, gridsize=gridsize, cmap='YlOrRd', mincnt=1, bins='log')
    t_min = min(all_true_t.min(), all_pred_t.min())
    t_max = max(all_true_t.max(), all_pred_t.max())
    axes[0].plot([t_min, t_max], [t_min, t_max], 'k--', lw=2, label='1:1 理想参考线')
    axes[0].set_title(f"全深度温度预测散点密度 (RMSE: {rmse_t:.3f}°C, R²: {r2_t:.4f})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("真值 Target Temperature (°C)", fontsize=11)
    axes[0].set_ylabel("预测 Predicted Temperature (°C)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc='upper left', fontsize=10)
    plt.colorbar(hb_t, ax=axes[0], label="样本点密度 log10(N)")

    # --- 2. Salinity Hexbin ---
    r2_s = calc_r2(all_pred_s, all_true_s)
    rmse_s = calc_rmse(all_pred_s, all_true_s)
    hb_s = axes[1].hexbin(all_true_s, all_pred_s, gridsize=gridsize, cmap='YlGnBu', mincnt=1, bins='log')
    s_min = min(all_true_s.min(), all_pred_s.min())
    s_max = max(all_true_s.max(), all_pred_s.max())
    axes[1].plot([s_min, s_max], [s_min, s_max], 'k--', lw=2, label='1:1 理想参考线')
    axes[1].set_title(f"全深度盐度预测散点密度 (RMSE: {rmse_s:.3f} PSU, R²: {r2_s:.4f})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("真值 Target Salinity (PSU)", fontsize=11)
    axes[1].set_ylabel("预测 Predicted Salinity (PSU)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc='upper left', fontsize=10)
    plt.colorbar(hb_s, ax=axes[1], label="样本点密度 log10(N)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
