# -*- coding: utf-8 -*-
"""
Mixed Layer Depth (MLD) Validation Plotting Module for Pinn-Ocean
Validates upper-ocean stratification interface against ground truth.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from ..utils.metrics import calc_rmse, calc_mae, calc_r2


def plot_mld_validation(
    all_true_mld, all_pred_mld,
    save_path="results/fig4_mld_validation.png"
):
    """
    Plots MLD prediction vs truth scatter validation.
    
    Args:
        all_true_mld: 1D array of ground truth MLD values in meters
        all_pred_mld: 1D array of reconstructed MLD values in meters
        save_path: output image filepath
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    r2_mld = calc_r2(all_pred_mld, all_true_mld)
    rmse_mld = calc_rmse(all_pred_mld, all_true_mld)
    mae_mld = calc_mae(all_pred_mld, all_true_mld)

    ax.scatter(all_true_mld, all_pred_mld, c='#8e44ad', alpha=0.6, edgecolors='none', s=24)
    mld_min = min(all_true_mld.min(), all_pred_mld.min())
    mld_max = max(all_true_mld.max(), all_pred_mld.max())
    ax.plot([mld_min, mld_max], [mld_min, mld_max], 'k--', lw=2, label='1:1 理想参考线')

    ax.set_title(
        f"上混合层深度 (MLD) 物理界面反演对比\n"
        f"(RMSE: {rmse_mld:.2f}m, MAE: {mae_mld:.2f}m, R²: {r2_mld:.3f})",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel("真值 GLORYS MLD (m)", fontsize=11)
    ax.set_ylabel("预测 PINN MLD (m)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
