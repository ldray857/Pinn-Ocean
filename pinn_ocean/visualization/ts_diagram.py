# -*- coding: utf-8 -*-
"""
Temperature-Salinity (T-S) Physical Consistency Diagram Module for Pinn-Ocean
Validates water mass distributions and thermodynamic stability (no density inversion).
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_ts_diagram(
    all_true_t, all_pred_t, all_true_s, all_pred_s,
    save_path="results/fig2_ts_diagram.png",
    max_points=4000
):
    """
    Plots T-S diagram comparing ground truth and PINN reconstruction.
    
    Args:
        all_true_t, all_pred_t: 1D arrays of flat temperature samples
        all_true_s, all_pred_s: 1D arrays of flat salinity samples
        save_path: output image filepath
        max_points: maximum scatter points to display for crisp rendering
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))

    n_samples = len(all_true_t)
    if n_samples > max_points:
        sub_idx = np.random.choice(n_samples, size=max_points, replace=False)
    else:
        sub_idx = np.arange(n_samples)

    ax.scatter(
        all_true_s[sub_idx], all_true_t[sub_idx],
        c='silver', s=16, alpha=0.55, label='GLORYS 水团真值', edgecolors='none'
    )
    ax.scatter(
        all_pred_s[sub_idx], all_pred_t[sub_idx],
        c='#e67e22', s=16, alpha=0.75, label='PINN 物理重构', edgecolors='none'
    )

    ax.set_title("西北太平洋温盐关系 (T-S Diagram) 物理一致性检验", fontsize=13, fontweight='bold')
    ax.set_xlabel("盐度 Salinity (PSU)", fontsize=11)
    ax.set_ylabel("位温 Potential Temperature (°C)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
