# -*- coding: utf-8 -*-
"""
Layer-wise Metrics Scientific Visualization Module for Pinn-Ocean
Vertical profile curves of layer-by-layer RMSE(z), MAE(z), and R^2(z)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False



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
