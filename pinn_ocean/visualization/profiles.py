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
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_multi_station_profiles(
    true_t, pred_t, true_s, pred_s, depths, lons, lats,
    stations=None,
    save_path="result/pic/fig4_multi_station_profiles.png"
):
    """
    Plots multi-station vertical profile comparisons across 4 contrasting
    oceanographic dynamic regimes in the Northwest Pacific.

    Default Stations:
    1. Kuroshio Jet Axis (黑潮延伸体急流轴): 148.0°E, 35.0°N (Strong shear & eddy interaction)
    2. Subtropical Warm Pool (亚热带再循环暖水区): 158.0°E, 32.0°N (Thick mixed layer, stable stratification)
    3. Subarctic Cold Water (北侧冷水边缘区): 152.0°E, 38.5°N (Shallow thermocline, cold subsurface)
    4. Open Ocean Center (开阔大洋中心区): 162.0°E, 35.5°N (Representative baseline)

    Args:
        true_t, pred_t: (D, H, W) temperature fields
        true_s, pred_s: (D, H, W) salinity fields
        depths: (D,) depth array
        lons: (W,) longitude array
        lats: (H,) latitude array
        stations: optional list of dicts with 'name', 'lon', 'lat'
        save_path: output filepath
    """
    if stations is None:
        stations = [
            {"name": "站位A: 黑潮急流主轴区", "lon": 148.0, "lat": 35.0},
            {"name": "站位B: 亚热带暖水再循环区", "lon": 158.0, "lat": 32.0},
            {"name": "站位C: 北侧亚极地过渡冷区", "lon": 152.0, "lat": 38.5},
            {"name": "站位D: 开阔大洋中心区", "lon": 162.0, "lat": 35.5}
        ]

    D, H, W = true_t.shape
    num_stations = len(stations)

    fig, axes = plt.subplots(2, num_stations, figsize=(5.2 * num_stations, 10.5), sharey=True, dpi=300)

    for col_idx, st in enumerate(stations):
        target_lon = st["lon"]
        target_lat = st["lat"]
        w_idx = int(np.argmin(np.abs(lons - target_lon)))
        h_idx = int(np.argmin(np.abs(lats - target_lat)))
        actual_lon = float(lons[w_idx])
        actual_lat = float(lats[h_idx])

        t_true_prof = true_t[:, h_idx, w_idx]
        t_pred_prof = pred_t[:, h_idx, w_idx]
        s_true_prof = true_s[:, h_idx, w_idx]
        s_pred_prof = pred_s[:, h_idx, w_idx]

        t_rmse = float(np.sqrt(np.mean((t_pred_prof - t_true_prof) ** 2)))
        s_rmse = float(np.sqrt(np.mean((s_pred_prof - s_true_prof) ** 2)))
        r_t = float(np.corrcoef(t_pred_prof, t_true_prof)[0, 1]) if np.std(t_pred_prof) > 1e-6 else 1.0
        r_s = float(np.corrcoef(s_pred_prof, s_true_prof)[0, 1]) if np.std(s_pred_prof) > 1e-6 else 1.0

        # Row 0: Temperature Profiles
        ax_t = axes[0, col_idx]
        ax_t.plot(t_true_prof, depths, 'k--', lw=2.2, label='GLORYS12V1 真值')
        ax_t.plot(t_pred_prof, depths, '#e74c3c', lw=2.6, label='PINN 重构')
        ax_t.grid(True, linestyle=":", alpha=0.6)
        ax_t.set_title(f"{st['name']}\n({actual_lon:.1f}°E, {actual_lat:.1f}°N)", fontsize=11, fontweight='bold')
        ax_t.set_xlabel("位温 Temperature (°C)", fontsize=10)
        if col_idx == 0:
            ax_t.set_ylabel("水深 Depth (m)", fontsize=11)
            ax_t.legend(loc='lower left', fontsize=9.5)

        info_t = f"$R = {r_t:.4f}$\n$\\mathrm{{RMSE}} = {t_rmse:.2f}^\\circ\\mathrm{{C}}$"
        ax_t.text(0.06, 0.72, info_t, transform=ax_t.transAxes, fontsize=9.5,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#bdc3c7'))

        # Row 1: Salinity Profiles
        ax_s = axes[1, col_idx]
        ax_s.plot(s_true_prof, depths, 'k--', lw=2.2, label='GLORYS12V1 真值')
        ax_s.plot(s_pred_prof, depths, '#2980b9', lw=2.6, label='PINN 重构')
        ax_s.grid(True, linestyle=":", alpha=0.6)
        ax_s.set_title(f"盐度剖面 ({actual_lon:.1f}°E, {actual_lat:.1f}°N)", fontsize=11)
        ax_s.set_xlabel("实用盐度 Salinity (PSU)", fontsize=10)
        if col_idx == 0:
            ax_s.set_ylabel("水深 Depth (m)", fontsize=11)
            ax_s.legend(loc='lower left', fontsize=9.5)

        info_s = f"$R = {r_s:.4f}$\n$\\mathrm{{RMSE}} = {s_rmse:.4f}\\ \\mathrm{{PSU}}$"
        ax_s.text(0.06, 0.72, info_s, transform=ax_s.transAxes, fontsize=9.5,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#bdc3c7'))

    axes[0, 0].set_ylim(1000.0, 0.0)  # Inverted depth: surface 0m on top, 1000m at bottom

    fig.suptitle("Swin-Ocean-PINN 西北太平洋四大典型动力学特征站位三维垂向温盐剖面对比 (0-1000m)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path

