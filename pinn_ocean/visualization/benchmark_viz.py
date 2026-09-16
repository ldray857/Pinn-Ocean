# -*- coding: utf-8 -*-
"""
Publication-Grade Comparative Visualization Suite for Pinn-Ocean Superiority Benchmark
Generates:
1. Fig B1: Multi-Model 6-Dimensional Superiority Radar Chart (Comprehensive Performance)
2. Fig B2: Vertical Transect Stratification Stability & Instability Patch Overlay (35°N Kuroshio)
3. Fig B3: GLORYS High-Resolution Super-Resolution & Fine-Scale Inset Comparison
4. Fig B4: Quantitative Ablation Improvement & Error Reduction Bar Summary
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from typing import Dict, List, Optional, Tuple

from pinn_ocean.utils.teos10 import calc_buoyancy_frequency_n2

# Publication styling defaults
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def plot_superiority_radar(
    models_metrics: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "多模型全维度学术优度雷达对比 (Superiority Benchmark Radar)"
):
    """
    Plots a multi-model 6-axis radar chart showing relative superiority across:
    1. 温度重构拟合度 (Temperature Fidelity)
    2. 盐度跃层拟合度 (Salinity Fidelity)
    3. 空间决定系数 (R^2 Score)
    4. 层结防倒置合规率 (Stratification Stability: 100% - CIR)
    5. 深水热力学单调性 (Deep Monotonicity: 100% - TMV)
    6. 混合层界面精准度 (MLD Boundary Fidelity)
    """
    categories = [
        "温度拟合精度\n(Temp Fidelity)",
        "盐度跃层拟合\n(Sal Fidelity)",
        "空间解释度\n(R² Score)",
        "层结防倒置率\n(Stability 1-CIR)",
        "深水单调合规\n(Monotonicity 1-TMV)",
        "混合层界面\n(MLD Accuracy)"
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Complete loop

    fig, ax = plt.subplots(figsize=(8.5, 8.0), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FBFDFF')

    # Color and styling palette
    model_styles = {
        "Trilinear (三维空间插值)": {"color": "#64748B", "linestyle": "--", "alpha": 0.10, "marker": "o", "linewidth": 1.5},
        "Pure-CNN (无物理卷积网络)": {"color": "#F59E0B", "linestyle": "-.", "alpha": 0.15, "marker": "s", "linewidth": 2.0},
        "Pure-Swin (无物理消融对照)": {"color": "#3B82F6", "linestyle": ":", "alpha": 0.20, "marker": "^", "linewidth": 2.2},
        "Swin-Ocean-PINN (本项目模型)": {"color": "#DC2626", "linestyle": "-", "alpha": 0.35, "marker": "D", "linewidth": 3.0}
    }

    for name, scores in models_metrics.items():
        style = model_styles.get(name, {"color": "#10B981", "linestyle": "-", "alpha": 0.2, "marker": "o", "linewidth": 2.0})
        raw_vals = [
            scores.get("score_t", 50.0),
            scores.get("score_s", 50.0),
            scores.get("score_r2", 50.0),
            scores.get("score_stratification", 50.0),
            scores.get("score_monotonicity", 50.0),
            scores.get("score_mld", 50.0)
        ]
        vals = [min(100.0, max(10.0, v)) for v in raw_vals]
        vals += vals[:1]

        ax.plot(angles, vals, label=name, color=style["color"],
                linestyle=style["linestyle"], linewidth=style["linewidth"],
                marker=style["marker"], markersize=6)
        ax.fill(angles, vals, color=style["color"], alpha=style["alpha"])

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5, fontweight='semibold', color='#1E293B')

    ax.set_rlabel_position(30)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100分"], color='#94A3B8', fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.grid(color='#CBD5E1', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.title(title, fontsize=14, fontweight='bold', pad=28, color='#0F172A')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9.5, frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> [Saved Fig] Multi-Model Superiority Radar: {output_path}")


def plot_physics_stability_transect(
    depths: np.ndarray,
    lons: np.ndarray,
    true_t_sec: np.ndarray,
    true_s_sec: np.ndarray,
    models_pred: Dict[str, Tuple[np.ndarray, np.ndarray]],
    slice_lat: float,
    output_path: str
):
    """
    Plots a multi-panel vertical transect along slice_lat (e.g. 35°N Kuroshio Extension).
    Overlays red warning markers on unstable convective points where N^2 < 0,
    graphically demonstrating that baseline models suffer from severe physical unviability
    while Swin-Ocean-PINN strictly preserves gravitational stratification.
    """
    model_keys = list(models_pred.keys())
    n_panels = len(model_keys) + 1  # 1 Truth + N Models
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True, dpi=300)
    if n_panels == 1:
        axes = [axes]

    levels = np.linspace(2.0, 24.0, 23)
    cmap = plt.cm.plasma

    # Panel 0: Ground Truth
    ax0 = axes[0]
    c0 = ax0.contourf(lons, depths, true_t_sec, levels=levels, cmap=cmap, extend='both')
    cs0 = ax0.contour(lons, depths, true_t_sec, levels=[5, 10, 15, 20], colors='white', linewidths=0.6, alpha=0.6)
    ax0.clabel(cs0, inline=True, fontsize=7.5, fmt='%.0f°C')
    ax0.set_ylim(1000, 0)
    ax0.set_ylabel("水深 Depth (m)", fontsize=10)
    ax0.set_title(f"(a) GLORYS12V1 参考真值 (Reference Truth) | 断面纬度: {slice_lat:.1f}°N", fontsize=11, fontweight='bold', loc='left')

    # Calculate truth N^2 instability
    n2_true = calc_buoyancy_frequency_n2(true_s_sec[:, None, :], true_t_sec[:, None, :], depths)
    unstable_true = (n2_true[:, 0, :] < 0)
    if np.any(unstable_true):
        z_mid = 0.5 * (depths[:-1] + depths[1:])
        pts_z, pts_x = np.where(unstable_true)
        ax0.scatter(lons[pts_x], z_mid[pts_z], color='#EF4444', s=8, alpha=0.8, marker='x', label='对流失稳 N²<0')

    cbar = fig.colorbar(c0, ax=ax0, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("位温 (°C)", fontsize=9)

    # Panels 1..N: Comparison Models
    labels = ["(b)", "(c)", "(d)", "(e)"]
    for idx, (m_name, (pred_t_sec, pred_s_sec)) in enumerate(models_pred.items()):
        ax = axes[idx + 1]
        c = ax.contourf(lons, depths, pred_t_sec, levels=levels, cmap=cmap, extend='both')
        cs = ax.contour(lons, depths, pred_t_sec, levels=[5, 10, 15, 20], colors='white', linewidths=0.6, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.0f°C')
        ax.set_ylim(1000, 0)
        ax.set_ylabel("水深 Depth (m)", fontsize=10)

        # Compute N^2
        n2_pred = calc_buoyancy_frequency_n2(pred_s_sec[:, None, :], pred_t_sec[:, None, :], depths)
        unstable = (n2_pred[:, 0, :] < 0)
        z_mid = 0.5 * (depths[:-1] + depths[1:])
        pts_z, pts_x = np.where(unstable)

        instability_rate = float(np.sum(unstable) / max(unstable.size, 1) * 100.0)

        scatter_lbl = f"失稳区域 N²<0 (失稳率: {instability_rate:.2f}%)"
        ax.scatter(lons[pts_x], z_mid[pts_z], color='#DC2626', s=10, alpha=0.85, marker='x', label=scatter_lbl)

        lbl = labels[idx] if idx < len(labels) else f"({chr(ord('b')+idx)})"
        ax.set_title(f"{lbl} {m_name} | 对流失稳率 CIR: {instability_rate:.2f}%", fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9, facecolor='#FEF2F2', edgecolor='#F87171')

        cbar = fig.colorbar(c, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label("位温 (°C)", fontsize=9)

    axes[-1].set_xlabel("经度 Longitude (°E)", fontsize=10.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> [Saved Fig] Physics Stratification Transect Comparison: {output_path}")


def plot_glorys_super_resolution_comparison(
    lons_coarse: np.ndarray,
    lats_coarse: np.ndarray,
    field_coarse: np.ndarray,
    lons_fine: np.ndarray,
    lats_fine: np.ndarray,
    field_trilinear: np.ndarray,
    field_tricubic: np.ndarray,
    field_pinn_hr: np.ndarray,
    output_path: str,
    depth_m: float = 100.0,
    var_name: str = "Temperature (°C)"
):
    """
    Plots a 4-panel comparison of GLORYS 3-D slice super-resolution & interpolation:
    (a) Coarse GLORYS Grid (1/12°) with grid cell outlines
    (b) 3-D Trilinear Interpolation (shows blocky diamond artifacts)
    (c) 3-D Tricubic Spline Interpolation (shows slight ringing/overshoot)
    (d) Swin-Ocean-PINN Super-Resolution (Continuous Neural Operator, sharp physical edges)
    Includes a zoomed-in local inset highlighting frontal sharp gradient preservation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    vmin = float(np.nanpercentile(field_coarse, 2))
    vmax = float(np.nanpercentile(field_coarse, 98))
    cmap = plt.cm.Spectral_r

    # Sub-window for zoom-in: e.g. 152°E~157°E, 33°N~37°N (Kuroshio Front)
    zoom_lon_min, zoom_lon_max = 152.0, 157.0
    zoom_lat_min, zoom_lat_max = 33.0, 37.0

    panels = [
        (axes[0, 0], "(a) 原始 GLORYS12V1 网格 (1/12° ~ 9.2km)", lons_coarse, lats_coarse, field_coarse),
        (axes[0, 1], "(b) 三维线性插值 Trilinear (1/24° 空间加密)", lons_fine, lats_fine, field_trilinear),
        (axes[1, 0], "(c) 三次样条插值 Tricubic Spline (1/24° 空间加密)", lons_fine, lats_fine, field_tricubic),
        (axes[1, 1], "(d) Swin-Ocean-PINN 神经算子超分 (1/24° 连续物理场)", lons_fine, lats_fine, field_pinn_hr)
    ]

    for ax, subtitle, lons, lats, field in panels:
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(subtitle, fontsize=11, fontweight='bold', loc='left', pad=6)
        ax.set_xlabel("经度 Longitude (°E)", fontsize=9.5)
        ax.set_ylabel("纬度 Latitude (°N)", fontsize=9.5)

        # Draw red dashed bounding box for the zoom-in area
        rect = plt.Rectangle(
            (zoom_lon_min, zoom_lat_min),
            zoom_lon_max - zoom_lon_min,
            zoom_lat_max - zoom_lat_min,
            fill=False, edgecolor='#DC2626', linestyle='--', linewidth=1.5, zorder=5
        )
        ax.add_patch(rect)

        # Add inset zoom-in axis
        axins = inset_axes(ax, width="38%", height="38%", loc="lower left",
                           bbox_to_anchor=(0.04, 0.06, 0.9, 0.9), bbox_transform=ax.transAxes)
        axins.pcolormesh(lons, lats, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        axins.set_xlim(zoom_lon_min, zoom_lon_max)
        axins.set_ylim(zoom_lat_min, zoom_lat_max)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor('#DC2626')
            spine.set_linewidth(1.5)

    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f"{var_name} (水深 Depth = {depth_m:.0f}m 核心温跃层)", fontsize=11, fontweight='semibold')

    plt.suptitle(f"GLORYS 三维温盐场空间插值高分与超分辨力对比 (水深 {depth_m:.0f}m)", fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(bottom=0.10, top=0.93, hspace=0.25, wspace=0.18)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> [Saved Fig] GLORYS Super-Resolution Comparison: {output_path}")


def plot_superiority_bar_summary(
    benchmark_data: Dict[str, Dict[str, float]],
    output_path: str
):
    """
    Plots a grouped bar chart visualizing error reduction and stability gains
    of Swin-Ocean-PINN over baseline models.
    """
    models = list(benchmark_data.keys())
    n_models = len(models)

    metrics = [
        ("temp_rmse", "全水深温度 RMSE (°C)", "lower_is_better"),
        ("sal_rmse_thermocline", "温跃层盐度 RMSE (PSU)", "lower_is_better"),
        ("cir_percent", "浮力频率失稳率 CIR (%)", "lower_is_better"),
        ("tmv_percent", "深水逆温违背率 TMV (%)", "lower_is_better")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    colors = ['#94A3B8', '#F59E0B', '#3B82F6', '#DC2626']

    for idx, (metric_key, metric_label, direction) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        vals = [benchmark_data[m].get(metric_key, 0.0) for m in models]
        bars = ax.bar(models, vals, color=colors[:n_models], width=0.55, edgecolor='#334155', linewidth=0.8)

        # Highlight best
        best_val = min(vals) if direction == "lower_is_better" else max(vals)
        ax.set_title(metric_label, fontsize=11, fontweight='bold', pad=8)
        ax.set_ylabel(metric_label.split(" ")[-1], fontsize=9.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Annotate numbers on top of bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            is_best = (val == best_val)
            weight = 'bold' if is_best else 'normal'
            txt_color = '#DC2626' if is_best else '#1E293B'
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (max(vals) * 0.02),
                f"{val:.3f}" if val < 1.0 else f"{val:.2f}",
                ha='center', va='bottom', fontsize=9.5, fontweight=weight, color=txt_color
            )

        # Rotate x labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split(" ")[0] for m in models], fontsize=9.5, rotation=15)

    plt.suptitle("多模型综合优度量化评测总览 (Superiority Performance Benchmark)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> [Saved Fig] Superiority Bar Summary: {output_path}")
