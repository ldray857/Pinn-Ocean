# -*- coding: utf-8 -*-
"""
3D Volumetric and Orthogonal Slice Scientific Visualization Module for Pinn-Ocean
Renders 3D oceanographic bounding box with intersecting orthogonal slices:
1. Horizontal depth slice (e.g. main thermocline at z=100m)
2. Zonal vertical transect wall (e.g. along Kuroshio Extension at 35°N)
3. Meridional vertical transect wall (e.g. at 155°E)
Also includes 3D Isothermal Surface Topography.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmocean

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _get_slice_indices(lons, lats, depths, slice_lon=155.0, slice_lat=35.0, slice_depth=100.0):
    """Find closest grid indices for slice coordinates."""
    w_idx = int(np.argmin(np.abs(lons - slice_lon)))
    h_idx = int(np.argmin(np.abs(lats - slice_lat)))
    d_idx = int(np.argmin(np.abs(depths - slice_depth)))
    return w_idx, h_idx, d_idx


def plot_3d_thermohaline_box(
    true_field, pred_field, lons, lats, depths,
    var_name="temperature",
    slice_lon=155.0, slice_lat=35.0, slice_depth=100.0,
    save_path="result/pic/fig1_3d_volume_slice.png",
    title=None
):
    """
    Renders 3D oceanographic orthogonal slices and fence box:
    Panel 1: GLORYS12V1 Reference (真值)
    Panel 2: Swin-Ocean-PINN Reconstructed (重构)
    Panel 3: Absolute Reconstruction Error (绝对误差)
    """
    w_idx, h_idx, d_idx = _get_slice_indices(lons, lats, depths, slice_lon, slice_lat, slice_depth)
    actual_lon = float(lons[w_idx])
    actual_lat = float(lats[h_idx])
    actual_depth = float(depths[d_idx])

    err_field = np.abs(pred_field - true_field)

    # Colormaps and ranges
    if var_name == "temperature":
        cmap_field = cmocean.cm.thermal
        unit_str = "°C"
        var_label = "位温 Potential Temperature"
        vmin = float(np.percentile(true_field, 1))
        vmax = float(np.percentile(true_field, 99))
        err_vmax = float(np.percentile(err_field, 98))
    else:
        cmap_field = cmocean.cm.haline
        unit_str = "PSU"
        var_label = "实用盐度 Practical Salinity"
        vmin = float(np.percentile(true_field, 1))
        vmax = float(np.percentile(true_field, 99))
        err_vmax = float(np.percentile(err_field, 98))

    cmap_err = cmocean.cm.amp
    norm_field = plt.Normalize(vmin=vmin, vmax=vmax)
    norm_err = plt.Normalize(vmin=0.0, vmax=max(err_vmax, 1e-4))

    # Prepare coordinate meshes
    # 1. Horizontal slice: (H, W) at actual_depth
    X_h, Y_h = np.meshgrid(lons, lats)
    Z_h = np.full_like(X_h, actual_depth)

    # 2. Zonal vertical transect wall: (D, W) at actual_lat
    X_zonal, Z_zonal = np.meshgrid(lons, depths)
    Y_zonal = np.full_like(X_zonal, actual_lat)

    # 3. Meridional vertical transect wall: (D, H) at actual_lon
    Y_merid, Z_merid = np.meshgrid(lats, depths)
    X_merid = np.full_like(Y_merid, actual_lon)

    fig = plt.figure(figsize=(20, 6.8), dpi=300)
    fig.patch.set_facecolor('white')

    panels = [
        ("GLORYS12V1 真值 (Ground Truth)", true_field, norm_field, cmap_field, False),
        ("Swin-Ocean-PINN 重构 (Reconstruction)", pred_field, norm_field, cmap_field, False),
        ("绝对重建误差 (Absolute Error)", err_field, norm_err, cmap_err, True)
    ]

    axes = []
    for idx, (p_title, field_data, p_norm, p_cmap, is_err) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        axes.append(ax)

        # Slice 1: Horizontal at z = actual_depth
        val_h = field_data[d_idx, :, :]
        c_h = p_cmap(p_norm(val_h))
        ax.plot_surface(X_h, Y_h, Z_h, facecolors=c_h, shade=False, alpha=0.92, rstride=1, cstride=1)

        # Slice 2: Zonal vertical wall at lat = actual_lat
        val_zonal = field_data[:, h_idx, :]
        c_zonal = p_cmap(p_norm(val_zonal))
        ax.plot_surface(X_zonal, Y_zonal, Z_zonal, facecolors=c_zonal, shade=False, alpha=0.95, rstride=1, cstride=1)

        # Slice 3: Meridional vertical wall at lon = actual_lon
        val_merid = field_data[:, :, w_idx]
        c_merid = p_cmap(p_norm(val_merid))
        ax.plot_surface(X_merid, Y_merid, Z_merid, facecolors=c_merid, shade=False, alpha=0.95, rstride=1, cstride=1)

        # Formatting
        ax.set_title(p_title, fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("经度 Lon (°E)", fontsize=9.5, labelpad=8)
        ax.set_ylabel("纬度 Lat (°N)", fontsize=9.5, labelpad=8)
        ax.set_zlabel("水深 Depth (m)", fontsize=9.5, labelpad=8)

        ax.set_xlim(float(lons.min()), float(lons.max()))
        ax.set_ylim(float(lats.min()), float(lats.max()))
        ax.set_zlim(1000.0, 0.0)  # Inverted depth: surface on top, deep below

        ax.view_init(elev=24, azim=-55)
        ax.xaxis.pane.set_alpha(0.06)
        ax.yaxis.pane.set_alpha(0.06)
        ax.zaxis.pane.set_alpha(0.06)
        ax.grid(True, linestyle=":", alpha=0.4)

    # Colorbars
    sm_field = plt.cm.ScalarMappable(cmap=cmap_field, norm=norm_field)
    sm_field.set_array([])
    cbar_field = fig.colorbar(sm_field, ax=[axes[0], axes[1]], shrink=0.72, aspect=20, pad=0.04)
    cbar_field.set_label(f"{var_label} ({unit_str})", fontsize=10.5)

    sm_err = plt.cm.ScalarMappable(cmap=cmap_err, norm=norm_err)
    sm_err.set_array([])
    cbar_err = fig.colorbar(sm_err, ax=[axes[2]], shrink=0.72, aspect=20, pad=0.08)
    cbar_err.set_label(f"绝对误差 Abs Error ({unit_str})", fontsize=10.5)

    main_title = title or (
        f"太平洋次表层 {var_label} 三维立体正交切片重构效果对比 (0-1000m)\n"
        f"切片位置: 水平水深 z = {actual_depth:.0f}m | 纬向断面 Lat = {actual_lat:.2f}°N | 经向断面 Lon = {actual_lon:.2f}°E"
    )
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_3d_isotherm_surface(
    true_t, pred_t, lons, lats, depths,
    target_temp=15.0,
    save_path="result/pic/fig8_3d_isotherm_15c.png"
):
    """
    Renders the 3D depth topography of a characteristic isothermal surface (e.g. 15 deg C thermocline).
    Shows True vs. Reconstructed vs. Depth Difference in 3D.
    """
    D, H, W = true_t.shape
    z_true = np.full((H, W), np.nan, dtype=np.float32)
    z_pred = np.full((H, W), np.nan, dtype=np.float32)

    for h in range(H):
        for w in range(W):
            # Interpolate depth of target_temp for true
            t_col_true = true_t[:, h, w]
            if t_col_true.min() <= target_temp <= t_col_true.max():
                for d in range(D - 1):
                    if (t_col_true[d] >= target_temp >= t_col_true[d+1]) or (t_col_true[d] <= target_temp <= t_col_true[d+1]):
                        if t_col_true[d+1] != t_col_true[d]:
                            frac = (target_temp - t_col_true[d]) / (t_col_true[d+1] - t_col_true[d])
                            z_true[h, w] = depths[d] + frac * (depths[d+1] - depths[d])
                        else:
                            z_true[h, w] = depths[d]
                        break

            # Interpolate depth of target_temp for pred
            t_col_pred = pred_t[:, h, w]
            if t_col_pred.min() <= target_temp <= t_col_pred.max():
                for d in range(D - 1):
                    if (t_col_pred[d] >= target_temp >= t_col_pred[d+1]) or (t_col_pred[d] <= target_temp <= t_col_pred[d+1]):
                        if t_col_pred[d+1] != t_col_pred[d]:
                            frac = (target_temp - t_col_pred[d]) / (t_col_pred[d+1] - t_col_pred[d])
                            z_pred[h, w] = depths[d] + frac * (depths[d+1] - depths[d])
                        else:
                            z_pred[h, w] = depths[d]
                        break

    # Fill NaNs with domain median if outside bounds
    med_true = np.nanmedian(z_true) if not np.isnan(np.nanmedian(z_true)) else 250.0
    z_true[np.isnan(z_true)] = med_true
    z_pred[np.isnan(z_pred)] = med_true

    z_diff = z_pred - z_true

    X, Y = np.meshgrid(lons, lats)
    vmin_z = float(np.percentile(z_true, 2))
    vmax_z = float(np.percentile(z_true, 98))
    norm_z = plt.Normalize(vmin=vmin_z, vmax=vmax_z)
    cmap_z = cmocean.cm.deep_r

    fig = plt.figure(figsize=(19, 6.2), dpi=300)

    # Panel 1: True Isotherm Depth Surface
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    c1 = cmap_z(norm_z(z_true))
    ax1.plot_surface(X, Y, z_true, facecolors=c1, shade=False, alpha=0.92, rstride=1, cstride=1)
    ax1.set_title(f"GLORYS12V1 真值 {target_temp}°C 等温面埋深 (m)", fontsize=11.5, fontweight='bold')
    ax1.set_zlim(vmax_z + 30, vmin_z - 30)
    ax1.view_init(elev=28, azim=-60)

    # Panel 2: Predicted Isotherm Depth Surface
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    c2 = cmap_z(norm_z(z_pred))
    ax2.plot_surface(X, Y, z_pred, facecolors=c2, shade=False, alpha=0.92, rstride=1, cstride=1)
    ax2.set_title(f"Swin-Ocean-PINN 重构 {target_temp}°C 等温面埋深 (m)", fontsize=11.5, fontweight='bold')
    ax2.set_zlim(vmax_z + 30, vmin_z - 30)
    ax2.view_init(elev=28, azim=-60)

    # Panel 3: Depth Error Surface
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    norm_diff = plt.Normalize(vmin=-30.0, vmax=30.0)
    cmap_diff = cmocean.cm.balance
    c3 = cmap_diff(norm_diff(z_diff))
    ax3.plot_surface(X, Y, z_diff, facecolors=c3, shade=False, alpha=0.92, rstride=1, cstride=1)
    ax3.set_title(f"{target_temp}°C 等温面埋深误差 ΔZ (m)", fontsize=11.5, fontweight='bold')
    ax3.view_init(elev=28, azim=-60)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("经度 Lon (°E)", fontsize=9)
        ax.set_ylabel("纬度 Lat (°N)", fontsize=9)
        ax.set_zlabel("水深 Depth (m)", fontsize=9)

    fig.suptitle(f"西北太平洋 {target_temp}°C 主跃层特征等温面三维空间拓扑起伏重建", fontsize=13.5, fontweight='bold', y=0.98)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path
