# -*- coding: utf-8 -*-
"""
Scientific Visualization Pipeline for Pinn-Ocean (Swin-Ocean-PINN)
Orchestrates inference and generates publication-grade oceanographic figures:

01_spatial_layers/ (空间逐层水平切片):
  - Fig01_depth_layers_50m_temp.png: 50m-Interval Subsurface Slices (0-1000m Temperature)
  - Fig02_depth_layers_50m_sal.png: 50m-Interval Subsurface Slices (0-1000m Salinity)

02_vertical_profiles/ (垂向结构与误差廓线):
  - Fig03_layer_metrics_depth.png: Subsurface Metric Profiles (RMSE, MAE, R^2 vs Depth)
  - Fig04_multi_station_profiles.png: Multi-Station Vertical Profiles (4 Ocean Dynamic Regimes)

03_physical_diagnostics/ (物理诊断与相关性统计):
  - Fig05_ts_diagram.png: Thermohaline Consistency (T-S Diagram with Isopycnals)
  - Fig06_scatter_density.png: Full-Depth Prediction vs Truth Scatter Density (Hexbin with R^2)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs.default_config import ModelConfig
from pinn_ocean.models.swin_ocean_pinn import SwinOceanPINN
from pinn_ocean.datasets.ocean_dataset import OceanContinuousDataset
from pinn_ocean.utils.metrics import calc_layer_metrics
from pinn_ocean.utils import get_result_dirs
from pinn_ocean.visualization import (
    plot_multi_station_profiles,
    plot_ts_diagram,
    plot_scatter_density,
    plot_layer_metrics_profile,
    plot_depth_layers_grid
)

# Publication styling defaults
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Scientific & 3D Evaluation Figures for Pinn-Ocean.")
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing downloaded NetCDF input datasets"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained Swin-Ocean-PINN model weights (default: result/<year_tag>/checkpoints/swin_ocean_pinn_best.pth)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory where output figure PNGs will be saved (default: result/<year_tag>/pic/)"
    )
    parser.add_argument(
        "--result_dir", type=str, default="result",
        help="Root result directory (default: result)"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Experiment/year tag name (default: auto-detected, e.g. 2015_2020)"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Optional specific years to include (e.g. --years 2017 2018 2019 2020)"
    )
    parser.add_argument(
        "--mode", type=str, default="test", choices=["train", "val", "test", "all"],
        help="Dataset subset partition to evaluate ('train', 'val', 'test', or 'all')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computing device (cuda or cpu)"
    )
    parser.add_argument(
        "--all", action="store_true", default=True,
        help="Generate all publication-grade figures (default: True)"
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=False,
        help="Also execute multi-model superiority benchmark and generate 4 comparative figures"
    )
    return parser.parse_args()



def run_visualization():
    args = parse_args()
    device = torch.device(args.device)

    sla_path = os.path.join(args.data_dir, "pacific_sla.nc")
    gt_path = os.path.join(args.data_dir, "pacific_glorys_3d_temp_sal.nc")

    # 1. Load Dataset
    try:
        dataset = OceanContinuousDataset(sla_path, gt_path, years=args.years, mode=args.mode)
    except FileNotFoundError as e:
        print(f"[Error] Required input NetCDF files not found in {args.data_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Setup standardized result directory structure: result/<year_tag>/pic/
    res_dirs = get_result_dirs(
        result_dir=args.result_dir,
        tag=args.tag,
        dataset=dataset,
        data_dir=args.data_dir,
        years=args.years
    )

    out_dir = args.output_dir if args.output_dir is not None else res_dirs['pic_dir']
    os.makedirs(out_dir, exist_ok=True)

    # Checkpoint resolution: prioritize result/<year_tag>/checkpoints/
    ckpt_path = args.checkpoint
    tag_ckpt = os.path.join(res_dirs['ckpt_dir'], "swin_ocean_pinn_best.pth")
    if ckpt_path is None:
        ckpt_path = tag_ckpt if os.path.exists(tag_ckpt) else "checkpoints/swin_ocean_pinn_best.pth"

    print("=" * 75)
    print("    Pinn-Ocean Multi-Dimensional & 3D Scientific Visualization Pipeline    ")
    print("=" * 75)
    print(f" Device     : {device}")
    print(f" Data Dir   : {os.path.abspath(args.data_dir)}")
    print(f" Result Tag : {res_dirs['tag']} ({res_dirs['exp_dir']})")
    print(f" Checkpoint : {os.path.abspath(ckpt_path)}")
    print(f" Output Pic : {os.path.abspath(out_dir)}")
    if args.years:
        print(f" Filter Years: {args.years}")
    print("=" * 75)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    stats = dataset.stats
    depths = dataset.depths
    lats = dataset.gt_ds.latitude.values
    lons = dataset.gt_ds.longitude.values

    # 3. Load Trained Model
    model_cfg = ModelConfig()
    model = SwinOceanPINN(
        in_channels=model_cfg.in_channels,
        embed_dim=model_cfg.embed_dim,
        window_size=model_cfg.window_size,
        physics_hidden_dim=model_cfg.physics_hidden_dim,
        out_dim=model_cfg.out_dim
    ).to(device)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        stats = checkpoint.get('stats', dataset.stats)
        print(f"--> Loaded model weights from {ckpt_path} (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"[Warning] Checkpoint {ckpt_path} not found. Running with initialized weights.")

    model.eval()

    z_raw = dataset.get_depth_tensor().to(device)

    # Accumulate evaluation samples across partition
    all_true_t_list, all_pred_t_list = [], []
    all_true_s_list, all_pred_s_list = [], []

    all_scatter_true_t, all_scatter_pred_t = [], []
    all_scatter_true_s, all_scatter_pred_s = [], []

    first_step_true_t = None
    first_step_pred_t = None
    first_step_true_s = None
    first_step_pred_s = None

    print("\nExecuting model forward pass across spatial domain...")
    with torch.no_grad():
        for i, (x_8ch, y_3d) in enumerate(loader):
            x_8ch = x_8ch.to(device)
            preds = model(x_8ch, z_raw, sample_idx=None)  # (1, 2, D, H, W)

            # Un-normalize to physical units (°C and PSU)
            pred_t = preds[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            pred_s = preds[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']
            true_t = y_3d[0, 0].numpy() * stats['std_t'] + stats['mean_t']
            true_s = y_3d[0, 1].numpy() * stats['std_s'] + stats['mean_s']

            all_pred_t_list.append(pred_t)
            all_true_t_list.append(true_t)
            all_pred_s_list.append(pred_s)
            all_true_s_list.append(true_s)

            if i == 0:
                first_step_true_t = true_t
                first_step_pred_t = pred_t
                first_step_true_s = true_s
                first_step_pred_s = pred_s

            # Subsample points (step=8) for dense scatter plots
            all_scatter_true_t.append(true_t.flatten()[::8])
            all_scatter_pred_t.append(pred_t.flatten()[::8])
            all_scatter_true_s.append(true_s.flatten()[::8])
            all_scatter_pred_s.append(pred_s.flatten()[::8])

    all_full_preds_t = np.array(all_pred_t_list)  # (T, D, H, W)
    all_full_true_t = np.array(all_true_t_list)
    all_full_preds_s = np.array(all_pred_s_list)
    all_full_true_s = np.array(all_true_s_list)

    all_scatter_true_t = np.concatenate(all_scatter_true_t)
    all_scatter_pred_t = np.concatenate(all_scatter_pred_t)
    all_scatter_true_s = np.concatenate(all_scatter_true_s)
    all_scatter_pred_s = np.concatenate(all_scatter_pred_s)

    # Compute layer metrics across full test partition
    print("\nComputing layer-by-layer evaluation metrics...")
    layer_metrics_t = calc_layer_metrics(all_full_preds_t, all_full_true_t, depths)
    layer_metrics_s = calc_layer_metrics(all_full_preds_s, all_full_true_s, depths)

    # 4. Generate Figures
    # -------------------------------------------------------------
    # Create categorized subdirectories under pic/
    dir_layers = os.path.join(out_dir, "01_spatial_layers")
    dir_profiles = os.path.join(out_dir, "02_vertical_profiles")
    dir_diagnostics = os.path.join(out_dir, "03_physical_diagnostics")
    os.makedirs(dir_layers, exist_ok=True)
    os.makedirs(dir_profiles, exist_ok=True)
    os.makedirs(dir_diagnostics, exist_ok=True)

    # [01_spatial_layers] Fig01 & Fig02: Subsurface Horizontal Slices (50m Interval)
    # -------------------------------------------------------------
    print(f"\n[1/6] Generating Fig01: 50m Interval Depth Layers (Temperature)...")
    fig1a = plot_depth_layers_grid(
        first_step_true_t, first_step_pred_t, lons, lats, depths,
        target_depths=np.arange(0, 1050, 50),
        var_name="temperature",
        save_path=os.path.join(dir_layers, "Fig01_depth_layers_50m_temp.png")
    )
    print(f"      --> Saved to {fig1a}")

    print(f"[2/6] Generating Fig02: 50m Interval Depth Layers (Salinity)...")
    fig1b = plot_depth_layers_grid(
        first_step_true_s, first_step_pred_s, lons, lats, depths,
        target_depths=np.arange(0, 1050, 50),
        var_name="salinity",
        save_path=os.path.join(dir_layers, "Fig02_depth_layers_50m_sal.png")
    )
    print(f"      --> Saved to {fig1b}")

    # -------------------------------------------------------------
    # [02_vertical_profiles] Fig03: Layer-by-Layer Subsurface Metric Profile Curves
    # -------------------------------------------------------------
    print("[3/6] Generating Fig03: Layer-by-Layer Error & R^2 Curves (0-1000m)...")
    fig3 = plot_layer_metrics_profile(
        layer_metrics_t, layer_metrics_s, depths,
        save_path=os.path.join(dir_profiles, "Fig03_layer_metrics_depth.png")
    )
    print(f"      --> Saved to {fig3}")

    # -------------------------------------------------------------
    # [02_vertical_profiles] Fig04: Multi-Station Vertical Profiles (4 Regimes)
    # -------------------------------------------------------------
    print("[4/6] Generating Fig04: Multi-Station Vertical Profiles (4 Regimes Array)...")
    fig4 = plot_multi_station_profiles(
        first_step_true_t, first_step_pred_t,
        first_step_true_s, first_step_pred_s,
        depths, lons, lats,
        save_path=os.path.join(dir_profiles, "Fig04_multi_station_profiles.png")
    )
    print(f"      --> Saved to {fig4}")

    # -------------------------------------------------------------
    # [03_physical_diagnostics] Fig05: Thermohaline Physical Consistency (T-S Diagram)
    # -------------------------------------------------------------
    print("[5/6] Generating Fig05: Temperature-Salinity (T-S) Physical Diagram...")
    fig5 = plot_ts_diagram(
        all_scatter_true_t, all_scatter_pred_t, all_scatter_true_s, all_scatter_pred_s,
        save_path=os.path.join(dir_diagnostics, "Fig05_ts_diagram.png")
    )
    print(f"      --> Saved to {fig5}")

    # -------------------------------------------------------------
    # [03_physical_diagnostics] Fig06: Full-Depth Scatter Density with R^2
    # -------------------------------------------------------------
    print("[6/6] Generating Fig06: Full-Depth Scatter Density Validation...")
    fig6 = plot_scatter_density(
        all_scatter_true_t, all_scatter_pred_t, all_scatter_true_s, all_scatter_pred_s,
        save_path=os.path.join(dir_diagnostics, "Fig06_scatter_density.png")
    )
    print(f"      --> Saved to {fig6}")

    print("\n" + "=" * 75)
    print(f" [SUCCESS] All 6 scientific figures categorized and exported to:")
    print(f"           - {os.path.abspath(dir_layers)}/ (Fig01, Fig02)")
    print(f"           - {os.path.abspath(dir_profiles)}/ (Fig03, Fig04)")
    print(f"           - {os.path.abspath(dir_diagnostics)}/ (Fig05, Fig06)")
    print("=" * 75)

    if args.benchmark:
        print("\nExecuting multi-model superiority benchmark suite (--benchmark enabled)...")
        from benchmark import run_benchmark
        run_benchmark()


if __name__ == "__main__":
    run_visualization()

