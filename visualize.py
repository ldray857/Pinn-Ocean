# -*- coding: utf-8 -*-
"""
Scientific Visualization & 3D Volumetric Pipeline for Pinn-Ocean (Swin-Ocean-PINN)
Orchestrates inference and generates publication-grade 2D/3D oceanographic figures:
1. Fig 1A: 50m-Interval Layer-by-Layer Subsurface Slices (0-1000m Temperature)
2. Fig 1B: 50m-Interval Layer-by-Layer Subsurface Slices (0-1000m Salinity)
3. Fig 2: Full-Depth Vertical Transect Section along Kuroshio Extension (35°N)
4. Fig 3: Layer-by-Layer Subsurface Metric Profiles (RMSE, MAE, R^2 vs Depth)
5. Fig 4: Multi-Station Vertical Profiles (4 Contrasting Ocean Dynamic Regimes)
6. Fig 5: Thermohaline Physical Consistency (T-S Diagram with Isopycnals)
7. Fig 6: Full-Depth Prediction vs Truth Scatter Density (Hexbin with R^2)
8. Fig 7: Mixed Layer Depth (MLD) Inversion Validation
9. Fig 8: 3D Isothermal Surface Topography (15°C Thermocline)
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
from pinn_ocean.utils.metrics import calc_mld, calc_layer_metrics
from pinn_ocean.utils import get_result_dirs
from pinn_ocean.visualization import (
    plot_vertical_profiles,
    plot_multi_station_profiles,
    plot_ts_diagram,
    plot_scatter_density,
    plot_mld_validation,
    plot_3d_thermohaline_box,
    plot_3d_isotherm_surface,
    plot_vertical_section,
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
        "--slice_lat", type=float, default=35.0,
        help="Latitude in degrees North for zonal vertical transects (default: 35.0)"
    )
    parser.add_argument(
        "--slice_lon", type=float, default=155.0,
        help="Longitude in degrees East for meridional vertical transects (default: 155.0)"
    )
    parser.add_argument(
        "--slice_depth", type=float, default=100.0,
        help="Depth in meters for horizontal slice (default: 100.0m, main thermocline)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computing device (cuda or cpu)"
    )
    parser.add_argument(
        "--all", action="store_true", default=True,
        help="Generate all 9 publication-grade figures (default: True)"
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
    all_true_mld, all_pred_mld = [], []

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

            # Mixed Layer Depth (MLD)
            D, H, W = true_t.shape
            for h in range(0, H, 5):
                for w in range(0, W, 5):
                    all_true_mld.append(calc_mld(true_t[:, h, w], depths))
                    all_pred_mld.append(calc_mld(pred_t[:, h, w], depths))

    all_full_preds_t = np.array(all_pred_t_list)  # (T, D, H, W)
    all_full_true_t = np.array(all_true_t_list)
    all_full_preds_s = np.array(all_pred_s_list)
    all_full_true_s = np.array(all_true_s_list)

    all_scatter_true_t = np.concatenate(all_scatter_true_t)
    all_scatter_pred_t = np.concatenate(all_scatter_pred_t)
    all_scatter_true_s = np.concatenate(all_scatter_true_s)
    all_scatter_pred_s = np.concatenate(all_scatter_pred_s)
    all_true_mld = np.array(all_true_mld)
    all_pred_mld = np.array(all_pred_mld)

    # Compute layer metrics across full test partition
    print("\nComputing layer-by-layer evaluation metrics...")
    layer_metrics_t = calc_layer_metrics(all_full_preds_t, all_full_true_t, depths)
    layer_metrics_s = calc_layer_metrics(all_full_preds_s, all_full_true_s, depths)

    # 4. Generate Figures
    # -------------------------------------------------------------
    # [Fig 1A & 1B] Layer-by-Layer Subsurface Evaluation (50m Interval)
    # -------------------------------------------------------------
    layers_50m_dir = os.path.join(out_dir, "layers_50m")
    print(f"\n[1/9] Generating Figure 1A: 50m Interval Depth Layers (Temperature)...")
    fig1a = plot_depth_layers_grid(
        first_step_true_t, first_step_pred_t, lons, lats, depths,
        target_depths=np.arange(0, 1050, 50),
        var_name="temperature",
        save_path=os.path.join(out_dir, "fig1_depth_layers_50m_temp.png"),
        export_individual_dir=layers_50m_dir
    )
    print(f"      --> Saved overview to {fig1a} and individual 50m layers to {layers_50m_dir}/")

    print(f"[2/9] Generating Figure 1B: 50m Interval Depth Layers (Salinity)...")
    fig1b = plot_depth_layers_grid(
        first_step_true_s, first_step_pred_s, lons, lats, depths,
        target_depths=np.arange(0, 1050, 50),
        var_name="salinity",
        save_path=os.path.join(out_dir, "fig1_depth_layers_50m_sal.png"),
        export_individual_dir=layers_50m_dir
    )
    print(f"      --> Saved overview to {fig1b} and individual 50m layers to {layers_50m_dir}/")

    # -------------------------------------------------------------
    # [Fig 2] Vertical Transect Section along Kuroshio Extension (35°N)
    # -------------------------------------------------------------
    print("[3/9] Generating Figure 2: Vertical Transect Section along Kuroshio Extension (35°N)...")
    fig2 = plot_vertical_section(
        first_step_true_t, first_step_pred_t, lons, lats, depths,
        slice_type="lat", slice_val=args.slice_lat, var_name="temperature",
        save_path=os.path.join(out_dir, "fig2_vertical_section_35n.png")
    )
    print(f"      --> Saved to {fig2}")

    # -------------------------------------------------------------
    # [Fig 3] Layer-by-Layer Subsurface Metric Profile Curves
    # -------------------------------------------------------------
    print("[4/9] Generating Figure 3: Layer-by-Layer Error & R^2 Curves (0-1000m)...")
    fig3 = plot_layer_metrics_profile(
        layer_metrics_t, layer_metrics_s, depths,
        save_path=os.path.join(out_dir, "fig3_layer_metrics_depth.png")
    )
    print(f"      --> Saved to {fig3}")

    # -------------------------------------------------------------
    # [Fig 4] Multi-Station Vertical Profiles (4 Regimes)
    # -------------------------------------------------------------
    print("[5/9] Generating Figure 4: Multi-Station Vertical Profiles (4 Regimes Array)...")
    fig4 = plot_multi_station_profiles(
        first_step_true_t, first_step_pred_t,
        first_step_true_s, first_step_pred_s,
        depths, lons, lats,
        save_path=os.path.join(out_dir, "fig4_multi_station_profiles.png")
    )
    print(f"      --> Saved to {fig4}")

    # -------------------------------------------------------------
    # [Fig 5] Thermohaline Physical Consistency (T-S Diagram)
    # -------------------------------------------------------------
    print("[6/9] Generating Figure 5: Temperature-Salinity (T-S) Physical Diagram...")
    fig5 = plot_ts_diagram(
        all_scatter_true_t, all_scatter_pred_t, all_scatter_true_s, all_scatter_pred_s,
        save_path=os.path.join(out_dir, "fig5_ts_diagram.png")
    )
    print(f"      --> Saved to {fig5}")

    # -------------------------------------------------------------
    # [Fig 6] Full-Depth Scatter Density with R^2
    # -------------------------------------------------------------
    print("[7/9] Generating Figure 6: Full-Depth Scatter Density Validation...")
    fig6 = plot_scatter_density(
        all_scatter_true_t, all_scatter_pred_t, all_scatter_true_s, all_scatter_pred_s,
        save_path=os.path.join(out_dir, "fig6_scatter_density.png")
    )
    print(f"      --> Saved to {fig6}")

    # -------------------------------------------------------------
    # [Fig 7] Mixed Layer Depth (MLD) Validation
    # -------------------------------------------------------------
    print("[8/9] Generating Figure 7: Mixed Layer Depth (MLD) Scatter Validation...")
    fig7 = plot_mld_validation(
        all_true_mld, all_pred_mld,
        save_path=os.path.join(out_dir, "fig7_mld_validation.png")
    )
    print(f"      --> Saved to {fig7}")

    # -------------------------------------------------------------
    # [Fig 8] 3D Isothermal Surface Topography (15°C Thermocline)
    # -------------------------------------------------------------
    print("[9/9] Generating Figure 8: 3D Isothermal Surface Topography (15°C Thermocline)...")
    fig8 = plot_3d_isotherm_surface(
        first_step_true_t, first_step_pred_t, lons, lats, depths,
        target_temp=15.0,
        save_path=os.path.join(out_dir, "fig8_3d_isotherm_15c.png")
    )
    print(f"      --> Saved to {fig8}")

    print("\n" + "=" * 75)
    print(f" [SUCCESS] All 9 high-resolution 2D/3D scientific figures exported to:")
    print(f"           {os.path.abspath(out_dir)}/")
    print("=" * 75)

    if args.benchmark:
        print("\nExecuting multi-model superiority benchmark suite (--benchmark enabled)...")
        from benchmark import run_benchmark
        run_benchmark()


if __name__ == "__main__":
    run_visualization()

