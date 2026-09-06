# -*- coding: utf-8 -*-
"""
Scientific Visualization CLI Pipeline for Pinn-Ocean (Swin-Ocean-PINN)
Orchestrates inference and calls modular plotting functions from pinn_ocean.visualization:
1. Fig 1: Representative Station Vertical Profiles (0-1000m T & S)
2. Fig 2: Thermohaline Physical Consistency (T-S Diagram)
3. Fig 3: Full-Depth Prediction vs Truth Scatter Density (Hexbin with R^2)
4. Fig 4: Mixed Layer Depth (MLD) Inversion Validation Scatter Plot
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs.default_config import ModelConfig
from pinn_ocean.models.swin_ocean_pinn import SwinOceanPINN
from pinn_ocean.datasets.ocean_dataset import OceanContinuousDataset
from pinn_ocean.utils.metrics import calc_mld
from pinn_ocean.visualization import (
    plot_vertical_profiles,
    plot_ts_diagram,
    plot_scatter_density,
    plot_mld_validation
)

# Publication styling defaults
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Scientific Evaluation Figures for Pinn-Ocean.")
    parser.add_argument(
        "--data_dir", type=str,
        default="data/2020" if os.path.exists("data/2020/pacific_sla_2013_2021.nc") else "data",
        help="Directory containing downloaded NetCDF input datasets"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/swin_ocean_pinn_best.pth",
        help="Path to trained Swin-Ocean-PINN model weights"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory where output figure PNGs will be saved"
    )
    parser.add_argument(
        "--mode", type=str, default="test", choices=["train", "val", "test"],
        help="Dataset subset partition to evaluate ('train', 'val', or 'test')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computing device (cuda or cpu)"
    )
    return parser.parse_args()


def run_visualization():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("      Pinn-Ocean Scientific Visualization & Physical Validation    ")
    print("=" * 70)
    print(f" Device     : {device}")
    print(f" Data Dir   : {os.path.abspath(args.data_dir)}")
    print(f" Checkpoint : {os.path.abspath(args.checkpoint)}")
    print(f" Output Dir : {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    sla_path = os.path.join(args.data_dir, "pacific_sla_2013_2021.nc")
    gt_path = os.path.join(args.data_dir, "pacific_glorys_3d_temp_sal_2013_2021.nc")

    if not (os.path.exists(sla_path) and os.path.exists(gt_path)):
        print(f"[Error] Required input NetCDF files not found in {args.data_dir}.", file=sys.stderr)
        sys.exit(1)

    # 1. Load Dataset
    dataset = OceanContinuousDataset(sla_path, gt_path, mode=args.mode)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    stats = dataset.stats
    depths = dataset.depths

    # 2. Load Trained Model
    model_cfg = ModelConfig()
    model = SwinOceanPINN(
        in_channels=model_cfg.in_channels,
        embed_dim=model_cfg.embed_dim,
        window_size=model_cfg.window_size,
        physics_hidden_dim=model_cfg.physics_hidden_dim,
        out_dim=model_cfg.out_dim
    ).to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        stats = checkpoint.get('stats', dataset.stats)
        print(f"--> Loaded model weights from {args.checkpoint} (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"[Warning] Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    model.eval()

    z_raw = dataset.get_depth_tensor().to(device)
    z_norm = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-6)

    # Accumulate evaluation samples across partition
    all_true_t, all_pred_t = [], []
    all_true_s, all_pred_s = [], []
    all_true_mld, all_pred_mld = [], []

    first_step_true_t = None
    first_step_pred_t = None
    first_step_true_s = None
    first_step_pred_s = None

    print("\nExecuting model forward pass across spatial domain...")
    with torch.no_grad():
        for i, (x_8ch, y_3d) in enumerate(loader):
            x_8ch = x_8ch.to(device)
            preds = model(x_8ch, z_norm, sample_idx=None)  # (1, 2, D, H, W)

            # Un-normalize to physical units (°C and PSU)
            pred_t = preds[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            pred_s = preds[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']
            true_t = y_3d[0, 0].numpy() * stats['std_t'] + stats['mean_t']
            true_s = y_3d[0, 1].numpy() * stats['std_s'] + stats['mean_s']

            if i == 0:
                first_step_true_t = true_t
                first_step_pred_t = pred_t
                first_step_true_s = true_s
                first_step_pred_s = pred_s

            # Subsample points (step=8) to keep plotting responsive
            all_true_t.append(true_t.flatten()[::8])
            all_pred_t.append(pred_t.flatten()[::8])
            all_true_s.append(true_s.flatten()[::8])
            all_pred_s.append(pred_s.flatten()[::8])

            # Compute Mixed Layer Depth (MLD) for sampled columns
            D, H, W = true_t.shape
            for h in range(0, H, 6):
                for w in range(0, W, 6):
                    all_true_mld.append(calc_mld(true_t[:, h, w], depths))
                    all_pred_mld.append(calc_mld(pred_t[:, h, w], depths))

    all_true_t = np.concatenate(all_true_t)
    all_pred_t = np.concatenate(all_pred_t)
    all_true_s = np.concatenate(all_true_s)
    all_pred_s = np.concatenate(all_pred_s)
    all_true_mld = np.array(all_true_mld)
    all_pred_mld = np.array(all_pred_mld)

    # 3. Call Modular Visualization Routines
    print("\n[1/4] Generating Figure 1: Representative Station Vertical Profile Comparison...")
    fig1 = plot_vertical_profiles(
        first_step_true_t, first_step_pred_t,
        first_step_true_s, first_step_pred_s,
        depths, save_path=os.path.join(args.output_dir, "fig1_profile_comparison.png")
    )
    print(f"      --> Saved to {fig1}")

    print("[2/4] Generating Figure 2: Temperature-Salinity (T-S) Consistency Diagram...")
    fig2 = plot_ts_diagram(
        all_true_t, all_pred_t, all_true_s, all_pred_s,
        save_path=os.path.join(args.output_dir, "fig2_ts_diagram.png")
    )
    print(f"      --> Saved to {fig2}")

    print("[3/4] Generating Figure 3: Full-Depth Scatter Density Validation with R^2...")
    fig3 = plot_scatter_density(
        all_true_t, all_pred_t, all_true_s, all_pred_s,
        save_path=os.path.join(args.output_dir, "fig3_scatter_density.png")
    )
    print(f"      --> Saved to {fig3}")

    print("[4/4] Generating Figure 4: Mixed Layer Depth (MLD) Scatter Validation...")
    fig4 = plot_mld_validation(
        all_true_mld, all_pred_mld,
        save_path=os.path.join(args.output_dir, "fig4_mld_validation.png")
    )
    print(f"      --> Saved to {fig4}")

    print("\n" + "=" * 70)
    print(f" [SUCCESS] All 4 scientific visualization figures exported to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    run_visualization()
