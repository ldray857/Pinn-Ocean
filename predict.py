# -*- coding: utf-8 -*-
"""
Full 3-D Ocean Thermohaline Reconstruction & NetCDF Export Script
Pinn-Ocean (Swin-Ocean-PINN)

Performs full-grid 3D subsurface temperature and salinity reconstruction
and exports results as standard CF-compliant NetCDF4 files for GIS/oceanographic analysis.
"""

import os
import sys
import argparse
import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from configs.default_config import ModelConfig
from pinn_ocean.models.swin_ocean_pinn import SwinOceanPINN
from pinn_ocean.datasets.ocean_dataset import OceanContinuousDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3-D Pacific Ocean Thermohaline Fields and Export to NetCDF."
    )
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
        "--output_file", type=str, default="data/2020/pacific_reconstructed_3d.nc",
        help="Path where reconstructed NetCDF will be saved"
    )
    parser.add_argument(
        "--mode", type=str, default="test", choices=["train", "val", "test", "all"],
        help="Dataset subset partition to reconstruct ('train', 'val', 'test', or 'all')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computing device (cuda or cpu)"
    )
    return parser.parse_args()


def predict_and_export():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("      Swin-Ocean-PINN 3-D Thermohaline Field Reconstruction       ")
    print("=" * 70)
    print(f" Computing Device: {device}")
    print(f" Input Data Dir  : {os.path.abspath(args.data_dir)}")
    print(f" Model Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f" Output NC Target: {os.path.abspath(args.output_file)}")
    print(f" Reconstruction  : {args.mode.upper()} partition")
    print("=" * 70)

    sla_path = os.path.join(args.data_dir, "pacific_sla_2013_2021.nc")
    gt_path = os.path.join(args.data_dir, "pacific_glorys_3d_temp_sal_2013_2021.nc")

    if not (os.path.exists(sla_path) and os.path.exists(gt_path)):
        print(f"[Error] Required input NetCDF files not found in {args.data_dir}.", file=sys.stderr)
        print(f"Expected: {sla_path} and {gt_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Dataset
    mode_arg = 'test' if args.mode == 'all' else args.mode
    dataset = OceanContinuousDataset(sla_path, gt_path, mode=mode_arg)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    stats = dataset.stats
    depths = dataset.depths
    latitudes = dataset.gt_ds.latitude.values
    longitudes = dataset.gt_ds.longitude.values
    times = dataset.times

    # 2. Initialize Model & Load Trained Weights
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
        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"--> Loaded model weights from {args.checkpoint} (Epoch: {epoch}, Val Loss: {val_loss})")
    else:
        print(f"[Notice] Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    model.eval()

    # 3. Continuous Depth Coordinate Tensor
    z_raw = dataset.get_depth_tensor().to(device)
    z_norm = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-6)

    all_pred_thetao = []
    all_pred_so = []
    all_true_thetao = []
    all_true_so = []

    print("\nStarting full-grid 3D spatial-vertical forward pass...")
    with torch.no_grad():
        for step, (x_8ch, y_3d) in enumerate(data_loader, 1):
            x_8ch = x_8ch.to(device)
            # Full grid volumetric reconstruction: (1, 2, D, H, W)
            preds = model(x_8ch, z_norm, sample_idx=None)

            # Un-normalize to physical dimensions: °C and PSU
            pred_t = preds[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            pred_s = preds[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']
            true_t = y_3d[0, 0].numpy() * stats['std_t'] + stats['mean_t']
            true_s = y_3d[0, 1].numpy() * stats['std_s'] + stats['mean_s']

            all_pred_thetao.append(pred_t)
            all_pred_so.append(pred_s)
            all_true_thetao.append(true_t)
            all_true_so.append(true_s)

            print(f"  [Step {step:02d}/{len(data_loader):02d}] Reconstructed 3D field for time step: {str(times[step-1])[:10]}")

    all_pred_thetao = np.stack(all_pred_thetao, axis=0)  # (T, D, H, W)
    all_pred_so = np.stack(all_pred_so, axis=0)
    all_true_thetao = np.stack(all_true_thetao, axis=0)
    all_true_so = np.stack(all_true_so, axis=0)

    # 4. Construct CF-Compliant xarray Dataset
    out_ds = xr.Dataset(
        data_vars={
            "reconstructed_thetao": (
                ("time", "depth", "latitude", "longitude"),
                all_pred_thetao,
                {
                    "long_name": "Reconstructed Sea Water Potential Temperature (Swin-Ocean-PINN)",
                    "standard_name": "sea_water_potential_temperature",
                    "units": "degrees_C",
                    "_FillValue": -9999.0
                }
            ),
            "reconstructed_so": (
                ("time", "depth", "latitude", "longitude"),
                all_pred_so,
                {
                    "long_name": "Reconstructed Sea Water Practical Salinity (Swin-Ocean-PINN)",
                    "standard_name": "sea_water_practical_salinity",
                    "units": "psu",
                    "_FillValue": -9999.0
                }
            ),
            "ground_truth_thetao": (
                ("time", "depth", "latitude", "longitude"),
                all_true_thetao,
                {
                    "long_name": "GLORYS12V1 Reference Potential Temperature",
                    "units": "degrees_C",
                    "_FillValue": -9999.0
                }
            ),
            "ground_truth_so": (
                ("time", "depth", "latitude", "longitude"),
                all_true_so,
                {
                    "long_name": "GLORYS12V1 Reference Practical Salinity",
                    "units": "psu",
                    "_FillValue": -9999.0
                }
            )
        },
        coords={
            "time": times,
            "depth": ("depth", depths, {"units": "m", "positive": "down", "standard_name": "depth"}),
            "latitude": ("latitude", latitudes, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": ("longitude", longitudes, {"units": "degrees_east", "standard_name": "longitude"})
        },
        attrs={
            "title": "Pinn-Ocean 3-D Pacific Ocean Thermohaline Reconstruction",
            "institution": "Zhejiang University, School of Earth Sciences",
            "program": "Zeng Xianzi Top-notch Innovation Talent Cultivation Program",
            "model": "Swin-Ocean-PINN (Shifted Window Self-Attention & Continuous Depth PINN)",
            "source": "Copernicus Marine Service (CMEMS) Satellite Observations & GLORYS12V1",
            "conventions": "CF-1.8"
        }
    )

    # 5. Export to NetCDF4
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nWriting reconstructed dataset to NetCDF4 file: {args.output_file} ...")
    out_ds.to_netcdf(args.output_file, engine="h5netcdf")

    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"--> [Success] Export complete! File size: {file_size_mb:.2f} MB")
    print(f"    Dimensions: {dict(out_ds.dims)}")
    print(f"    Temperature Range: {float(all_pred_thetao.min()):.2f}°C ~ {float(all_pred_thetao.max()):.2f}°C")
    print(f"    Salinity Range   : {float(all_pred_so.min()):.2f} PSU ~ {float(all_pred_so.max()):.2f} PSU")
    print("=" * 70)


if __name__ == "__main__":
    predict_and_export()
