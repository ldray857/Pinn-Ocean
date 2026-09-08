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
        help="Path where reconstructed NetCDF (aligned with GLORYS) will be saved"
    )
    parser.add_argument(
        "--export_regular", action="store_true", default=True,
        help="Simultaneously export a strictly regular (equal-interval) vertical grid NetCDF for ArcGIS Pro Voxel layer (default: True)"
    )
    parser.add_argument(
        "--no_regular", action="store_false", dest="export_regular",
        help="Disable regular grid export"
    )
    parser.add_argument(
        "--regular_step", type=float, default=10.0,
        help="Vertical depth interval in meters for regular grid (default: 10.0m, giving 101 layers from 0 to 1000m)"
    )
    parser.add_argument(
        "--output_regular_file", type=str, default=None,
        help="Custom output file path for regular NetCDF (default: auto-appends '_regular.nc')"
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
    dataset = OceanContinuousDataset(sla_path, gt_path, mode=args.mode)
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

    # 3. Continuous Depth Coordinate Tensors
    # Mode A: Original GLORYS depths (35 layers, irregular, for ground truth comparison)
    z_raw = dataset.get_depth_tensor().to(device)

    # Mode B: Strictly regular equal-interval depth coordinates (for perfect ArcGIS Pro Voxel rendering)
    if args.export_regular:
        z_reg_vals = np.arange(0.0, 1000.0 + args.regular_step / 2.0, args.regular_step, dtype=np.float32)
        z_reg = torch.from_numpy(z_reg_vals).to(device)
        all_pred_thetao_reg = []
        all_pred_so_reg = []
    else:
        z_reg_vals, z_reg = None, None

    all_pred_thetao = []
    all_pred_so = []
    all_true_thetao = []
    all_true_so = []

    print("\nStarting full-grid 3D spatial-vertical forward pass...")
    with torch.no_grad():
        for step, (x_8ch, y_3d) in enumerate(data_loader, 1):
            x_8ch = x_8ch.to(device)
            # 1. Evaluate on GLORYS-aligned depth grid: (1, 2, D, H, W)
            preds = model(x_8ch, z_raw, sample_idx=None)

            # Un-normalize to physical dimensions: °C and PSU
            pred_t = preds[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            pred_s = preds[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']
            true_t = y_3d[0, 0].numpy() * stats['std_t'] + stats['mean_t']
            true_s = y_3d[0, 1].numpy() * stats['std_s'] + stats['mean_s']

            all_pred_thetao.append(pred_t)
            all_pred_so.append(pred_s)
            all_true_thetao.append(true_t)
            all_true_so.append(true_s)

            # 2. Evaluate on strictly regular equal-interval depth grid
            if args.export_regular:
                preds_reg = model(x_8ch, z_reg, sample_idx=None)
                pred_t_reg = preds_reg[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
                pred_s_reg = preds_reg[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']
                all_pred_thetao_reg.append(pred_t_reg)
                all_pred_so_reg.append(pred_s_reg)

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
                    "units": "degrees_C"
                }
            ),
            "reconstructed_so": (
                ("time", "depth", "latitude", "longitude"),
                all_pred_so,
                {
                    "long_name": "Reconstructed Sea Water Practical Salinity (Swin-Ocean-PINN)",
                    "standard_name": "sea_water_practical_salinity",
                    "units": "psu"
                }
            ),
            "ground_truth_thetao": (
                ("time", "depth", "latitude", "longitude"),
                all_true_thetao,
                {
                    "long_name": "GLORYS12V1 Reference Potential Temperature",
                    "units": "degrees_C"
                }
            ),
            "ground_truth_so": (
                ("time", "depth", "latitude", "longitude"),
                all_true_so,
                {
                    "long_name": "GLORYS12V1 Reference Practical Salinity",
                    "units": "psu"
                }
            )
        },
        coords={
            "time": ("time", times, {"standard_name": "time", "axis": "T"}),
            "depth": ("depth", depths, {"units": "m", "positive": "down", "standard_name": "depth", "axis": "Z"}),
            "latitude": ("latitude", latitudes, {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
            "longitude": ("longitude", longitudes, {"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
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

    # 5. Export to NetCDF4 (Fully compatible with ArcGIS Pro Voxel Layer & Multidimensional Raster)
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Encoding configuration strictly compliant with ArcGIS Pro Voxel requirements:
    # 1. Coordinate dimensions (depth, latitude, longitude) must NOT contain _FillValue attribute
    # 2. Time coordinate must be float64 with standard CF units (avoids int64 incompatibility)
    # 3. Data variables store float32 with standard _FillValue = -9999.0
    encoding = {
        var: {"_FillValue": -9999.0, "dtype": "float32"} for var in out_ds.data_vars
    }
    encoding.update({
        "depth": {"_FillValue": None, "dtype": "float32"},
        "latitude": {"_FillValue": None, "dtype": "float32"},
        "longitude": {"_FillValue": None, "dtype": "float32"},
        "time": {"_FillValue": None, "dtype": "float64"}
    })

    print(f"\nWriting reconstructed dataset to NetCDF4 file: {args.output_file} ...")
    out_ds.to_netcdf(args.output_file, engine="netcdf4", encoding=encoding)

    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"--> [Success] Aligned NetCDF Export complete! File size: {file_size_mb:.2f} MB")
    print(f"    Dimensions: {dict(out_ds.sizes)}")
    print(f"    Temperature Range: {float(all_pred_thetao.min()):.2f}°C ~ {float(all_pred_thetao.max()):.2f}°C")
    print(f"    Salinity Range   : {float(all_pred_so.min()):.2f} PSU ~ {float(all_pred_so.max()):.2f} PSU")

    # 6. Export Strictly Regular Equal-Interval NetCDF4 for ArcGIS Pro Voxel (Zero Warning, True Proportions)
    if args.export_regular and z_reg_vals is not None:
        all_pred_thetao_reg = np.stack(all_pred_thetao_reg, axis=0)
        all_pred_so_reg = np.stack(all_pred_so_reg, axis=0)

        reg_file = args.output_regular_file or (
            args.output_file[:-3] + "_regular.nc" if args.output_file.endswith(".nc") else args.output_file + "_regular.nc"
        )

        reg_ds = xr.Dataset(
            data_vars={
                "reconstructed_thetao": (
                    ("time", "depth", "latitude", "longitude"),
                    all_pred_thetao_reg,
                    {
                        "long_name": "Reconstructed Potential Temperature (Equal-Interval PINN)",
                        "standard_name": "sea_water_potential_temperature",
                        "units": "degrees_C",
                        "actual_range": np.array([float(all_pred_thetao_reg.min()), float(all_pred_thetao_reg.max())], dtype=np.float32)
                    }
                ),
                "reconstructed_so": (
                    ("time", "depth", "latitude", "longitude"),
                    all_pred_so_reg,
                    {
                        "long_name": "Reconstructed Practical Salinity (Equal-Interval PINN)",
                        "standard_name": "sea_water_practical_salinity",
                        "units": "psu",
                        "actual_range": np.array([float(all_pred_so_reg.min()), float(all_pred_so_reg.max())], dtype=np.float32)
                    }
                )
            },
            coords={
                "time": ("time", times, {"standard_name": "time", "axis": "T"}),
                "depth": ("depth", z_reg_vals, {
                    "units": "m",
                    "positive": "down",
                    "standard_name": "depth",
                    "axis": "Z",
                    "step": f"{args.regular_step}m"
                }),
                "latitude": ("latitude", latitudes, {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
                "longitude": ("longitude", longitudes, {"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
            },
            attrs={
                "title": "Pinn-Ocean 3-D Pacific Regular Equal-Interval Thermohaline Reconstruction",
                "institution": "Zhejiang University, School of Earth Sciences",
                "program": "Zeng Xianzi Top-notch Innovation Talent Cultivation Program",
                "model": "Swin-Ocean-PINN (Continuous Depth PINN Representation)",
                "description": f"Strictly equal-interval vertical coordinate (0-1000m, step={args.regular_step}m) optimized for ArcGIS Pro Voxel Layer with zero distortion.",
                "conventions": "CF-1.8"
            }
        )

        reg_encoding = {
            var: {"_FillValue": -9999.0, "dtype": "float32"} for var in reg_ds.data_vars
        }
        reg_encoding.update({
            "depth": {"_FillValue": None, "dtype": "float32"},
            "latitude": {"_FillValue": None, "dtype": "float32"},
            "longitude": {"_FillValue": None, "dtype": "float32"},
            "time": {"_FillValue": None, "dtype": "float64"}
        })

        print(f"\nWriting strictly regular equal-interval dataset to NetCDF4: {reg_file} ...")
        reg_ds.to_netcdf(reg_file, engine="netcdf4", encoding=reg_encoding)

        reg_size_mb = os.path.getsize(reg_file) / (1024 * 1024)
        print(f"--> [Success] Regular Voxel NetCDF Export complete! File size: {reg_size_mb:.2f} MB")
        print(f"    Dimensions: {dict(reg_ds.sizes)}")
        print(f"    Depth Resolution: Strictly regular {args.regular_step}m ({len(z_reg_vals)} layers: 0.0m ~ {z_reg_vals[-1]}m)")
        print(f"    Temperature Range: {float(all_pred_thetao_reg.min()):.2f}°C ~ {float(all_pred_thetao_reg.max()):.2f}°C")
        print(f"    Salinity Range   : {float(all_pred_so_reg.min()):.2f} PSU ~ {float(all_pred_so_reg.max()):.2f} PSU")

    print("=" * 70)


if __name__ == "__main__":
    predict_and_export()
