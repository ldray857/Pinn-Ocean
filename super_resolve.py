# -*- coding: utf-8 -*-
"""
High-Resolution Spatial-Vertical Interpolation & Super-Resolution Script
Pinn-Ocean (Swin-Ocean-PINN)

Performs 3-D super-resolution on GLORYS ocean fields and multi-source satellite observations:
- Horizontal: Continuous sub-pixel downscaling (e.g. 2x, 4x spatial magnification)
- Vertical: Continuous depth coordinate evaluation (e.g. 10m or 5m equal-interval regular voxels)
- Methods supported:
    1. 'pinn': Swin-Ocean-PINN neural operator continuous space-depth decoding
    2. 'physics_regularized': Physics-constrained 3D interpolation with deep monotonicity preservation
    3. 'tricubic': Classical 3D cubic spline interpolation
    4. 'trilinear': Classical 3D linear interpolation

Exports publication-grade CF-1.8 NetCDF4 assets natively compatible with ArcGIS Pro 3.x Voxel Layers.
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
from pinn_ocean.models.super_resolution import (
    ContinuousSpaceDepthSuperResolver,
    GLORYS3DInterpolator,
    compute_super_resolution_metrics
)
from pinn_ocean.datasets.ocean_dataset import OceanContinuousDataset
from pinn_ocean.utils import get_result_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pinn-Ocean GLORYS 3-D Field High-Resolution Super-Resolution & Interpolation"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing downloaded NetCDF input datasets"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained Swin-Ocean-PINN model weights (default: auto-detected in result/<tag>/checkpoints/)"
    )
    parser.add_argument(
        "--scale_factor", type=float, default=2.0,
        help="Horizontal spatial super-resolution magnification factor (default: 2.0, e.g. 1/12° -> 1/24°)"
    )
    parser.add_argument(
        "--depth_step", type=float, default=10.0,
        help="Vertical depth interval in meters for regular voxel grid (default: 10.0m, giving 101 layers 0-1000m)"
    )
    parser.add_argument(
        "--method", type=str, default="pinn",
        choices=["pinn", "physics_regularized", "tricubic", "trilinear"],
        help="Super-resolution method: 'pinn' (neural operator), 'physics_regularized', 'tricubic', 'trilinear'"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Optional specific years to include (e.g. --years 2015 2016 2017 2018 2019 2020)"
    )
    parser.add_argument(
        "--mode", type=str, default="test", choices=["train", "val", "test", "all"],
        help="Dataset partition to reconstruct ('train', 'val', 'test', or 'all')"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Destination path for high-res NetCDF (default: result/<tag>/con/pacific_glorys_super_res_3d_<mode>.nc)"
    )
    parser.add_argument(
        "--result_dir", type=str, default="result",
        help="Root result directory (default: result)"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Experiment tag name (default: auto-detected, e.g. 2015_2020)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computing device (cuda or cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    sla_path = os.path.join(args.data_dir, "pacific_sla.nc")
    gt_path = os.path.join(args.data_dir, "pacific_glorys_3d_temp_sal.nc")

    # 1. Load Dataset
    try:
        dataset = OceanContinuousDataset(sla_path, gt_path, years=args.years, mode=args.mode)
    except FileNotFoundError as e:
        print(f"[Error] Required input NetCDF files not found: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Setup directory paths
    res_dirs = get_result_dirs(
        result_dir=args.result_dir,
        tag=args.tag,
        dataset=dataset,
        data_dir=args.data_dir,
        years=args.years
    )

    ckpt_path = args.checkpoint
    tag_ckpt = os.path.join(res_dirs['ckpt_dir'], "swin_ocean_pinn_best.pth")
    if ckpt_path is None:
        ckpt_path = tag_ckpt if os.path.exists(tag_ckpt) else "checkpoints/swin_ocean_pinn_best.pth"

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            res_dirs['con_dir'],
            f"pacific_glorys_super_res_3d_{args.method}_{args.mode}.nc"
        )

    orig_lats = dataset.gt_ds.latitude.values
    orig_lons = dataset.gt_ds.longitude.values
    orig_depths = dataset.depths
    times = dataset.times
    stats = dataset.stats

    # Target high-resolution coordinates
    H_orig, W_orig = len(orig_lats), len(orig_lons)
    H_hr = int(round(H_orig * args.scale_factor))
    W_hr = int(round(W_orig * args.scale_factor))
    target_lats = np.linspace(orig_lats.min(), orig_lats.max(), H_hr, dtype=np.float32)
    target_lons = np.linspace(orig_lons.min(), orig_lons.max(), W_hr, dtype=np.float32)

    if args.depth_step > 0:
        target_depths = np.arange(0.0, 1000.0 + args.depth_step / 2.0, args.depth_step, dtype=np.float32)
    else:
        target_depths = orig_depths.astype(np.float32)
    D_hr = len(target_depths)

    print("=" * 75)
    print("      GLORYS 3-D Ocean Field Continuous Super-Resolution Engine         ")
    print("=" * 75)
    print(f" Method              : {args.method.upper()}")
    print(f" Computing Device    : {device}")
    print(f" Scale Factor        : {args.scale_factor}x ({H_orig}x{W_orig} -> {H_hr}x{W_hr})")
    print(f" Target Resolution   : Lat {len(target_lats)} pts, Lon {len(target_lons)} pts (~{1.0/(12.0*args.scale_factor):.4f}°)")
    print(f" Vertical Depths     : {D_hr} layers (Interval: {args.depth_step:.1f}m, 0-1000m)")
    print(f" Time Steps          : {len(times)} months ({args.mode.upper()})")
    print(f" Target NetCDF       : {os.path.abspath(output_file)}")
    print("=" * 75)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_pred_t_hr = []
    all_pred_s_hr = []

    # Branch A: Swin-Ocean-PINN Neural Operator Super-Resolution
    if args.method == "pinn":
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
            stats = checkpoint.get('stats', stats)
            print(f"--> Loaded model weights from {ckpt_path} (Epoch: {checkpoint.get('epoch', 'N/A')})")
        else:
            print(f"[Warning] Checkpoint {ckpt_path} not found. Operating with initial weights.")

        super_resolver = ContinuousSpaceDepthSuperResolver(model=model, stats=stats, device=device)
        z_tensor = torch.from_numpy(target_depths).to(device)

        print("\nExecuting neural operator continuous space-depth forward pass...")
        with torch.no_grad():
            for step, (x_8ch, _) in enumerate(data_loader, 1):
                t_hr, s_hr = super_resolver.reconstruct_3d_high_res(
                    x_8ch=x_8ch,
                    z_coords=z_tensor,
                    scale_factor=args.scale_factor,
                    target_hw=(H_hr, W_hr),
                    unnormalize=True
                )
                all_pred_t_hr.append(t_hr[0])
                all_pred_s_hr.append(s_hr[0])
                print(f"  [Step {step:02d}/{len(data_loader):02d}] Super-resolved 3D grid for: {str(times[step-1])[:10]}")

    # Branch B: Direct 3D Interpolator (Physics-Regularized, Tricubic, or Trilinear)
    else:
        interpolator = GLORYS3DInterpolator(depths=orig_depths, lats=orig_lats, lons=orig_lons)
        print(f"\nExecuting 3D {args.method} spatial-vertical field interpolation...")
        for step, (_, y_3d) in enumerate(data_loader, 1):
            # Unnormalize original target
            t_orig = y_3d[0, 0].numpy() * stats['std_t'] + stats['mean_t']
            s_orig = y_3d[0, 1].numpy() * stats['std_s'] + stats['mean_s']

            t_hr = interpolator.interpolate_field(
                t_orig, target_depths, target_lats, target_lons,
                method=args.method, var_name='temp'
            )
            s_hr = interpolator.interpolate_field(
                s_orig, target_depths, target_lats, target_lons,
                method=args.method, var_name='sal'
            )
            all_pred_t_hr.append(t_hr)
            all_pred_s_hr.append(s_hr)
            print(f"  [Step {step:02d}/{len(data_loader):02d}] Interpolated 3D grid for: {str(times[step-1])[:10]}")

    all_pred_t_hr = np.stack(all_pred_t_hr, axis=0)  # (T, D_hr, H_hr, W_hr)
    all_pred_s_hr = np.stack(all_pred_s_hr, axis=0)

    # 4. Construct CF-1.8 NetCDF Dataset
    ds_hr = xr.Dataset(
        data_vars={
            "super_res_thetao": (
                ("time", "depth", "latitude", "longitude"),
                all_pred_t_hr,
                {
                    "long_name": f"High-Resolution Reconstructed Potential Temperature ({args.method.upper()})",
                    "standard_name": "sea_water_potential_temperature",
                    "units": "degrees_C"
                }
            ),
            "super_res_so": (
                ("time", "depth", "latitude", "longitude"),
                all_pred_s_hr,
                {
                    "long_name": f"High-Resolution Reconstructed Practical Salinity ({args.method.upper()})",
                    "standard_name": "sea_water_practical_salinity",
                    "units": "psu"
                }
            )
        },
        coords={
            "time": ("time", times, {"standard_name": "time", "axis": "T"}),
            "depth": ("depth", target_depths, {"units": "m", "positive": "down", "standard_name": "depth", "axis": "Z"}),
            "latitude": ("latitude", target_lats, {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
            "longitude": ("longitude", target_lons, {"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        },
        attrs={
            "title": f"Pinn-Ocean GLORYS 3-D Super-Resolution & Regular Voxel Dataset ({args.method.upper()})",
            "institution": "Zhejiang University, School of Earth Sciences",
            "program": "Zeng Xianzi Top-notch Innovation Talent Cultivation Program",
            "scale_factor": f"{args.scale_factor}x",
            "vertical_spacing": f"{args.depth_step}m regular",
            "conventions": "CF-1.8",
            "comment": "Fully compatible with ArcGIS Pro 3.x Multidimensional Voxel Layer."
        }
    )

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    encoding = {
        "super_res_thetao": {"_FillValue": -9999.0, "dtype": "float32"},
        "super_res_so": {"_FillValue": -9999.0, "dtype": "float32"},
        "depth": {"_FillValue": None, "dtype": "float32"},
        "latitude": {"_FillValue": None, "dtype": "float32"},
        "longitude": {"_FillValue": None, "dtype": "float32"},
        "time": {"_FillValue": None, "dtype": "float64"}
    }

    print(f"\nWriting high-resolution dataset to NetCDF4 file: {output_file} ...")
    ds_hr.to_netcdf(output_file, engine="netcdf4", encoding=encoding)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print(f"--> [Success] Super-Resolution Export Complete! File size: {file_size_mb:.2f} MB")
    print(f"    Grid Dimensions: {dict(ds_hr.sizes)}")
    print(f"    Temperature Range: {float(all_pred_t_hr.min()):.2f}°C ~ {float(all_pred_t_hr.max()):.2f}°C")
    print(f"    Salinity Range   : {float(all_pred_s_hr.min()):.2f} PSU ~ {float(all_pred_s_hr.max()):.2f} PSU")


if __name__ == "__main__":
    main()
