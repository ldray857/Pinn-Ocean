# -*- coding: utf-8 -*-
"""
Argo In-situ Profile Acquisition Tool for Pinn-Ocean (Swin-Ocean-PINN)
Fetches quality-controlled discrete in-situ CTD temperature and salinity profiles 
from the International Argo float network via Copernicus Marine Service (CMEMS).

Product ID : INSITU_GLO_PHY_TS_DISCRETE_MY_013_001
Title      : Global Ocean - CORA - In-situ Observations Yearly Delivery in Delayed Mode
Target Area: Open Pacific Ocean (Kuroshio Extension Basin: 145°E-165°E, 30°N-40°N)
Temporal   : 2013-01-01 to 2021-12-31 (Matches 108-month modeling window)
Vertical   : 0.0m to 1000.0m subsurface water column
"""

import os
import sys
import argparse
from typing import Optional, List


# Default Open Pacific bounding box (100% deep ocean)
DEFAULT_MIN_LON = 145.0
DEFAULT_MAX_LON = 165.0
DEFAULT_MIN_LAT = 30.0
DEFAULT_MAX_LAT = 40.0

# 9-Year Time Window: 2013-01-01 to 2021-12-31
DEFAULT_START_TIME = "2013-01-01"
DEFAULT_END_TIME = "2021-12-31"

# Depth range for subsurface verification (0-1000m)
DEFAULT_MIN_DEPTH = 0.0
DEFAULT_MAX_DEPTH = 1000.0

# CMEMS In-Situ CORA Discrete Dataset Identifiers
ARGO_DATASETS = {
    "cora": {
        "dataset_id": "cmems_obs-ins_glo_phy-temp-sal_my_cora_irr",
        "default_filename": "pacific_argo_cora_2013_2021.nc",
        "variables": ["TEMP", "PSAL", "PRES"],
        "description": "Standard CORA Delayed-Mode In-Situ Profiles (TEMP, PSAL, PRES)"
    },
    "easycora": {
        "dataset_id": "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr",
        "default_filename": "pacific_argo_easycora_2013_2021.nc",
        "variables": ["TEMP", "PSAL", "PRES"],
        "description": "EasyCORA Simplified User-Friendly Profile Format"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Open Pacific In-situ Argo CTD profiles from CMEMS CORA (013_001)."
    )
    parser.add_argument("--output_dir", type=str, default="data/argo",
                        help="Directory to save downloaded NetCDF profile files (default: data/argo)")
    parser.add_argument("--dataset_type", type=str, default="cora", choices=["cora", "easycora"],
                        help="CORA dataset format to download ('cora' or 'easycora', default: cora)")
    parser.add_argument("--output_filename", type=str, default=None,
                        help="Custom output NetCDF filename (default: auto-assigned)")
    parser.add_argument("--min_lon", type=float, default=DEFAULT_MIN_LON,
                        help=f"Minimum longitude in degrees east (default: {DEFAULT_MIN_LON})")
    parser.add_argument("--max_lon", type=float, default=DEFAULT_MAX_LON,
                        help=f"Maximum longitude in degrees east (default: {DEFAULT_MAX_LON})")
    parser.add_argument("--min_lat", type=float, default=DEFAULT_MIN_LAT,
                        help=f"Minimum latitude in degrees north (default: {DEFAULT_MIN_LAT})")
    parser.add_argument("--max_lat", type=float, default=DEFAULT_MAX_LAT,
                        help=f"Maximum latitude in degrees north (default: {DEFAULT_MAX_LAT})")
    parser.add_argument("--start_time", type=str, default=DEFAULT_START_TIME,
                        help=f"Start datetime YYYY-MM-DD (default: {DEFAULT_START_TIME})")
    parser.add_argument("--end_time", type=str, default=DEFAULT_END_TIME,
                        help=f"End datetime YYYY-MM-DD (default: {DEFAULT_END_TIME})")
    parser.add_argument("--min_depth", type=float, default=DEFAULT_MIN_DEPTH,
                        help=f"Minimum depth in meters (default: {DEFAULT_MIN_DEPTH})")
    parser.add_argument("--max_depth", type=float, default=DEFAULT_MAX_DEPTH,
                        help=f"Maximum depth in meters (default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--username", type=str, default=None,
                        help="CMEMS account username (optional if already logged in)")
    parser.add_argument("--password", type=str, default=None,
                        help="CMEMS account password")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print subset query parameters without executing network download")
    return parser.parse_args()


def check_dependencies():
    """Verify copernicusmarine availability."""
    try:
        import copernicusmarine
        return True, copernicusmarine
    except ImportError:
        return False, None


def download_argo_profiles(args):
    target_meta = ARGO_DATASETS[args.dataset_type]
    out_name = args.output_filename if args.output_filename else target_meta["default_filename"]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, out_name)

    print("=" * 70)
    print("      Pinn-Ocean Argo Float In-Situ Verification Data Fetcher     ")
    print("=" * 70)
    print(f" Target Product : Global Ocean CORA Delayed Mode (INSITU_GLO_PHY_TS_DISCRETE_MY_013_001)")
    print(f" Dataset ID     : {target_meta['dataset_id']}")
    print(f" Variables      : {target_meta['variables']}")
    print(f" Target Region  : {args.min_lon}°E - {args.max_lon}°E, {args.min_lat}°N - {args.max_lat}°N")
    print(f" Time Range     : {args.start_time} to {args.end_time}")
    print(f" Depth Range    : {args.min_depth}m - {args.max_depth}m")
    print(f" Output Path    : {os.path.abspath(out_path)}")
    print(f" Dry Run Mode   : {'ENABLED' if args.dry_run else 'DISABLED'}")
    print("=" * 70)

    if os.path.exists(out_path):
        print(f"[Info] File already exists at {out_path}. Skipping.")
        return True

    kwargs = {
        "dataset_id": target_meta["dataset_id"],
        "variables": target_meta["variables"],
        "minimum_longitude": args.min_lon,
        "maximum_longitude": args.max_lon,
        "minimum_latitude": args.min_lat,
        "maximum_latitude": args.max_lat,
        "start_datetime": args.start_time,
        "end_datetime": args.end_time,
        "minimum_depth": args.min_depth,
        "maximum_depth": args.max_depth,
        "output_directory": args.output_dir,
        "output_filename": out_name,
        "force_download": False,
        "overwrite": False
    }

    if args.username and args.password:
        kwargs["username"] = args.username
        kwargs["password"] = args.password

    if args.dry_run:
        print("[DRY RUN] Prepared copernicusmarine.subset parameters:")
        for k, v in kwargs.items():
            if k not in ["password"]:
                print(f"  - {k}: {v}")
        print("\nDry run completed successfully.")
        return True

    has_lib, cm_module = check_dependencies()
    if not has_lib:
        print("\n[Error] 'copernicusmarine' library is not installed in current environment.", file=sys.stderr)
        print("Please install it: pip install copernicusmarine", file=sys.stderr)
        sys.exit(1)

    try:
        print("Starting in-situ Argo profile extraction from Copernicus Marine Data Store...")
        cm_module.subset(**kwargs)
        print(f"--> Successfully downloaded Argo verification profiles to: {out_path}")
        return True
    except Exception as e:
        print(f"[Error] Failed to download Argo in-situ profiles: {e}", file=sys.stderr)
        return False


def main():
    args = parse_args()
    success = download_argo_profiles(args)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
