# -*- coding: utf-8 -*-
"""
Automated Data Acquisition Script for Pinn-Ocean (Swin-Ocean-PINN)
Fetches satellite surface observations and 3-D ocean reanalysis labels from CMEMS
(Copernicus Marine Environment Monitoring Service).

Region: Open Pacific Ocean (Default: Kuroshio Extension Deep Basin, 145°E-165°E, 30°N-40°N)
No land, no islands, 100% valid water grid points.
Time Span: 2013-01-01 to 2021-12-31 (108 months)
"""

import os
import sys
import argparse
from typing import Optional, List


# Default Open Pacific bounding box (100% deep ocean, zero land points)
DEFAULT_MIN_LON = 145.0
DEFAULT_MAX_LON = 165.0
DEFAULT_MIN_LAT = 30.0
DEFAULT_MAX_LAT = 40.0

# 9-Year Time Window: 2013-01-01 to 2021-12-31
DEFAULT_START_TIME = "2013-01-01"
DEFAULT_END_TIME = "2021-12-31"

# Depth range for subsurface thermohaline fields (meters)
# (GLORYS surface begins at 0.494m; using 0.49 avoids boundary warnings)
DEFAULT_MIN_DEPTH = 0.49
DEFAULT_MAX_DEPTH = 1000.0

# CMEMS Dataset Identifiers
DATASET_IDS = {
    # 1. Sea Level Anomaly (DUACS L4 Altimetry, 0.125 deg, Monthly)
    "sla": {
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m",
        "variables": ["sla"],
        "filename": "pacific_sla_2013_2021.nc",
        "description": "Sea Surface Height Anomaly (DUACS L4)"
    },
    # 2. GLORYS12V1 3-D Reanalysis (0.083 deg, Monthly, 0-1000m)
    "glorys_3d": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1M-m",
        "variables": ["thetao", "so"],
        "filename": "pacific_glorys_3d_temp_sal_2013_2021.nc",
        "description": "3-D Potential Temperature and Practical Salinity (GLORYS12V1)"
    },
    # 3. Sea Surface Temperature (OSTIA / Reprocessed L4, Monthly)
    "sst": {
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "variables": ["analysed_sst"],
        "filename": "pacific_sst_2013_2021.nc",
        "description": "Sea Surface Temperature (OSTIA L4)"
    },
    # 4. Sea Surface Salinity (Multi-Observation SMOS/SMAP L4 OI, LOPS-v2025)
    "sss": {
        "dataset_id": "cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",
        "variables": ["sss"],
        "filename": "pacific_sss_2013_2021.nc",
        "description": "Sea Surface Salinity (SMOS/SMAP L4 OI - LOPS-v2025)"
    },
    # 5. Sea Surface Wind (Blended Wind L4, Monthly)
    "wind": {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_P1M",
        "variables": ["eastward_wind", "northward_wind"],
        "filename": "pacific_wind_2013_2021.nc",
        "description": "Sea Surface Wind Vectors U/V (Scatterometer & Model Monthly L4)"
    },
    # 6. In-situ Argo Float Temperature and Salinity Profiles (CORA Delayed Mode 013_001)
    "argo": {
        "dataset_id": "cmems_obs-ins_glo_phy-temp-sal_my_cora_irr",
        "variables": ["TEMP", "PSAL", "PRES"],
        "filename": "pacific_argo_cora_2013_2021.nc",
        "description": "In-situ Argo Temperature & Salinity Discrete Profiles (CORA 013_001)"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Open Pacific multi-source ocean observations & 3-D reanalysis from CMEMS."
    )
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save downloaded NetCDF files (default: data)")
    parser.add_argument("--targets", nargs="+", default=["sla", "glorys_3d"],
                        choices=["all", "sla", "glorys_3d", "sst", "sss", "wind", "argo"],
                        help="Datasets to download. 'sla' and 'glorys_3d' are core required datasets.")
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
                        help=f"Minimum depth in meters for 3D products (default: {DEFAULT_MIN_DEPTH})")
    parser.add_argument("--max_depth", type=float, default=DEFAULT_MAX_DEPTH,
                        help=f"Maximum depth in meters for 3D products (default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--username", type=str, default=None,
                        help="CMEMS account username (optional if already logged in via copernicusmarine login)")
    parser.add_argument("--password", type=str, default=None,
                        help="CMEMS account password")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print download plan and parameters without making network calls.")
    return parser.parse_args()


def check_dependencies():
    """Verify copernicusmarine library availability."""
    try:
        import copernicusmarine
        return True, copernicusmarine
    except ImportError:
        return False, None


def download_dataset(cm_module, key, meta, args):
    """Download a single dataset subset."""
    output_path = os.path.join(args.output_dir, meta["filename"])
    print(f"\n[{key.upper()}] {meta['description']}")
    print(f"  Dataset ID : {meta['dataset_id']}")
    print(f"  Variables  : {meta['variables']}")
    print(f"  Output File: {output_path}")

    if os.path.exists(output_path):
        print(f"  [Info] File already exists at {output_path}. Skipping.")
        return True

    kwargs = {
        "dataset_id": meta["dataset_id"],
        "variables": meta["variables"],
        "minimum_longitude": args.min_lon,
        "maximum_longitude": args.max_lon,
        "minimum_latitude": args.min_lat,
        "maximum_latitude": args.max_lat,
        "start_datetime": args.start_time,
        "end_datetime": args.end_time,
        "output_directory": args.output_dir,
        "output_filename": meta["filename"],
        "overwrite": False
    }

    # Depth constraints for 3-D products
    if key == "glorys_3d":
        kwargs["minimum_depth"] = args.min_depth
        kwargs["maximum_depth"] = args.max_depth

    if args.username and args.password:
        kwargs["username"] = args.username
        kwargs["password"] = args.password

    if args.dry_run:
        print("  [DRY RUN] Would execute copernicusmarine.subset with parameters:")
        for k, v in kwargs.items():
            if k not in ["password"]:
                print(f"    - {k}: {v}")
        return True

    try:
        print("  Starting download from Copernicus Marine Data Store...")
        cm_module.subset(**kwargs)
        print(f"  --> Successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"  [Error] Failed to download {key}: {e}", file=sys.stderr)
        return False


def main():
    args = parse_args()

    print("=" * 70)
    print("      Pinn-Ocean Open Pacific Data Collection Tool (CMEMS)       ")
    print("=" * 70)
    print(f" Target Region : {args.min_lon}°E - {args.max_lon}°E, {args.min_lat}°N - {args.max_lat}°N (Pure Open Ocean)")
    print(f" Temporal Range: {args.start_time} to {args.end_time} (9 Years / 108 Months)")
    print(f" Depth Range   : {args.min_depth}m - {args.max_depth}m (Subsurface 3-D)")
    print(f" Output Folder : {os.path.abspath(args.output_dir)}")
    print(f" Dry Run Mode  : {'ENABLED (No network request)' if args.dry_run else 'DISABLED'}")
    print("=" * 70)

    # Determine targets
    target_keys = list(DATASET_IDS.keys()) if "all" in args.targets else args.targets

    if not args.dry_run:
        has_lib, cm_module = check_dependencies()
        if not has_lib:
            print("\n[Error] 'copernicusmarine' library is not installed.")
            print("Please install it using: pip install copernicusmarine")
            print("And optionally log in using: copernicusmarine login")
            print("\nTo preview download parameters without connecting, run with --dry_run:")
            print(f"  python {os.path.basename(__file__)} --dry_run")
            sys.exit(1)
    else:
        cm_module = None

    os.makedirs(args.output_dir, exist_ok=True)

    success_count = 0
    for key in target_keys:
        if key in DATASET_IDS:
            ok = download_dataset(cm_module, key, DATASET_IDS[key], args)
            if ok:
                success_count += 1

    print("\n" + "=" * 70)
    print(f" Collection summary: {success_count}/{len(target_keys)} datasets processed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
