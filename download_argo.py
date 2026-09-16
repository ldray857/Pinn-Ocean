# -*- coding: utf-8 -*-
"""
Argo In-Situ Float Profiling Data Downloader for Pinn-Ocean
Fetches real-world, independent Argo profiling float observations from the international Argo program
using argopy (ERDDAP / Argovis data fetchers).

Target Region: Open Northwest Pacific / Kuroshio Extension (Default: 145°E-165°E, 30°N-40°N, 0-1000m)
Target Temporal Scope: 2020 (2020-01-01 to 2020-12-31)
Output Destination: data/argo/<year>/ (e.g. data/argo/2020/)

Artifacts generated:
1. argo_pacific_<year>.nc: Full-depth in-situ observation NetCDF4 dataset
2. argo_profiles_summary.csv: Profiling metadata table (WMO Platform ID, Cycle, Time, Lon, Lat, Levels)
3. argo_profiles_summary.json: Aggregated spatial and quality statistics
4. argo_spatial_distribution.png: Visual station distribution map across the target basin
"""

import os
import sys
import json
import calendar
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

# Optional visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.default_config import DataConfig


def parse_args():
    cfg = DataConfig()
    parser = argparse.ArgumentParser(
        description="Download in-situ Argo float profiles for model validation using argopy."
    )
    parser.add_argument(
        "--year", type=int, default=2020,
        help="Target observation year to fetch (default: 2020)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/argo",
        help="Root directory for saving Argo data (default: data/argo, automatically saved to data/argo/<year>/)"
    )
    parser.add_argument(
        "--min_lon", type=float, default=cfg.min_lon,
        help=f"Minimum longitude in degrees East (default: {cfg.min_lon})"
    )
    parser.add_argument(
        "--max_lon", type=float, default=cfg.max_lon,
        help=f"Maximum longitude in degrees East (default: {cfg.max_lon})"
    )
    parser.add_argument(
        "--min_lat", type=float, default=cfg.min_lat,
        help=f"Minimum latitude in degrees North (default: {cfg.min_lat})"
    )
    parser.add_argument(
        "--max_lat", type=float, default=cfg.max_lat,
        help=f"Maximum latitude in degrees North (default: {cfg.max_lat})"
    )
    parser.add_argument(
        "--min_depth", type=float, default=0.0,
        help="Minimum pressure/depth in meters/dbar (default: 0.0)"
    )
    parser.add_argument(
        "--max_depth", type=float, default=1000.0,
        help="Maximum pressure/depth in meters/dbar (default: 1000.0m)"
    )
    parser.add_argument(
        "--src", type=str, default="erddap", choices=["erddap", "argovis"],
        help="Data fetcher backend source (default: erddap - Ifremer server)"
    )
    parser.add_argument(
        "--qc_filter", action="store_true", default=True,
        help="Apply Argo standard quality control filtering (QC == 1 or 2, default: True)"
    )
    parser.add_argument(
        "--monthly_chunks", action="store_true", default=True,
        help="Fetch month-by-month to avoid network timeout on large bulk queries (default: True)"
    )
    parser.add_argument(
        "--plot_map", action="store_true", default=True,
        help="Generate visual station distribution map for defense PPT (default: True)"
    )
    return parser.parse_args()


def plot_argo_stations_map(df_summary: pd.DataFrame, bbox: list, year: int, save_path: str):
    """Plot spatial distribution of Argo profiling float stations."""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_facecolor("#EDF4FA")

    min_lon, max_lon, min_lat, max_lat = bbox

    # Plot bounding box boundary
    ax.plot(
        [min_lon, max_lon, max_lon, min_lon, min_lon],
        [min_lat, min_lat, max_lat, max_lat, min_lat],
        'r--', lw=2.0, label="Pinn-Ocean 研究区域 (Kuroshio Extension)"
    )

    if not df_summary.empty:
        # Group by platform to show float trajectories
        platforms = df_summary['platform_number'].unique()
        try:
            cmap = plt.colormaps['tab20']
        except (AttributeError, KeyError):
            cmap = plt.cm.get_cmap('tab20')

        for idx, p_num in enumerate(platforms):
            sub = df_summary[df_summary['platform_number'] == p_num].sort_values('time')
            color = cmap(idx % 20)
            ax.plot(sub['longitude'], sub['latitude'], '-', color=color, alpha=0.5, lw=1.2)
            ax.scatter(
                sub['longitude'], sub['latitude'],
                color=color, s=28, edgecolors='black', linewidth=0.5, alpha=0.85
            )

        ax.set_title(
            f"西北太平洋 {year} 年 Argo 真实物理浮标观测站位与漂移轨迹分布\n"
            f"(共检索到 {len(platforms)} 个有效浮标, {len(df_summary)} 个全深度温度/盐度剖面)",
            fontsize=13, fontweight='bold', pad=12
        )
    else:
        ax.set_title(f"西北太平洋 {year} 年 Argo 浮标站位分布 (空数据集)", fontsize=13, fontweight='bold')

    ax.set_xlim(min_lon - 1.5, max_lon + 1.5)
    ax.set_ylim(min_lat - 1.0, max_lat + 1.0)
    ax.set_xlabel("经度 Longitude (°E)", fontsize=11)
    ax.set_ylabel("纬度 Latitude (°N)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"--> [Visual Map] Saved Argo station distribution map to: {save_path}")


def download_argo_data():
    args = parse_args()

    # Verify argopy availability
    try:
        import argopy
        from argopy import DataFetcher as ArgoDataFetcher
    except ImportError:
        print("\n" + "=" * 75, file=sys.stderr)
        print(" [Error] Python package 'argopy' is not installed!", file=sys.stderr)
        print(" To install it in your conda environment, run:", file=sys.stderr)
        print("     conda activate pinn_ocean", file=sys.stderr)
        print("     pip install argopy", file=sys.stderr)
        print("=" * 75 + "\n", file=sys.stderr)
        sys.exit(1)

    target_dir = os.path.join(args.output_dir, str(args.year))
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 75)
    print("      Pinn-Ocean In-Situ Argo Float Profile Acquisition Pipeline      ")
    print("=" * 75)
    print(f" Target Year   : {args.year}")
    print(f" Spatial BBox  : [{args.min_lon}°E, {args.max_lon}°E] x [{args.min_lat}°N, {args.max_lat}°N]")
    print(f" Depth Range   : {args.min_depth} m to {args.max_depth} m (dbar)")
    print(f" Fetcher Source: {args.src.upper()} (Ifremer / OceanOPS)")
    print(f" Quality Filter: {'QC in [1, 2] (Good data)' if args.qc_filter else 'Raw unflagged'}")
    print(f" Destination   : {os.path.abspath(target_dir)}/")
    print("=" * 75)

    # Set argopy global options
    argopy.set_options(src=args.src, mode="standard")

    monthly_datasets = []
    
    if args.monthly_chunks:
        print(f"\n[1/3] Downloading monthly batches across {args.year} (12 months)...")
        for m in range(1, 13):
            last_day = calendar.monthrange(args.year, m)[1]
            t_start = f"{args.year}-{m:02d}-01"
            t_end = f"{args.year}-{m:02d}-{last_day:02d}"

            box = [
                args.min_lon, args.max_lon,
                args.min_lat, args.max_lat,
                args.min_depth, args.max_depth,
                t_start, t_end
            ]

            try:
                fetcher = ArgoDataFetcher(src=args.src).region(box)
                ds_m = fetcher.to_xarray()
                if ds_m is not None and ds_m.sizes.get("N_POINTS", 0) > 0:
                    if args.qc_filter:
                        try:
                            ds_m = ds_m.argo.filter_qc()
                        except Exception as qc_err:
                            pass
                    monthly_datasets.append(ds_m)
                    print(f"  --> {args.year}-{m:02d}: Fetched {ds_m.sizes.get('N_POINTS', 0)} in-situ observation points")
                else:
                    print(f"  --> {args.year}-{m:02d}: No active Argo float profiles found in bounding box")
            except Exception as e:
                print(f"  --> {args.year}-{m:02d}: [Warning] Fetch failed or no data: {e}")
    else:
        print(f"\n[1/3] Downloading full-year block for {args.year}...")
        box = [
            args.min_lon, args.max_lon,
            args.min_lat, args.max_lat,
            args.min_depth, args.max_depth,
            f"{args.year}-01-01", f"{args.year}-12-31"
        ]
        try:
            fetcher = ArgoDataFetcher(src=args.src).region(box)
            ds_full = fetcher.to_xarray()
            if args.qc_filter and ds_full is not None:
                try:
                    ds_full = ds_full.argo.filter_qc()
                except Exception:
                    pass
            if ds_full is not None and ds_full.sizes.get("N_POINTS", 0) > 0:
                monthly_datasets.append(ds_full)
        except Exception as e:
            print(f"[Error] Failed to fetch full year {args.year}: {e}", file=sys.stderr)

    if not monthly_datasets:
        print("\n[Warning] No Argo observation points could be retrieved for the specified criteria.")
        print("Please check your network connection or verify ERDDAP status.")
        return

    # Combine monthly datasets
    import xarray as xr
    if len(monthly_datasets) == 1:
        ds_combined = monthly_datasets[0]
    else:
        # Concatenate along N_POINTS
        ds_combined = xr.concat(monthly_datasets, dim="N_POINTS")

    total_points = ds_combined.sizes.get("N_POINTS", 0)
    print(f"\n[2/3] Total points retrieved across {args.year}: {total_points}")

    # Export NetCDF dataset
    nc_out_path = os.path.join(target_dir, f"argo_pacific_{args.year}.nc")
    ds_combined.to_netcdf(nc_out_path)
    print(f"  --> [Saved NetCDF] CF-1.8 Argo in-situ dataset: {nc_out_path}")

    # Extract profiling summary table
    print(f"\n[3/3] Generating profile metadata index and spatial distribution...")
    summary_records = []

    try:
        # Convert to pandas for metadata aggregation
        df_pts = ds_combined[[
            "PLATFORM_NUMBER", "CYCLE_NUMBER", "TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL"
        ]].to_dataframe().reset_index()

        # Group by platform and cycle to extract distinct vertical casts
        grp = df_pts.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"])
        for (p_num, c_num), group in grp:
            summary_records.append({
                "platform_number": int(p_num) if str(p_num).isdigit() else str(p_num),
                "cycle_number": int(c_num) if str(c_num).isdigit() else str(c_num),
                "time": str(group["TIME"].iloc[0]),
                "longitude": round(float(group["LONGITUDE"].iloc[0]), 4),
                "latitude": round(float(group["LATITUDE"].iloc[0]), 4),
                "min_pressure_dbar": round(float(group["PRES"].min()), 1),
                "max_pressure_dbar": round(float(group["PRES"].max()), 1),
                "valid_obs_levels": int(len(group)),
                "mean_temp_c": round(float(group["TEMP"].mean()), 3) if "TEMP" in group else None,
                "mean_psal_psu": round(float(group["PSAL"].mean()), 3) if "PSAL" in group else None
            })
    except Exception as e:
        print(f"[Warning] Failed to group profiles: {e}")

    df_summary = pd.DataFrame(summary_records)
    csv_out_path = os.path.join(target_dir, "argo_profiles_summary.csv")
    df_summary.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
    print(f"  --> [Saved CSV] Profile metadata table ({len(df_summary)} profiles): {csv_out_path}")

    # Aggregated JSON summary
    json_out_path = os.path.join(target_dir, "argo_profiles_summary.json")
    stats_dict = {
        "year": args.year,
        "spatial_bbox": [args.min_lon, args.max_lon, args.min_lat, args.max_lat],
        "depth_range_m": [args.min_depth, args.max_depth],
        "total_observation_points": int(total_points),
        "total_vertical_profiles": int(len(df_summary)),
        "unique_argo_platforms": int(df_summary["platform_number"].nunique()) if not df_summary.empty else 0,
        "platform_wmo_list": df_summary["platform_number"].unique().tolist() if not df_summary.empty else [],
        "source": args.src,
        "download_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(json_out_path, "w", encoding="utf-8") as f_json:
        json.dump(stats_dict, f_json, indent=2, ensure_ascii=False)
    print(f"  --> [Saved JSON] Statistical overview: {json_out_path}")

    # Plot station distribution map
    if args.plot_map:
        map_path = os.path.join(target_dir, "argo_spatial_distribution.png")
        plot_argo_stations_map(
            df_summary=df_summary,
            bbox=[args.min_lon, args.max_lon, args.min_lat, args.max_lat],
            year=args.year,
            save_path=map_path
        )

    print("\n" + "=" * 75)
    print(" [SUCCESS] Argo In-Situ Observation Download & Processing Completed!")
    print(f"           Asset directory: {os.path.abspath(target_dir)}/")
    print("=" * 75)


if __name__ == "__main__":
    download_argo_data()
