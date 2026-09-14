# -*- coding: utf-8 -*-
"""
Dataset module for Pinn-Ocean
Loads multi-source satellite observations and 3-D ocean reanalysis data from NetCDF files.
"""

import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from typing import Optional, List, Union


KNOWN_DATASET_PREFIXES = [
    "pacific_glorys_3d_temp_sal",
    "pacific_sla",
    "pacific_sst",
    "pacific_sss",
    "pacific_wind"
]


def _extract_dataset_prefix(filename_or_path: str) -> str:
    """Extract standard dataset prefix (e.g. pacific_sla) from filename or path."""
    base = os.path.basename(filename_or_path)
    for p in KNOWN_DATASET_PREFIXES:
        if base.startswith(p):
            return p
    clean = base[:-3] if base.endswith(".nc") else base
    parts = clean.split("_")
    non_year = [p for p in parts if not p.isdigit()]
    return "_".join(non_year) if non_year else clean


def _find_matching_file(folder: str, prefix: str, year: Optional[str] = None) -> Optional[str]:
    """Find NetCDF file matching prefix and optional year inside folder."""
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if f.endswith(".nc")]
    if year:
        exact_year_name = f"{prefix}_{year}.nc"
        if exact_year_name in files:
            return os.path.join(folder, exact_year_name)
        year_matches = [f for f in files if f.startswith(prefix) and str(year) in f]
        if year_matches:
            return os.path.join(folder, sorted(year_matches)[0])
    prefix_matches = [f for f in files if f.startswith(prefix)]
    if prefix_matches:
        return os.path.join(folder, sorted(prefix_matches)[0])
    return None


def _resolve_and_load_dataset(target_path: Optional[str], default_filename: str, years: Optional[List[int]] = None) -> Optional[xr.Dataset]:
    """
    Intelligently resolves NetCDF dataset from multiple potential structures:
    1. Direct file path (e.g. data/2015/pacific_sla_2015.nc or data/pacific_sla_2013_2021.nc)
    2. A directory containing the target file matching prefix (e.g. data/2020)
    3. A directory containing yearly subfolders (e.g. 2015/, 2016/, ...) with prefix_{year}.nc
       If multiple year subfolders are detected, they are opened and concatenated along 'time'.

    Returns:
        xr.Dataset or None if file cannot be found.
    """
    if target_path and os.path.isfile(target_path):
        return xr.open_dataset(target_path)

    if target_path and os.path.isdir(target_path):
        check_dir = target_path
    elif target_path:
        check_dir = os.path.dirname(os.path.abspath(target_path))
    else:
        return None

    if not os.path.exists(check_dir):
        return None

    prefix = _extract_dataset_prefix(default_filename)

    # 1. Check for 4-digit yearly subdirectories
    subdirs = []
    try:
        subdirs = [
            d for d in os.listdir(check_dir)
            if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(check_dir, d))
        ]
    except OSError:
        pass

    if subdirs:
        if years:
            year_set = set(int(y) for y in years)
            subdirs = [d for d in subdirs if int(d) in year_set]

        subdirs.sort(key=lambda x: int(x))

        loaded_datasets = []
        for d in subdirs:
            yr_dir = os.path.join(check_dir, d)
            yr_file = _find_matching_file(yr_dir, prefix, year=d)
            if yr_file and os.path.isfile(yr_file):
                loaded_datasets.append(xr.open_dataset(yr_file))

        if len(loaded_datasets) > 1:
            print(f"[Dataset] Concatenating {len(loaded_datasets)} yearly files for '{prefix}' across: {subdirs}")
            return xr.concat(loaded_datasets, dim='time')
        elif len(loaded_datasets) == 1:
            return loaded_datasets[0]

    # 2. Direct file check within check_dir
    direct_match = _find_matching_file(check_dir, prefix)
    if direct_match and os.path.isfile(direct_match):
        return xr.open_dataset(direct_match)

    return None


class OceanContinuousDataset(Dataset):
    """
    Pacific Ocean Thermohaline Dataset Loader:
    Inputs (8 channels):
        0: SST (Sea Surface Temperature)
        1: SLA (Sea Level Anomaly)
        2: SSS (Sea Surface Salinity)
        3: Wind U (Zonal wind component)
        4: Wind V (Meridional wind component)
        5: Longitude (Normalized coordinate)
        6: Latitude (Normalized coordinate)
        7: Month (Cyclic time encoding)
    Labels (2 channels, 3-D):
        0: Potential Temperature (0-1000m)
        1: Practical Salinity (0-1000m)
    """
    def __init__(self, sla_path, gt_path, sst_path=None, sss_path=None, wind_path=None,
                 years=None, mode='train', train_ratio=0.75, val_ratio=0.15):
        super().__init__()
        self.mode = mode
        
        # 1. Load core NetCDF datasets (SLA and 3D Ground Truth)
        self.sla_ds_full = _resolve_and_load_dataset(sla_path, "pacific_sla.nc", years=years)
        self.gt_ds_full = _resolve_and_load_dataset(gt_path, "pacific_glorys_3d_temp_sal.nc", years=years)

        if self.sla_ds_full is None or self.gt_ds_full is None:
            raise FileNotFoundError(
                f"Required data files not found for SLA ({sla_path}) or GLORYS GT ({gt_path}). "
                "Please check file paths or ensure yearly subdirectories exist."
            )

        # 2. Resolve auxiliary satellite datasets (SST, SSS, Wind)
        data_dir = os.path.dirname(os.path.abspath(sla_path)) if os.path.isfile(sla_path) else sla_path
        if not os.path.isdir(data_dir):
            data_dir = os.path.dirname(os.path.abspath(sla_path))

        sst_ds = _resolve_and_load_dataset(sst_path or data_dir, "pacific_sst.nc", years=years)
        sss_ds = _resolve_and_load_dataset(sss_path or data_dir, "pacific_sss.nc", years=years)
        wind_ds = _resolve_and_load_dataset(wind_path or data_dir, "pacific_wind.nc", years=years)

        total_months = len(self.gt_ds_full.time)
        n_train = max(1, int(total_months * train_ratio))
        n_val = max(1, int(total_months * val_ratio)) if total_months > 2 else 0
        if n_train + n_val >= total_months and total_months > 2:
            n_train = total_months - n_val - 1

        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, total_months)

        # 3. Spatially & temporally align observations to GLORYS 3D target grid
        # Align SLA
        sla_aligned_full = self.sla_ds_full.interp(
            time=self.gt_ds_full.time,
            latitude=self.gt_ds_full.latitude,
            longitude=self.gt_ds_full.longitude,
            method="linear",
            kwargs={"fill_value": "extrapolate"}
        )
        sla_all = np.nan_to_num(sla_aligned_full.sla.values, nan=0.0)

        # Align SST
        if sst_ds is not None:
            sst_aligned = sst_ds.interp(
                time=self.gt_ds_full.time,
                latitude=self.gt_ds_full.latitude,
                longitude=self.gt_ds_full.longitude,
                method="linear",
                kwargs={"fill_value": "extrapolate"}
            )
            sst_var = "analysed_sst" if "analysed_sst" in sst_aligned else list(sst_aligned.data_vars.keys())[0]
            sst_all = np.nan_to_num(sst_aligned[sst_var].values, nan=0.0)
            if sst_all.mean() > 100.0:  # Convert Kelvin to Celsius
                sst_all = sst_all - 273.15
        else:
            sst_all = np.nan_to_num(self.gt_ds_full.thetao.values[:, 0, :, :], nan=0.0)

        # Align SSS
        if sss_ds is not None:
            sss_aligned = sss_ds.interp(
                time=self.gt_ds_full.time,
                latitude=self.gt_ds_full.latitude,
                longitude=self.gt_ds_full.longitude,
                method="linear",
                kwargs={"fill_value": "extrapolate"}
            )
            sss_var = "sss" if "sss" in sss_aligned else list(sss_aligned.data_vars.keys())[0]
            sss_all = np.nan_to_num(sss_aligned[sss_var].values, nan=0.0)
        else:
            sss_all = np.nan_to_num(self.gt_ds_full.so.values[:, 0, :, :], nan=0.0)

        # Align Wind U & V
        if wind_ds is not None:
            wind_aligned = wind_ds.interp(
                time=self.gt_ds_full.time,
                latitude=self.gt_ds_full.latitude,
                longitude=self.gt_ds_full.longitude,
                method="linear",
                kwargs={"fill_value": "extrapolate"}
            )
            u_var = "eastward_wind" if "eastward_wind" in wind_aligned else list(wind_aligned.data_vars.keys())[0]
            v_var = "northward_wind" if "northward_wind" in wind_aligned else list(wind_aligned.data_vars.keys())[1]
            wind_u_all = np.nan_to_num(wind_aligned[u_var].values, nan=0.0)
            wind_v_all = np.nan_to_num(wind_aligned[v_var].values, nan=0.0)
        else:
            wind_u_all = np.zeros_like(sla_all)
            wind_v_all = np.zeros_like(sla_all)

        temp_all = np.nan_to_num(self.gt_ds_full.thetao.values, nan=0.0)
        sal_all = np.nan_to_num(self.gt_ds_full.so.values, nan=0.0)

        # 3. Compute normalization statistics strictly from the training partition
        self.stats = {
            'mean_sst': float(sst_all[train_idx].mean()),
            'std_sst': float(sst_all[train_idx].std() + 1e-6),
            'mean_sla': float(sla_all[train_idx].mean()),
            'std_sla': float(sla_all[train_idx].std() + 1e-6),
            'mean_sss': float(sss_all[train_idx].mean()),
            'std_sss': float(sss_all[train_idx].std() + 1e-6),
            'mean_wind_u': float(wind_u_all[train_idx].mean()),
            'std_wind_u': float(wind_u_all[train_idx].std() + 1e-6),
            'mean_wind_v': float(wind_v_all[train_idx].mean()),
            'std_wind_v': float(wind_v_all[train_idx].std() + 1e-6),
            'mean_t': float(temp_all[train_idx].mean()),
            'std_t': float(temp_all[train_idx].std() + 1e-6),
            'mean_s': float(sal_all[train_idx].mean()),
            'std_s': float(sal_all[train_idx].std() + 1e-6),
        }

        # 4. Extract subset for specified mode
        if mode == 'train':
            current_idx = train_idx
        elif mode == 'val':
            current_idx = val_idx
        elif mode == 'test':
            current_idx = test_idx
        elif mode == 'all':
            current_idx = slice(0, total_months)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'train', 'val', 'test', 'all'.")

        self.gt_ds = self.gt_ds_full.isel(time=current_idx)
        self.times = self.gt_ds.time.values
        self.depths = self.gt_ds.depth.values

        self.sst_raw = sst_all[current_idx]
        self.sla_raw = sla_all[current_idx]
        self.sss_raw = sss_all[current_idx]
        self.wind_u_raw = wind_u_all[current_idx]
        self.wind_v_raw = wind_v_all[current_idx]

        # 5. Normalized Spatial Coordinates
        lon_vals = self.gt_ds.longitude.values
        lat_vals = self.gt_ds.latitude.values
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        self.lon_norm = (lon_grid - lon_vals.min()) / (lon_vals.max() - lon_vals.min() + 1e-6)
        self.lat_norm = (lat_grid - lat_vals.min()) / (lat_vals.max() - lat_vals.min() + 1e-6)

        # 6. Normalized Labels
        self.temp_norm = (temp_all[current_idx] - self.stats['mean_t']) / self.stats['std_t']
        self.sal_norm = (sal_all[current_idx] - self.stats['mean_s']) / self.stats['std_s']

        # 7. Normalized Cyclic Month Encoding
        # Uses oceanographic thermal cycle phase: Coldest in Feb (month 2), warmest in Aug (month 8)
        # Bounded in [-1.0, 1.0], completely eliminating out-of-distribution winter extrapolation
        month_vals = np.array([
            float(t.astype('datetime64[M]').astype(int) % 12 + 1)
            for t in self.times
        ])
        self.months_norm = - np.cos(2.0 * np.pi * (month_vals - 2.0) / 12.0)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        sst = (self.sst_raw[idx] - self.stats['mean_sst']) / self.stats['std_sst']
        sla = (self.sla_raw[idx] - self.stats['mean_sla']) / self.stats['std_sla']
        sss = (self.sss_raw[idx] - self.stats['mean_sss']) / self.stats['std_sss']

        std_u = self.stats['std_wind_u']
        wind_u = (self.wind_u_raw[idx] - self.stats['mean_wind_u']) / std_u if std_u > 1e-4 else self.wind_u_raw[idx]

        std_v = self.stats['std_wind_v']
        wind_v = (self.wind_v_raw[idx] - self.stats['mean_wind_v']) / std_v if std_v > 1e-4 else self.wind_v_raw[idx]

        lon = self.lon_norm
        lat = self.lat_norm
        month = np.full_like(sla, self.months_norm[idx])

        # Stack into 8-channel 2D input
        x_8ch = np.stack([sst, sla, sss, wind_u, wind_v, lon, lat, month], axis=0)
        # Stack into 2-channel 3D output: [Temp, Sal]
        y_3d = np.stack([self.temp_norm[idx], self.sal_norm[idx]], axis=0)

        return torch.tensor(x_8ch, dtype=torch.float32), torch.tensor(y_3d, dtype=torch.float32)

    def get_depth_tensor(self):
        return torch.tensor(self.depths, dtype=torch.float32)
