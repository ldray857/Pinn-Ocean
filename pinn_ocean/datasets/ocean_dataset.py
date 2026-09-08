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
                 mode='train', train_ratio=0.75, val_ratio=0.15):
        super().__init__()
        self.mode = mode
        
        if not (os.path.exists(sla_path) and os.path.exists(gt_path)):
            raise FileNotFoundError(
                f"Required data files not found at {sla_path} or {gt_path}. Please check data path configuration."
            )

        data_dir = os.path.dirname(os.path.abspath(sla_path))
        # Auto-detect auxiliary satellite datasets if not explicitly specified
        if sst_path is None:
            candidate = os.path.join(data_dir, "pacific_sst_2013_2021.nc")
            if os.path.exists(candidate):
                sst_path = candidate

        if sss_path is None:
            candidate = os.path.join(data_dir, "pacific_sss_2013_2021.nc")
            if os.path.exists(candidate):
                sss_path = candidate

        if wind_path is None:
            candidate = os.path.join(data_dir, "pacific_wind_2013_2021.nc")
            if os.path.exists(candidate):
                wind_path = candidate

        # 1. Load NetCDF datasets
        self.sla_ds_full = xr.open_dataset(sla_path)
        self.gt_ds_full = xr.open_dataset(gt_path)

        total_months = len(self.gt_ds_full.time)
        n_train = max(1, int(total_months * train_ratio))
        n_val = max(1, int(total_months * val_ratio)) if total_months > 2 else 0
        if n_train + n_val >= total_months and total_months > 2:
            n_train = total_months - n_val - 1

        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, total_months)

        # 2. Spatially & temporally align observations to GLORYS 3D target grid
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
        if sst_path and os.path.exists(sst_path):
            sst_ds = xr.open_dataset(sst_path)
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
        if sss_path and os.path.exists(sss_path):
            sss_ds = xr.open_dataset(sss_path)
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
        if wind_path and os.path.exists(wind_path):
            wind_ds = xr.open_dataset(wind_path)
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
        self.months_norm = np.array([
            float(t.astype('datetime64[M]').astype(int) % 12 + 1) / 12.0
            for t in self.times
        ])

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
